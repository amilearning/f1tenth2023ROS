#!/usr/bin/env python3
"""   
 Software License Agreement (BSD License)
 Copyright (c) 2022 Ulsan National Institute of Science and Technology (UNIST)
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************** 
  @author: Hojin Lee <hojinlee@unist.ac.kr>
  @date: September 10, 2022
  @copyright 2022 Ulsan National Institute of Science and Technology (UNIST)
  @brief: ROS node for sampling based nonlinear model predictive controller (Model predictive path integrator) 
  @details: main node for MPPI
"""
from re import L
import rospy
import time
import threading

import numpy as np
import math 
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
from vesc_msgs.msg import VescStateStamped

from hmcl_msgs.srv import mpcc
import torch 
import rospkg
from predictor.common.pytypes import VehicleState, ParametricPose, BodyLinearVelocity, VehiclePrediction, VehicleActuation
from predictor.utils import pose_to_vehicleState, odom_to_vehicleState
from predictor.path_generator import PathGenerator
from predictor.prediction.thetapolicy_predictor import ThetaPolicyPredictor
from predictor.controllers.headless_MPCC import MPCC_H2H_approx
from predictor.dynamics.models.dynamics_models import CasadiDynamicBicycleFull
from predictor.h2h_configs import *
rospack = rospkg.RosPack()
pkg_dir = rospack.get_path('predictor')

class Predictor:
    def __init__(self):       
        
        self.n_nodes = rospy.get_param('~n_nodes', default=10)
        self.t_horizon = rospy.get_param('~t_horizon', default=1.0)                   
        self.torch_device = "cuda:0"   ## Specify the name of GPU 
        # self.torch_dtype  = torch.double
        self.dt = self.t_horizon / self.n_nodes*1.0        
        ## 
        # Generate Racing track info 
        self.track_info = PathGenerator()
        
        while self.track_info.track_ready is False:
             rospy.sleep(0.01)
        ##
        gp_model_name = "aggressive_blocking"
        use_GPU = True
        M = 3
        self.predictor = ThetaPolicyPredictor(N=self.n_nodes, track=self.track_info.track, policy_name=gp_model_name, use_GPU=use_GPU, M=M, cov_factor=np.sqrt(2))            
        self.vehicle_model = CasadiDynamicBicycleFull(0.0, ego_dynamics_config, track=self.track_info.track)
        self.gp_mpcc_ego_controller = MPCC_H2H_approx(self.vehicle_model, self.track_info.track, control_params = gp_mpcc_ego_params, name="gp_mpcc_h2h_ego", track_name="test_track")
        self._thread = threading.Thread()        

        self.cur_ego_odom = Odometry()        
        self.cur_ego_pose = PoseStamped()
        self.cur_ego_vehicle_state_msg  =VescStateStamped()

        self.cur_tar_odom = Odometry()
        self.cur_tar_pose = PoseStamped()

        self.cur_ego_state = VehicleState(t=0.0,
                                      p=ParametricPose(s=0.1, x_tran=0.2, e_psi=0.0),
                                      v=BodyLinearVelocity(v_long=0.5),
                                      u=VehicleActuation(t=0.0, u_a=0.0, u_steer = 0.0))
        self.cur_tar_state = VehicleState()
            
        
        ego_odom_topic = "/pose_estimate"
        ego_pose_topic = "/tracked_pose"
        ego_vehicle_status_topic = "/vesc/sensor/core"
        ego_control_topic = "vesc/cmd"

        target_odom_topic = "/target/pose_estimate"
        target_pose_topic = "/target/tracked_pose"
        target_vehicle_status_topic = "/target/sensor/core"
        
        self.ego_odom_ready = False
        self.tar_odom_ready = False
        
        # Service client 
        self.mpcc_srv = rospy.ServiceProxy('compute_mpcc',mpcc)

        # Publishers                
        self.ego_vehicleState_pub = rospy.Publisher(ego_vehicle_status_topic, Bool, queue_size=2)            
        self.target_vehicleState_pub = rospy.Publisher(ego_vehicle_status_topic, Bool, queue_size=2)            
        self.target_predict_pub = rospy.Publisher(ego_vehicle_status_topic, Bool, queue_size=2)            
        
        # Subscribers
        self.ego_odom_sub = rospy.Subscriber(ego_odom_topic, Odometry, self.ego_odom_callback)                        
        self.ego_pose_sub = rospy.Subscriber(ego_pose_topic, PoseStamped, self.ego_pose_callback)                        
        # self.ego_vehicle_status_sub = rospy.Subscriber(ego_vehicle_status_topic, VescStateStamped, self.ego_vehicle_status_callback)                
        
        self.target_odom_sub = rospy.Subscriber(target_odom_topic, Odometry, self.target_odom_callback)                     
        self.target_pose_sub = rospy.Subscriber(target_pose_topic, PoseStamped, self.target_pose_callback)                           
        

        # prediction callback   
        self.tv_pred = None
        self.prediction_hz = rospy.get_param('~prediction_hz', default=0.5)
        self.prediction_timer = rospy.Timer(rospy.Duration(1/self.prediction_hz), self.prediction_callback)         
        ## controller callback
        self.cmd_hz = 0.5
        self.cmd_timer = rospy.Timer(rospy.Duration(1/self.cmd_hz), self.cmd_callback)         
        
        
        rate = rospy.Rate(1)     
        while not rospy.is_shutdown():            
            msg = Bool()
            msg.data = True
            # self.status_pub.publish(msg)          
            rate.sleep()

        

    def ego_odom_callback(self,msg):
        if self.ego_odom_ready is False:
            self.ego_odom_ready = True
        self.cur_ego_odom = msg

    def ego_vehicle_status_callback(self,msg):
        self.cur_ego_vehicle_state_msg = msg
    
    def ego_pose_callback(self,msg):
        self.cur_ego_pose = msg

    def target_odom_callback(self,msg):
        if self.tar_odom_ready is False:
            self.tar_odom_ready = True
        self.cur_tar_odom = msg
    
    def target_pose_callback(self,msg):
        self.cur_tar_pose = msg
    
        
    def prediction_callback(self,event):
        start_time = time.time()
        if self.ego_odom_ready and self.tar_odom_ready:
            
            pose_to_vehicleState(self.track_info.track, self.cur_ego_state, self.cur_ego_pose)
            odom_to_vehicleState(self.cur_ego_state, self.cur_ego_odom)
            
            pose_to_vehicleState(self.track_info.track, self.cur_tar_state, self.cur_tar_pose)
            odom_to_vehicleState(self.cur_tar_state, self.cur_tar_odom)
            
        else:
            rospy.loginfo("state not ready")
            return 
        
        if self.predictor and self.cur_ego_state is not None:            
            ego_pred = self.predictor.get_constant_vel_prediction_par(self.cur_ego_state)
            # print(ego_pred.s)            
            if self.cur_ego_state.t is not None and self.cur_tar_state.t is not None:            
                self.tv_pred = self.predictor.get_prediction(self.cur_ego_state, self.cur_tar_state, ego_pred)
                
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Prediction execution time: {execution_time} seconds")

    def cmd_callback(self,event):
        print("reqeust service ")         
        self.use_predictions_from_module = True
        problem = self.gp_mpcc_ego_controller.step(self.cur_ego_state, tv_state=self.cur_tar_state, tv_pred=self.tv_pred if self.use_predictions_from_module else None)

        try:
            srv_res = self.mpcc_srv(problem["xinit"], problem["all_parameters"], problem["x0"])
            print("got the response")
            print(srv_res)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        
        

###################################################################################

def main():
    rospy.init_node("predictor")    
    Predictor()

if __name__ == "__main__":
    main()




 
    


