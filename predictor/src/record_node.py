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
import pickle
import rospy
import time
import os
import numpy as np 
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray, Marker
import threading
from vesc_msgs.msg import VescStateStamped
from hmcl_msgs.srv import mpcc
import rospkg
from predictor.common.pytypes import VehicleState, ParametricPose, BodyLinearVelocity, VehicleActuation
from predictor.utils import shift_in_local_x, pose_to_vehicleState, odom_to_vehicleState, prediction_to_marker, fill_global_info, compute_local_velocity
from predictor.path_generator import PathGenerator
# from predictor.prediction.thetapolicy_predictor import ThetaPolicyPredictor
# from predictor.prediction.gp_thetapolicy_predictor import GPThetaPolicyPredictor

from predictor.prediction.covGP.covGPNN_predictor import CovGPPredictor

from predictor.prediction.trajectory_predictor import ConstantAngularVelocityPredictor, NLMPCPredictor, GPPredictor
from predictor.h2h_configs import *
from predictor.common.utils.file_utils import *
from predictor.common.utils.scenario_utils import RealData
from predictor.utils import prediction_to_rosmsg, rosmsg_to_prediction, interp_state, prediction_to_std_trace
from predictor.simulation.dynamics_simulator import DynamicsSimulator
from predictor.controllers.MPCC_H2H_approx import MPCC_H2H_approx
from hmcl_msgs.msg import VehiclePredictionROS 
from dynamic_reconfigure.server import Server
from predictor.cfg import predictorDynConfig
from predictor.controllers.utils.controllerTypes import PIDParams
from predictor.controllers.PID import PIDLaneFollower
from predictor.dynamics.models.dynamics_models import CasadiDynamicBicycleFull
from collections import deque
from message_filters import ApproximateTimeSynchronizer, Subscriber

rospack = rospkg.RosPack()
pkg_dir = rospack.get_path('predictor')

class Recorder:
    def __init__(self):       
      
        self.n_nodes = rospy.get_param('~n_nodes', default=10)
        self.t_horizon = rospy.get_param('~t_horizon', default=1.0)                   
        self.torch_device = "cuda:0"   ## Specify the name of GPU 
        # self.torch_dtype  = torch.double
        self.dt = self.t_horizon / self.n_nodes*1.0        
        ## 
        # Generate Racing track info 
        self.track_load = False
        track_file_path = os.path.join(track_dir, 'track.pickle')        
        self.track_info = PathGenerator()
        if self.track_load:
            with open(track_file_path, 'rb') as file:            
                self.track_info = pickle.load(file)
        else: ## save track
            with open(track_file_path, 'wb') as file:
                pickle.dump(self.track_info, file)
        

# Open the file in binary mode for writing

        self.track_info = PathGenerator()
        
        while self.track_info.track_ready is False:
             rospy.sleep(0.01)
        
        

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
        target_odom_topic = "/target/pose_estimate"
        target_pose_topic = "/target/tracked_pose"        
        
        self.ego_odom_ready = False
        self.tar_odom_ready = False

        # prediction callback   
        self.tv_pred = None
        self.tv_cav_pred = None
        self.tv_nmpc_pred = None
        self.tv_gp_pred = None
        self.xy_cov_trace = None 

        ## controller callback        
        self.ego_list = []
        self.tar_list = []
        self.tar_pred_list = []
        
        
        

        self.data_save = False
        self.prev_data_save = False
        self.pred_data_save = False
        self.save_buffer_length = 200

        # Subscribers
        self.ego_odom_sub = rospy.Subscriber(ego_odom_topic, Odometry, self.ego_odom_callback)                        
        # self.ego_pose_sub = rospy.Subscriber(ego_pose_topic, PoseStamped, self.ego_pose_callback)                        
        self.target_odom_sub = rospy.Subscriber(target_odom_topic, Odometry, self.target_odom_callback)                     
        # self.target_pose_sub = rospy.Subscriber(target_pose_topic, PoseStamped, self.target_pose_callback)     

        self.ego_prev_pose = None
        self.tar_prev_pose = None
        self.ego_local_vel = None
        self.tar_local_vel = None
        

        self.ego_pose_sub = Subscriber(ego_pose_topic, PoseStamped)        
        self.target_pose_sub = Subscriber(target_pose_topic, PoseStamped)
        
        self.ats = ApproximateTimeSynchronizer([self.ego_pose_sub,  self.target_pose_sub], queue_size=10, slop=0.05)
        self.sync_prev_time = rospy.Time.now().to_sec()
        self.ats.registerCallback(self.sync_callback)


        # NLMPCPredictor(N, None, cov=.01, v_ref=mpcc_tv_params.vx_max),
        self.dyn_srv = Server(predictorDynConfig, self.dyn_callback)
        self.prediction_hz = rospy.get_param('~prediction_hz', default=10)        
        self.data_logging_hz = rospy.get_param('~data_logging_hz', default=10)
        self.prev_dl_time = rospy.Time.now().to_sec()
        self.prediction_timer = rospy.Timer(rospy.Duration(1/self.data_logging_hz), self.datalogging_callback)         
        
  
        
        
        
        rate = rospy.Rate(1)     
        while not rospy.is_shutdown():            
            msg = Bool()
            msg.data = True
            # self.status_pub.publish(msg)          
            rate.sleep()

 

    def dyn_callback(self,config,level):        
        
        self.pred_data_save = config.logging_prediction_results

        if config.clear_buffer:
            self.clear_buffer()
        self.predictor_type = config.predictor_type
        print("self.predictor_type = " + str(self.predictor_type))
        self.data_save = config.logging_vehicle_states
        if self.prev_data_save is True and self.data_save is False and config.clear_buffer is not True:
            self.save_buffer_in_thread()
            print("save data by turnning off the switch")
        self.prev_data_save = config.logging_vehicle_states
        print("dyn reconfigured")
        
        return config
        
    def clear_buffer(self):
        if len(self.ego_list) > 0:
            self.ego_list.clear()
            self.tar_list.clear()
            self.tar_pred_list.clear()
    
    def save_buffer_in_thread(self):
        # Create a new thread to run the save_buffer function
        
            
        t = threading.Thread(target=self.save_buffer)
        t.start()

        

# @dataclass
# class RealData():
#     track: RadiusArclengthTrack
#     N: int
#     ego_states: List[VehicleState]
#     tar_states: List[VehicleState]

    def save_buffer(self):        
        real_data = RealData(self.track_info.track, len(self.tar_list), self.ego_list, self.tar_list, self.tar_pred_list)
        create_dir(path=real_dir)        
        pickle_write(real_data, os.path.join(real_dir, str(self.cur_ego_state.t) + '_'+ str(len(self.tar_list))+'.pkl'))
        rospy.loginfo("states data saved")
        self.clear_buffer()
        rospy.loginfo("states buffer has been cleaned")

    def sync_callback(self,  ego_pose_topic,  target_pose_topic):
                        # self.ego_odom_sub, self.ego_pose_sub, self.target_odom_sub, self.target_pose_sub
        
        sync_cur_time = rospy.Time.now().to_sec()
        diff_sync_time = sync_cur_time - self.sync_prev_time         
        if abs(diff_sync_time) > 0.05:
            rospy.logwarn("sync diff time " + str(diff_sync_time))
        self.sync_prev_time = sync_cur_time
        
        # if self.ego_prev_pose is not None and self.tar_prev_pose is not None:
        #     if abs(self.ego_prev_pose.header.stamp.to_sec()- ego_pose_topic.header.stamp.to_sec()) > 0.5:
        #         self.ego_local_vel = compute_local_velocity(self.ego_prev_pose, ego_pose_topic)
        #         self.tar_local_vel = compute_local_velocity(self.tar_prev_pose, target_pose_topic)
        #         self.ego_prev_pose = ego_pose_topic
        #         self.tar_prev_pose = target_pose_topic
        #     if self.ego_local_vel is not None:
        #         tmp_debug_msg = PoseStamped()
        #         tmp_debug_msg.header = target_pose_topic.header
        #         tmp_debug_msg.pose.position.x = self.tar_local_vel.x
        #         tmp_debug_msg.pose.position.y = self.tar_local_vel.y
        #         tmp_debug_msg.pose.position.z = self.tar_local_vel.z
        #         self.debug_pub.publish(tmp_debug_msg)

        # else:
        #     self.ego_prev_pose = ego_pose_topic
        #     self.tar_prev_pose = target_pose_topic

        # self.ego_odom_callback(ego_odom_topic)
        self.ego_pose_callback(ego_pose_topic)
        # self.target_odom_callback(target_odom_topic)
        self.target_pose_callback(target_pose_topic)

        

    def ego_odom_callback(self,msg):
        if self.ego_odom_ready is False:
            self.ego_odom_ready = True
        self.cur_ego_odom = msg
        
        # self.cur_ego_pose.header = msg.header 
        # self.cur_ego_pose.pose = msg.pose.pose

    def ego_vehicle_status_callback(self,msg):
        self.cur_ego_vehicle_state_msg = msg

    
    def ego_pose_callback(self,msg):
        self.cur_ego_pose = msg

    def target_odom_callback(self,msg):
        if self.tar_odom_ready is False:
            self.tar_odom_ready = True
        self.cur_tar_odom = msg

        # self.cur_tar_pose.header = msg.header
        # self.cur_tar_pose.pose = msg.pose.pose
        # shift_in_local_x(self.cur_tar_pose, dist = -0.10)
        
    
    def target_pose_callback(self,msg):
        self.cur_tar_pose = msg
        shift_in_local_x(self.cur_tar_pose, dist = -0.01)
    
      
   

     

                        
    def datalogging_callback(self, event):
      
        if self.ego_odom_ready and self.tar_odom_ready:
            pose_to_vehicleState(self.track_info.track, self.cur_ego_state, self.cur_ego_pose)
            odom_to_vehicleState(self.track_info.track, self.cur_ego_state, self.cur_ego_odom)
            
            
            pose_to_vehicleState(self.track_info.track, self.cur_tar_state, self.cur_tar_pose)            
            odom_to_vehicleState(self.track_info.track, self.cur_tar_state, self.cur_tar_odom)
            
            
        else:
            rospy.loginfo("state not ready")
            
        

        if self.data_save:
                if isinstance(self.cur_ego_state.t,float) and isinstance(self.cur_tar_state.t,float) and self.cur_ego_state.p.s is not None and self.cur_tar_state.p.s is not None and abs(self.cur_tar_state.p.x_tran) < self.track_info.track.track_width and abs(self.cur_ego_state.p.x_tran) < self.track_info.track.track_width:
                    if self.pred_data_save:
                        if self.tv_pred is None:                                                 
                            return
                        else:
                            self.tar_pred_list.append(self.tv_pred)
                            
                    
                    self.ego_list.append(self.cur_ego_state.copy())
                    self.tar_list.append(self.cur_tar_state.copy())                     
                    callback_time = self.ego_list[-1].t        
                    delta_time = self.prev_dl_time - callback_time
                    self.prev_dl_time = callback_time                    
                    
                    if len(self.tar_list) > self.save_buffer_length:
                        self.save_buffer_in_thread()
                elif len(self.tar_list) > 0 and len(self.ego_list) > 0: ## if suddent crash or divergence in local, save data and do not continue from the next iteration
                    self.save_buffer_in_thread()   
        
            
            



    
    
###################################################################################

def main():
    rospy.init_node("recorder")    
    Recorder()

if __name__ == "__main__":
    main()




 
    


