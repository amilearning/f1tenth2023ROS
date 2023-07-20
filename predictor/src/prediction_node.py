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
from visualization_msgs.msg import MarkerArray

from vesc_msgs.msg import VescStateStamped
from hmcl_msgs.srv import mpcc
import rospkg
from predictor.common.pytypes import VehicleState, ParametricPose, BodyLinearVelocity, VehicleActuation
from predictor.utils import shift_in_local_x, pose_to_vehicleState, odom_to_vehicleState, prediction_to_marker, fill_global_info
from predictor.path_generator import PathGenerator
from predictor.prediction.thetapolicy_predictor import ThetaPolicyPredictor
from predictor.prediction.trajectory_predictor import ConstantAngularVelocityPredictor, NLMPCPredictor, GPPredictor
from predictor.h2h_configs import *
from predictor.common.utils.file_utils import *
from predictor.common.utils.scenario_utils import RealData
from predictor.utils import prediction_to_rosmsg, rosmsg_to_prediction
from predictor.simulation.dynamics_simulator import DynamicsSimulator
from predictor.controllers.MPCC_H2H_approx import MPCC_H2H_approx
from hmcl_msgs.msg import VehiclePredictionROS 
from dynamic_reconfigure.server import Server
from predictor.cfg import predictorDynConfig
from predictor.controllers.utils.controllerTypes import PIDParams
from predictor.controllers.PID import PIDLaneFollower
from predictor.dynamics.models.dynamics_models import CasadiDynamicBicycleFull
from collections import deque

rospack = rospkg.RosPack()
pkg_dir = rospack.get_path('predictor')

class Predictor:
    def __init__(self):       
        self.dyn_srv = Server(predictorDynConfig, self.dyn_callback)
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
        ##
        gp_model_name = "aggressive_blocking"
        use_GPU = True
        M = 10
        

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
        
        # Service client 
        self.mpcc_srv = rospy.ServiceProxy('compute_mpcc',mpcc)

        # Publishers                
        self.tv_pred_marker_pub = rospy.Publisher('/tv_pred_marker',MarkerArray,queue_size = 2)                             
        self.tv_cav_pred_marker_pub = rospy.Publisher('/tv_cav_pred_marker',MarkerArray,queue_size = 2)         
        self.tv_nmpc_pred_marker_pub = rospy.Publisher('/tv_nmpc_pred_marker',MarkerArray,queue_size = 2)         
        self.tv_gp_pred_marker_pub = rospy.Publisher('/tv_gp_pred_marker',MarkerArray,queue_size = 2)         
        
        
        self.tar_pred_pub = rospy.Publisher("/tar_pred", VehiclePredictionROS, queue_size=2)
        # Subscribers
        self.ego_odom_sub = rospy.Subscriber(ego_odom_topic, Odometry, self.ego_odom_callback)                        
        self.ego_pose_sub = rospy.Subscriber(ego_pose_topic, PoseStamped, self.ego_pose_callback)                        
        self.target_odom_sub = rospy.Subscriber(target_odom_topic, Odometry, self.target_odom_callback)                     
        self.target_pose_sub = rospy.Subscriber(target_pose_topic, PoseStamped, self.target_pose_callback)                           
        
        
        ### setup ego controller to compute the ego prediction 
        self.vehicle_model = CasadiDynamicBicycleFull(0.0, ego_dynamics_config, track=self.track_info.track)
        self.gp_mpcc_ego_controller = MPCC_H2H_approx(self.vehicle_model, self.track_info.track, control_params = gp_mpcc_ego_params, name="gp_mpcc_h2h_ego", track_name="test_track")        
        self.ego_warm_start()
        gp_policy_name = 'gpberkely'        
        self.gp_predictor = GPPredictor(self.n_nodes, self.track_info.track, gp_policy_name, True, M, cov_factor=np.sqrt(2))
        
        ### our method 
        self.predictor = ThetaPolicyPredictor(N=self.n_nodes, track=self.track_info.track, policy_name=gp_model_name, use_GPU=use_GPU, M=M, cov_factor=np.sqrt(2))            
        # N=10, track: RadiusArclengthTrack = None, interval=0.1, startup_cycles=5, clear_timeout=1, destroy_timeout=5,  cov: float = 0):

        ## constant angular velocity model 
        self.cav_predictor = ConstantAngularVelocityPredictor(N=self.n_nodes, cov= .001)            

        ## NMPC based game theoretic approach 
        self.nmpc_predictor = NLMPCPredictor(N=self.n_nodes, track=self.track_info.track, cov=.01, v_ref=mpcc_tv_params.vx_max)
        self.nmpc_predictor.set_warm_start()
        
        
        # NLMPCPredictor(N, None, cov=.01, v_ref=mpcc_tv_params.vx_max),
        
            
        # prediction callback   
        self.tv_pred = None
        self.tv_cav_pred = None
        self.tv_nmpc_pred = None
        self.tv_gp_pred = None
        self.prediction_hz = rospy.get_param('~prediction_hz', default=10)
        self.prediction_timer = rospy.Timer(rospy.Duration(1/self.prediction_hz), self.prediction_callback)         
        ## controller callback        
        self.ego_list = []
        self.tar_list = []
        self.data_save = False
        self.save_buffer_legnth = 100
        
        
        
        rate = rospy.Rate(1)     
        while not rospy.is_shutdown():            
            msg = Bool()
            msg.data = True
            # self.status_pub.publish(msg)          
            rate.sleep()

    def ego_warm_start(self):
        cur_state_copy = self.cur_ego_state.copy()
        x_ref = cur_state_copy.p.x_tran
        
        pid_steer_params = PIDParams()
        pid_steer_params.dt = self.dt
        pid_steer_params.default_steer_params()
        pid_steer_params.Kp = 1
        pid_speed_params = PIDParams()
        pid_speed_params.dt = self.dt
        pid_speed_params.default_speed_params()
        pid_controller_1 = PIDLaneFollower(cur_state_copy.v.v_long, x_ref, self.dt, pid_steer_params, pid_speed_params)
        ego_dynamics_simulator = DynamicsSimulator(0.0, ego_dynamics_config, track=self.track_info.track) 
        input_ego = VehicleActuation()
        t = 0.0
        state_history_ego = deque([], self.n_nodes); input_history_ego = deque([], self.n_nodes)
        n_iter = self.n_nodes+1
        approx = True
        while n_iter > 0:
            pid_controller_1.step(cur_state_copy)
            ego_dynamics_simulator.step(cur_state_copy)            
            self.track_info.track.update_curvature(cur_state_copy)
            input_ego.t = t
            cur_state_copy.copy_control(input_ego)
            q, _ = ego_dynamics_simulator.model.state2qu(cur_state_copy)
            u = ego_dynamics_simulator.model.input2u(input_ego)
            if approx:
                q = np.append(q, cur_state_copy.p.s)
                q = np.append(q, cur_state_copy.p.s)
                u = np.append(u, cur_state_copy.v.v_long)
            state_history_ego.append(q)
            input_history_ego.append(u)
            t += self.dt
            n_iter-=1
           
        compose_history = lambda state_history, input_history: (np.array(state_history), np.array(input_history))
        ego_warm_start_history = compose_history(state_history_ego, input_history_ego)
        self.gp_mpcc_ego_controller.initialize()
        self.gp_mpcc_ego_controller.set_warm_start(*ego_warm_start_history)
        print("warm start done")
 

    def dyn_callback(self,config,level):
        self.data_save = config.logging_vehicle_states
        print("dyn reconfigured")
        
        return config
        
    def clear_buffer(self):
        if len(self.ego_list) > 0:
            self.ego_list.clear()
            self.tar_list.clear()

# @dataclass
# class RealData():
#     track: RadiusArclengthTrack
#     N: int
#     ego_states: List[VehicleState]
#     tar_states: List[VehicleState]


    def save_buffer(self):
        real_data = RealData(self.track_info.track, self.n_nodes, self.ego_list, self.tar_list)
        create_dir(path=real_dir)        
        pickle_write(real_data, os.path.join(real_dir, str(self.cur_ego_state.t) + '.pkl'))
        rospy.loginfo("states data saved")
        self.clear_buffer()
        rospy.loginfo("states buffer has been cleaned")

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
        shift_in_local_x(self.cur_tar_pose, dist = -0.10)
    
   

    def prediction_callback(self,event):
        start_time = time.time()
        if self.ego_odom_ready is False or self.tar_odom_ready is False:
            
            return
        
        if self.ego_odom_ready and self.tar_odom_ready:
            pose_to_vehicleState(self.track_info.track, self.cur_ego_state, self.cur_ego_pose)
            odom_to_vehicleState(self.cur_ego_state, self.cur_ego_odom)
            
            
            pose_to_vehicleState(self.track_info.track, self.cur_tar_state, self.cur_tar_pose)            
            odom_to_vehicleState(self.cur_tar_state, self.cur_tar_odom)
            
            
        else:
            rospy.loginfo("state not ready")
            return 

        if self.predictor and self.cur_ego_state is not None:    
            
            if self.data_save:
                self.ego_list.append(self.cur_ego_state)
                self.tar_list.append(self.cur_tar_state)
                if len(self.tar_list) > self.save_buffer_legnth:
                    self.save_buffer()

            ## TODO : receive ego prediction from mpcc ctrl, instead computing one more time            
            self.use_predictions_from_module = True
            _, problem, cur_obstacles = self.gp_mpcc_ego_controller.step(self.cur_ego_state, tv_state=self.cur_tar_state, tv_pred=self.tv_pred if self.use_predictions_from_module else None)
            ego_pred = self.gp_mpcc_ego_controller.get_prediction()
            # ego_pred = self.predictor.get_constant_vel_prediction_par(self.cur_ego_state)
            
            if self.cur_ego_state.t is not None and self.cur_tar_state.t is not None:            
                
                self.tv_pred = self.predictor.get_prediction(self.cur_ego_state, self.cur_tar_state, ego_pred)               
                # self.tv_cav_pred = self.cav_predictor.get_prediction(ego_state = self.cur_ego_state, target_state = self.cur_tar_state, ego_prediction = ego_pred)                               
                # self.tv_nmpc_pred = self.nmpc_predictor.get_prediction(ego_state = self.cur_ego_state, target_state = self.cur_tar_state, ego_prediction = ego_pred)
                # self.tv_gp_pred = self.gp_predictor.get_prediction(ego_state = self.cur_ego_state, target_state = self.cur_tar_state, ego_prediction = ego_pred)
                #################### predict only target is close to ego #####################################
                cur_ego_s = self.cur_ego_state.p.s.copy()
                cur_tar_s = self.cur_tar_state.p.s.copy()
                diff_s = abs(cur_ego_s - cur_tar_s)
                
                if diff_s > self.track_info.track.track_length-3:                 
                    diff_s = diff_s - self.track_info.track.track_length
                
                if diff_s < 7.0:                 
                #################### predict only target is close to ego END #####################################
                    ## publish our proposed method prediction 
                    if self.tv_pred is not None:            
                        fill_global_info(self.track_info.track, self.tv_pred)                    
                        tar_pred_msg = prediction_to_rosmsg(self.tv_pred)                        
                        self.tar_pred_pub.publish(tar_pred_msg)
                        # convert covarariance in local coordinate for visualization
                        # self.tv_pred.convert_local_to_global_cov()
                        ###
                        tv_pred_markerArray = prediction_to_marker(self.tv_pred)
                        self.tv_pred_marker_pub.publish(tv_pred_markerArray)
                    
                    if self.tv_cav_pred is not None:            
                        fill_global_info(self.track_info.track, self.tv_cav_pred)                    
                        tar_cav_pred_msg = prediction_to_rosmsg(self.tv_cav_pred)                        
                        # self.tar_cav_pred_pub.publish(tar_cav_pred_msg)                        
                        ###
                        tv_cav_pred_markerArray = prediction_to_marker(self.tv_cav_pred)
                        self.tv_cav_pred_marker_pub.publish(tv_cav_pred_markerArray)

                    if self.tv_nmpc_pred is not None:
                        fill_global_info(self.track_info.track, self.tv_nmpc_pred)                    
                        tar_nmpc_pred_msg = prediction_to_rosmsg(self.tv_nmpc_pred)                        
                        # self.tar_cav_pred_pub.publish(tar_cav_pred_msg)                        
                        ###
                        tv_nmpc_pred_markerArray = prediction_to_marker(self.tv_nmpc_pred)
                        self.tv_nmpc_pred_marker_pub.publish(tv_nmpc_pred_markerArray)

                    if self.tv_gp_pred is not None:
                        fill_global_info(self.track_info.track, self.tv_gp_pred)                    
                        tar_gp_pred_msg = prediction_to_rosmsg(self.tv_nmpc_pred)                        
                        # self.tar_cav_pred_pub.publish(tar_cav_pred_msg)                        
                        ###
                        tv_gp_pred_markerArray = prediction_to_marker(self.tv_gp_pred)
                        self.tv_gp_pred_marker_pub.publish(tv_gp_pred_markerArray)


                
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Prediction execution time: {execution_time} seconds")

    
###################################################################################

def main():
    rospy.init_node("predictor")    
    Predictor()

if __name__ == "__main__":
    main()




 
    


