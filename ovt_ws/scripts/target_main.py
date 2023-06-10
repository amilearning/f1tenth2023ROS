#!/usr/bin/env python3
import sys
import numpy as np  
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
#ROS Imports
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float64, ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker
import rospkg
from collections import deque
from PID import PIDLaneFollower
from pytypes import VehicleState, VehiclePrediction
from radius_arclength_track import RadiusArclengthTrack
from path_generator import PathGenerator
from MPCC_H2H_approx import MPCC_H2H_approx
from dynamics_models import CasadiDynamicBicycleFull
from dynamics_simulator import DynamicsSimulator
# import matplotlib.pyplot as plt
from h2h_configs import *

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from utils import get_local_vel

class OvertakingAgent:
    def __init__(self):
        print("init")
        #Topics & Subs, Pubs
        self.use_predictions_from_module = False
        self.track_info = PathGenerator()
        self.cur_ego_state = VehicleState(t=0.0,
                                      p=ParametricPose(s=0.1, x_tran=0.2, e_psi=0.0),
                                      v=BodyLinearVelocity(v_long=0.5))
        self.cur_odom = Odometry()
        self.cur_pose = PoseStamped()
        # self.N = 10 ## number of step for MPC prediction 


        while self.track_info.track_ready is False:
             rospy.sleep(0.01)
        
        self.vehicle_model = CasadiDynamicBicycleFull(0.0, ego_dynamics_config, track=self.track_info.track)
        # self.gp_mpcc_ego_controller = MPCC_H2H_approx(self.vehicle_model, self.track_info.track, control_params = gp_mpcc_ego_params, name="gp_mpcc_h2h_ego", track_name="test_track")
        # self.gp_mpcc_ego_controller.initialize()
        
        # self.warm_start()

        rospy.loginfo("ego controller initialized")
        
        self.odom_sub = rospy.Subscriber("/target/pose_estimate",Odometry,self.odom_callback)
        self.pose_sub = rospy.Subscriber("/target/tracked_pose",PoseStamped,self.pose_callback)
        
        self.ackman_pub = rospy.Publisher('/target/low_level/ackermann_cmd_mux/input/nav_hmcl',AckermannDriveStamped,queue_size = 2)
        
        self.timercallback = rospy.Timer(rospy.Duration(0.05), self.timer_callback)  
        
    
    def odom_callback(self,msg):
        self.cur_odom = msg
    def pose_callback(self,msg):
        self.cur_pose = msg
    
    def pose_to_vehicleState(self,state : VehicleState,pose : PoseStamped):
        state.x.x = pose.pose.position.x
        state.x.y = pose.pose.position.y
        orientation_q = pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (cur_roll, cur_pitch, cur_yaw) = euler_from_quaternion (orientation_list)
        state.e.psi = cur_yaw
        xy_coord = (state.x.x, state.x.y, state.e.psi)
        cl_coord = self.track_info.track.global_to_local(xy_coord)
        if cl_coord is None:
            return
        state.p.s = cl_coord[0]
        state.p.x_tran = cl_coord[1]
        state.p.e_psi = cl_coord[2]
       
    def odom_to_vehicleState(self,state:VehicleState, odom: Odometry):
       
        local_vel = get_local_vel(odom, is_odom_local_frame = False)
        state.v.v_long = local_vel[0]
        state.v.v_tran = local_vel[1]
        state.w.w_psi = odom.twist.twist.angular.z

    def timer_callback(self,event):
        if self.track_info.track_ready:     
            
            ## 
            self.pose_to_vehicleState(self.cur_ego_state, self.cur_pose)
            self.odom_to_vehicleState(self.cur_ego_state, self.cur_odom)
            

            
    def warm_start(self):
        cur_state_copy = self.cur_ego_state.copy()
        x_ref = cur_state_copy.p.x_tran
        
        pid_steer_params = PIDParams()
        pid_steer_params.dt = dt
        pid_steer_params.default_steer_params()
        pid_steer_params.Kp = 1
        pid_speed_params = PIDParams()
        pid_speed_params.dt = dt
        pid_speed_params.default_speed_params()
        pid_controller_1 = PIDLaneFollower(cur_state_copy.v.v_long, x_ref, dt, pid_steer_params, pid_speed_params)
        ego_dynamics_simulator = DynamicsSimulator(0.0, ego_dynamics_config, track=self.track_info.track) 
        input_ego = VehicleActuation()
        t = 0.0
        state_history_ego = deque([], N); input_history_ego = deque([], N)
        n_iter = N+1
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
            t += dt
            n_iter-=1
           
        compose_history = lambda state_history, input_history: (np.array(state_history), np.array(input_history))
        ego_warm_start_history = compose_history(state_history_ego, input_history_ego)
        
        self.gp_mpcc_ego_controller.set_warm_start(*ego_warm_start_history)
        print("warm start done")
            
def main(args):
    rospy.init_node("ovt_ws", anonymous=False)
    OA = OvertakingAgent()
    
    rospy.spin()

if __name__=='__main__':
	main(sys.argv)
