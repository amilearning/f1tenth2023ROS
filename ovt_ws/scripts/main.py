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

from hmcl_msgs.srv import mpcc, mpccResponse
###
class OvertakingAgent:
    def __init__(self):
        print("init")
        #Topics & Subs, Pubs
        self.use_predictions_from_module = False
        self.track_info = PathGenerator()
        self.cur_ego_state = VehicleState(t=0.0,
                                      p=ParametricPose(s=0.1, x_tran=0.2, e_psi=0.0),
                                      v=BodyLinearVelocity(v_long=0.5))
        self.cur_tar_state = VehicleState()
        self.cur_odom = Odometry()
        self.cur_tar_odom = Odometry()
        self.cur_pose = PoseStamped()
        self.cur_tar_pose = PoseStamped()

        self.tar_pose_received = False
        self.ego_pose_received = False
        # self.N = 10 ## number of step for MPC prediction 


        while self.track_info.track_ready is False:
             rospy.sleep(0.01)
        
        self.vehicle_model = CasadiDynamicBicycleFull(0.0, ego_dynamics_config, track=self.track_info.track)
        self.gp_mpcc_ego_controller = MPCC_H2H_approx(self.vehicle_model, self.track_info.track, control_params = gp_mpcc_ego_params, name="gp_mpcc_h2h_ego", track_name="test_track")
        self.gp_mpcc_ego_controller.initialize()
        
        self.warm_start()

        rospy.loginfo("ego controller initialized")
        self.center_pub = rospy.Publisher('/center_line',MarkerArray,queue_size = 2)
        self.bound_in_pub = rospy.Publisher('/track_bound_in',MarkerArray,queue_size = 2)
        self.bound_out_pub = rospy.Publisher('/track_bound_out',MarkerArray,queue_size = 2)
        self.odom_sub = rospy.Subscriber("/pose_estimate",Odometry,self.odom_callback)
        self.pose_sub = rospy.Subscriber("/tracked_pose",PoseStamped,self.pose_callback)
        
        self.tar_odom_sub = rospy.Subscriber("/target/pose_estimate",Odometry,self.tar_odom_callback)
        self.tar_pose_sub = rospy.Subscriber("/target/tracked_pose",PoseStamped,self.tar_pose_callback)

        self.controller_server = rospy.Service('compute_mpcc', mpcc, self.mpcc_srv_hanlder)

        self.ackman_pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/nav_hmcl',AckermannDriveStamped,queue_size = 2)
        
        self.timercallback = rospy.Timer(rospy.Duration(0.05), self.timer_callback)  
        # self.timercallback = rospy.Timer(rospy.Duration(2.0), self.timer_callback)  
        return 

        # rate = rospy.Rate(1)     
        # while not rospy.is_shutdown():                        
        #     rate.sleep()
    
    def mpcc_srv_hanlder(self,req):
        print("service recieved")
        problem = dict()
        problem["xinit"] = np.array(req.xinit)
        problem["all_parameters"] = np.array(req.all_parameters)
        problem["x0"] = np.array(req.x0)
        output, exitflag, solve_info = self.gp_mpcc_ego_controller.srv_solve(problem)
        
        stacked_output  = np.concatenate([value for value in output.values()])
        print("exitflag = ")
        print(exitflag)
        return mpccResponse(exitflag, stacked_output)


    def tar_odom_callback(self,msg):
        self.cur_tar_odom = msg
    def tar_pose_callback(self,msg):
        if self.tar_pose_received is False:
            self.tar_pose_received = True
        self.cur_tar_pose = msg

    def odom_callback(self,msg):
        self.cur_odom = msg

    def pose_callback(self,msg):
        if self.ego_pose_received is False:
            self.ego_pose_received = True
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
            self.center_pub.publish(self.track_info.centerline)
            self.bound_in_pub.publish(self.track_info.track_bound_in)
            self.bound_out_pub.publish(self.track_info.track_bound_out)
            
            ## 
            if self.ego_pose_received:
                self.pose_to_vehicleState(self.cur_ego_state, self.cur_pose)
                self.odom_to_vehicleState(self.cur_ego_state, self.cur_odom)
            else:
                return 
            
            if self.tar_pose_received:
                self.odom_to_vehicleState(self.cur_tar_state, self.cur_tar_odom)
                self.pose_to_vehicleState(self.cur_tar_state, self.cur_tar_pose)

            
            
            cur_tv_state = VehicleState(t=0.0,
                                        p=ParametricPose(s=0.0, x_tran=0.0, e_psi=0.0),
                                        v=BodyLinearVelocity(v_long=0.5))
            # print(self.cur_ego_state.p)
            info, b, exitflag = self.gp_mpcc_ego_controller.step(self.cur_ego_state, tv_state=cur_tv_state, tv_pred=None if self.use_predictions_from_module else None)
            pp_cmd = AckermannDriveStamped()        
            pp_cmd.header.stamp = self.cur_pose.header.stamp        
            vel_cmd = 0.0            
            if not info["success"]:
                print(f"EGO infeasible - Exitflag: {exitflag}")                
            else:
                pred_v_lon = self.gp_mpcc_ego_controller.x_pred[:,0] 
                vel_cmd = pred_v_lon[2]
                # vel_cmd = np.clip(vel_cmd, 0.5, 2.0)
                
            pp_cmd.drive.speed = vel_cmd            
            # pp_cmd.drive.speed = 0.0            
            pp_cmd.drive.steering_angle = self.cur_ego_state.u.u_steer
            self.ackman_pub.publish(pp_cmd)
            # print("accel = : {:.5f}".format(self.cur_ego_state.u.u_a))
            print("steer = : {:.5f}".format(self.cur_ego_state.u.u_steer))

            
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
