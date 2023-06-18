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
        

        rospy.loginfo("ego controller initialized")
        self.center_pub = rospy.Publisher('/rc_center_line',MarkerArray,queue_size = 2)
        self.bound_in_pub = rospy.Publisher('/rc_track_bound_in',MarkerArray,queue_size = 2)
        self.bound_out_pub = rospy.Publisher('/rc_track_bound_out',MarkerArray,queue_size = 2)
        
        
        self.timercallback = rospy.Timer(rospy.Duration(0.5), self.timer_callback)  
        # self.timercallback = rospy.Timer(rospy.Duration(2.0), self.timer_callback)  
        return 

        # rate = rospy.Rate(1)     
        # while not rospy.is_shutdown():                        
        #     rate.sleep()
    


    def timer_callback(self,event):
        if self.track_info.track_ready:     
            self.center_pub.publish(self.track_info.centerline)
            self.bound_in_pub.publish(self.track_info.track_bound_in)
            self.bound_out_pub.publish(self.track_info.track_bound_out)
            
            return 
            

            
            
def main(args):
    rospy.init_node("ovt_ws", anonymous=False)
    OA = OvertakingAgent()
    
    rospy.spin()

if __name__=='__main__':
	main(sys.argv)
