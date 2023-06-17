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
        self.track_info = PathGenerator()
        while self.track_info.track_ready is False:
             rospy.sleep(0.01)
        self.vehicle_model = CasadiDynamicBicycleFull(0.0, ego_dynamics_config, track=self.track_info.track)
        self.gp_mpcc_ego_controller = MPCC_H2H_approx(self.vehicle_model, self.track_info.track, control_params = gp_mpcc_ego_params, name="gp_mpcc_h2h_ego", track_name="test_track")
        self.gp_mpcc_ego_controller.initialize()
        print("solver generatoin completed")
        return 

            
def main(args):
    rospy.init_node("ovt_ws", anonymous=False)
    OA = OvertakingAgent()
    
    rospy.spin()

if __name__=='__main__':
	main(sys.argv)
