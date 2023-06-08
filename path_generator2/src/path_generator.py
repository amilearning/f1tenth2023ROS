#!/usr/bin/env python3
import sys
import numpy as np  

#ROS Imports
import rospy
from std_msgs.msg import Float64
from visualization_msgs.msg import MarkerArray
from radius_arclength_track import RadiusArclengthTrack

class PathGenerator:
    def __init__(self):
        #Topics & Subs, Pubs
        self.cur_velocity=0.0
        self.cmd_velocity=0.0
        self.ttc_stop_signal = False
        self.centerline = MarkerArray()
        self.dt = 0.1  
        self.track_width = 1.5
        self.slack = 0.45
        self.cl_segs = None
        self.define_cl_segs()
        self.track = RadiusArclengthTrack()
        
        
        self.delta_pub = rospy.Publisher('/center_line',Float64,queue_size = 2)
        self.timercallback = rospy.Timer(rospy.Duration(0.5), self.timer_callback)  
    
    def timer_callback(self,event):
        rospy.loginfo("Timer callback executed.")
         
         
    def gen_path(self):
        self.track.initialize(self.track_width,self.slack, self.cl_segs)
        return
        # for(int i=0; i < marker_data.markers.size(); i++){
        #     x = marker_data.markers[i].pose.position.x;
        #     y = marker_data.markers[i].pose.position.y;
        #     vx = marker_data.markers[i].pose.position.z;
        # return

    def define_cl_segs(self):
        self.cl_segs = np.array([[2.0, 0.0],[2.0, 1.0],[2.0, 0.0], [2.0, -1.0]])
    

            
def main(args):
    rospy.init_node("path_generator", anonymous=False)
    PG = PathGenerator()
    rospy.sleep(0.02)
    rospy.spin()

if __name__=='__main__':
	main(sys.argv)
