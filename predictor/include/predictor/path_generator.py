#!/usr/bin/env python3
import numpy as np  
import tf
#ROS Imports
import rospy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker
from predictor.common.tracks.radius_arclength_track import RadiusArclengthTrack
# import matplotlib.pyplot as plt

class PathGenerator:
    def __init__(self):
        print("init")
        #Topics & Subs, Pubs
        self.cur_velocity=0.0
        self.cmd_velocity=0.0
        self.ttc_stop_signal = False
        self.centerline = MarkerArray()
        self.track_bound_in = MarkerArray()
        self.track_bound_out = MarkerArray()
        self.dt = 0.05
        self.track_width = 1.5
        self.slack = 0.45
        self.cl_segs = None
        self.track = None 
        self.track_ready = False
        self.center_marker = MarkerArray()
        self.gen_path()
        
        
        # self.center_pub = rospy.Publisher('/center_line',MarkerArray,queue_size = 2)
        # self.bound_in_pub = rospy.Publisher('/track_bound_in',MarkerArray,queue_size = 2)
        # self.bound_out_pub = rospy.Publisher('/track_bound_out',MarkerArray,queue_size = 2)
        # self.timercallback = rospy.Timer(rospy.Duration(0.5), self.timer_callback)  
    
    # def timer_callback(self,event):
    #     if self.track_ready:            
    #         rospy.loginfo("Timer callback executed.")
    #         self.center_pub.publish(self.centerline)
    #         self.bound_in_pub.publish(self.track_bound_in)
    #         self.bound_out_pub.publish(self.track_bound_out)
        
    
        
    def gen_path(self):
        self.track = RadiusArclengthTrack()

        stright = np.array([[4.0, 0.0]])
        curvy_straith = np.array([[1.5*np.pi/15.0, 1.5],[0.5, 0.0],[1.5*np.pi/15.0, -1.5],[0.5, 0.0],[1.5*np.pi/15.0, 1.5],[0.5, 0.0],[1.5*np.pi/15.0, -1.5],[0.5, 0.0],[1.5*np.pi/15.0, 1.5],[0.5, 0.0],[1.5*np.pi/15.0, -1.5],[0.5, 0.0]])
        curve = np.array([[2.0*np.pi, 2.0]])
        end_curve = np.array([[2.0*np.pi-0.01, 2.0]])
        track = np.vstack([stright,curve, stright,end_curve])
        # track = np.vstack([curve, end_curve])
        self.cl_segs = track
        # self.cl_segs = np.array([[1.5*np.pi/15.0, 1.5],[0.5, 0.0],[1.5*np.pi/15.0, -1.5],[0.5, 0.0],[1.5*np.pi/15.0, 1.5],[0.5, 0.0],[1.5*np.pi/15.0, -1.5],[0.5, 0.0]])                                
        
        # ,[1.5*np.pi/30.0, 1.5],[1.5, 0.0],[1.5*np.pi, 1.5],[1.5, 0.0],[1.5*np.pi/30.0, -1.5],[1.5, 0.0],[1.5*np.pi/30.0, 1.5],[1.5, 0.0],[1.5*np.pi-0.2, 1.5]])                                
            # [5.0, 0.0],[1.5*np.pi, 1.5],[5.0, 0.0],[1.5*np.pi-0.01, 1.5]])
        self.track.initialize(self.track_width,self.slack, self.cl_segs, init_pos=(0.5, -.9, -0.1))
        
        self.track_ready = True
        self.get_track_points()
        # fig, ax = plt.subplots()
        # self.track.plot_map(ax)
        # plt.show()
        return
        # for(int i=0; i < marker_data.markers.size(); i++){
        #     x = marker_data.markers[i].pose.position.x;
        #     y = marker_data.markers[i].pose.position.y;
        #     vx = marker_data.markers[i].pose.position.z;
        # return
    def get_marker_from_track(self,x,y,psi,color):
        tmpmarkerArray = MarkerArray()
        time = rospy.Time.now()
        for i in range(len(x)):
            tmp_marker = Marker()
            tmp_marker.header.stamp = time
            tmp_marker.header.frame_id = "map"
            tmp_marker.id = i
            tmp_marker.type = Marker.ARROW
            # Set the scale of the marker
            tmp_marker.scale.x = 0.05
            tmp_marker.scale.y = 0.05
            tmp_marker.scale.z = 0.01

            # Set the color
            tmp_marker.color.r = color.r
            tmp_marker.color.g = color.g
            tmp_marker.color.b = color.b
            tmp_marker.color.a = color.a

            tmp_marker.pose.position.x = x[i]
            tmp_marker.pose.position.y =y[i]
            tmp_marker.pose.position.z = 0.0
            quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, psi[i])
            tmp_marker.pose.orientation.x = quaternion[0]
            tmp_marker.pose.orientation.y = quaternion[1]
            tmp_marker.pose.orientation.z = quaternion[2]
            tmp_marker.pose.orientation.w = quaternion[3]
            tmpmarkerArray.markers.append(tmp_marker)
        return tmpmarkerArray

    def get_track_points(self):
        if self.track is None:
            return
        n = int(self.track.track_length/self.dt)        

        x_track = []
        x_bound_in = []
        x_bound_out = []
        y_track = []
        y_bound_in = []
        y_bound_out = []
        psi_track = []
        psi_bound_in = []
        psi_bound_out = []
   
        for i in range(n):
            j = i*self.dt 
            cl_coord = (j, 0, 0)
            xy_coord = self.track.local_to_global(cl_coord)
            x_track.append(xy_coord[0])
            y_track.append(xy_coord[1])
            psi_track.append(xy_coord[2])
            cl_coord = (j, self.track_width / 2, 0)
            xy_coord = self.track.local_to_global(cl_coord)
            x_bound_in.append(xy_coord[0])
            y_bound_in.append(xy_coord[1])
            psi_bound_in.append(xy_coord[2])
            cl_coord = (j, -self.track_width / 2, 0)
            xy_coord = self.track.local_to_global(cl_coord)
            x_bound_out.append(xy_coord[0])
            y_bound_out.append(xy_coord[1])
            psi_bound_out.append(xy_coord[2])

        color_center = ColorRGBA()        
        color_center.g = 1.0
        color_center.a = 0.3
        self.centerline = self.get_marker_from_track(x_track,y_track,psi_track,color_center)

        color_bound = ColorRGBA()        
        color_bound.r = 1.0
        color_bound .a = 0.3
        self.track_bound_in = self.get_marker_from_track(x_bound_in,y_bound_in,psi_bound_in,color_bound)
        self.track_bound_out = self.get_marker_from_track(x_bound_out,y_bound_out,psi_bound_out,color_bound)


