import rospy
import numpy as np
import time
import math

from visualization_msgs.msg import MarkerArray, Marker


class PublishTrack():
    def __init__(self):
        
        self.cst_pub = rospy.Publisher("track_info", MarkerArray, queue_size = 1)
        
        file_name = "./result/center_traj_with_boundary.txt"
        self.read_file(file_name)
        
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.cst_pub.publish(self.track)
            r.sleep()
    
    def read_file(self, filename):
        track = np.loadtxt(filename, delimiter=",", dtype = float)
        
        track_markers = MarkerArray()
        id = 0
        for i in range(len(track)):
            track_marker = Marker()
            track_marker.header.frame_id = "map"  
            track_marker.header.stamp = rospy.Time.now()
            track_marker.ns = "track"
            track_marker.id = id
            track_marker.type = Marker.SPHERE
            track_marker.action = Marker.ADD          
            
                 
            track_marker.pose.position.x = track[i,0] #position x
            track_marker.pose.position.y = track[i,1] #posiiton y
            track_marker.pose.position.z = track[i,2] #velocity
            track_marker.pose.orientation.x = track[i,3] #curvature
            track_marker.pose.orientation.y = track[i,4] #left_width
            track_marker.pose.orientation.z = track[i,5] #right_width
            
            track_marker.color.r, track_marker.color.g, track_marker.color.b = (0, 255, 0)
            track_marker.color.a = 1
            track_marker.scale.x = 0.2
            track_marker.scale.y = 0.2
            track_markers.markers.append(track_marker)
            id += 1
        
        self.track = track_markers

            
def main():
    rospy.init_node("track_info_node")
    PublishTrack()
    
if __name__ == '__main__':
    main()