#!/usr/bin/env python
import sys
import math
import numpy as np  

#ROS Imports
import rospy
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64



#TO DO: Tune parameters
#PARAMS 
max_a_x = -4.0
required_ttc = 2.0
view_angle = 10

class EmergencyStop:
    def __init__(self):
        #Topics & Subs, Pubs
        self.velocity=0.0
        self.stop_signal = 0
        self.velocity_sub = rospy.Subscriber("/vesc/odom", Odometry, self.callback_vel)#: Subscribe to VESC
        self.lidar_sub = rospy.Subscriber("/scan", LaserScan, self.callback)#: Subscribe to LIDAR
        self.drive_pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/safety', AckermannDriveStamped, queue_size=10)#: Publish to drive
        self.brake_pub = rospy.Publisher('/vesc/commands/motor/brake',Float64,queue_size = 10)
        self.brake_governer_current = rospy.get_param('/brake_governer/current')

    def callback_vel(self,data):
        self.velocity=-data.twist.twist.linear.x
    
    def callback(self, data):
        # if self.velocity < 0.1:
            # return
  
        range_len = len(data.ranges)
        # print(range_len)
        angle_increment=data.angle_increment
        angle_offset=90-view_angle/2
        angle_offset_idx = int((angle_offset*(np.pi/180))/angle_increment)
        # print(angle_offset)
        right_idx = int(range_len/8+angle_offset_idx)
        left_idx = int(range_len*7/8-angle_offset_idx)
        range = data.ranges[right_idx:left_idx]
        # print(right_idx)
        # print(left_idx)
        # print(len(range))
        range = np.array([range])
        range= range.flatten()
        range[range<0.15]=1.0 
        angle_min=data.angle_min
        angle_max=data.angle_max
        
        theta = np.arange(right_idx*angle_increment,left_idx*angle_increment,angle_increment)
        vel_directed = self.velocity*np.cos(theta)
        dist = vel_directed*required_ttc+0.5*max_a_x*required_ttc*required_ttc
        dist[dist<0]=0.01
        # ttc = range/vel_directed
        safe_score=range/dist
        min_safe_score_idx = np.argmin(safe_score)
        net_min_safe_score_idx = min_safe_score_idx + right_idx
        min_safe_score = safe_score[min_safe_score_idx]
        
        if (min_safe_score < 1 ) or (range[min_safe_score_idx] <0.4):
            print("idx: ", net_min_safe_score_idx)
            print("range: ",range[min_safe_score_idx])
            print("safe_score: ",min_safe_score)
            print("vel: ",self.velocity)
            print("dist:",dist[min_safe_score_idx])       
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = rospy.Time.now()
            drive_msg.header.frame_id = "laser"
            drive_msg.drive.speed = 0
            if net_min_safe_score_idx < len(range)/2:
                #drive_msg.drive.steering_angle = 0.5
                drive_msg.drive.steering_angle = 0.1
            else:
                #drive_msg.drive.steering_angle = -0.5
                drive_msg.drive.steering_angle = -0.1
            self.drive_pub.publish(drive_msg)
            brake_msg = Float64()
            brake_msg.data = self.brake_governer_current
            self.brake_pub.publish(brake_msg) 
         
    def brake(self, current):
        brake_msg = Float64()
        brake_msg.data = current
        self.brake_pub.publish(brake_msg) 
    
def main(args):
    rospy.init_node("Emergengy_Stop", anonymous=False)
    ES = EmergencyStop()
    rospy.sleep(0.025)
    rospy.spin()

if __name__=='__main__':
	main(sys.argv)
