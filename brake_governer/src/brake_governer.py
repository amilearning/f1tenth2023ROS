#!/usr/bin/env python3
import sys
import numpy as np  

#ROS Imports
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from vesc_msgs.msg import VescStateStamped
from std_msgs.msg import Float64
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

max_a_x = -2.0
required_ttc = 2.0
view_angle = 10

class Break_governer:
    def __init__(self):
        #Topics & Subs, Pubs
        self.cur_velocity=0.0
        self.cmd_velocity=0.0
        self.ttc_stop_signal = False
        # self.cur_velocity_sub = rospy.Subscriber("/vesc/sensors/core", VescStateStamped, self.callback_cur_vel)
        self.cur_velocity_sub = rospy.Subscriber("/est_odom", Odometry, self.callback_cur_vel_factor)
        self.cmd_velocity_sub = rospy.Subscriber("/vesc/low_level/ackermann_cmd_mux/input/nav_hmcl2", AckermannDriveStamped, self.callback_cmd_vel)#: Subscribe to VESC
        self.lidar_sub = rospy.Subscriber("/scan", LaserScan, self.callback_laser)#: Subscribe to LIDAR
        self.drive_pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/nav_hmcl', AckermannDriveStamped, queue_size=10)#: Publish to drive
        self.brake_pub = rospy.Publisher('/vesc/commands/motor/brake',Float64,queue_size = 10)
        
        self.brake_governer_current = rospy.get_param('/brake_governer/current')

    def callback_cur_vel(self,data):
        speed_to_erpm_gain = rospy.get_param('/vesc/speed_to_erpm_gain')
        speed_to_erpm_offset = rospy.get_param('/vesc/speed_to_erpm_offset')
        self.cur_velocity = (-data.state.speed-speed_to_erpm_offset)/speed_to_erpm_gain

    def callback_cur_vel_factor(self,data):
        self.cur_velocity = data.twist.twist.linear.x
            
    def callback_cmd_vel(self,data):
        self.brake_governer_tolerance= rospy.get_param('/brake_governer/tolerance')
        self.cmd_velocity = data.drive.speed
        cmd_stop_signal = (self.cmd_velocity+self.brake_governer_tolerance < self.cur_velocity)
        if  cmd_stop_signal or self.ttc_stop_signal:
            self.brake( self.brake_governer_current)
            if cmd_stop_signal and self.ttc_stop_signal:
                print("brake cmd,ttc: cur, cmd: ", self.cur_velocity, ", ",self.cmd_velocity)
            elif cmd_stop_signal and not self.ttc_stop_signal:
                print("brake cmd : cur, cmd: ", self.cur_velocity, ", ",self.cmd_velocity)
            elif self.ttc_stop_signal and not cmd_stop_signal:
                print("brake ttc")
        else:
            # print("NOT brake: cur, cmd: ", self.cur_velocity, ", ",self.cmd_velocity)
            new_drive_msg = AckermannDriveStamped()
            new_drive_msg = data
            new_drive_msg.header.stamp = rospy.Time.now()
            self.drive_pub.publish(new_drive_msg)

    def callback_laser(self, data):
        range_len = len(data.ranges)
        angle_increment=data.angle_increment
        angle_offset=90-view_angle/2
        angle_offset_idx = int((angle_offset*(np.pi/180))/angle_increment)
        right_idx = int(range_len/8+angle_offset_idx)
        left_idx = int(range_len*7/8-angle_offset_idx)
        range = data.ranges[right_idx:left_idx]
        range = np.array([range])
        range= range.flatten()
        range[range<0.15]=1.0 
        
        theta = np.arange(right_idx*angle_increment,left_idx*angle_increment,angle_increment)
        vel_directed = self.cur_velocity*np.cos(theta)
        dist = vel_directed*required_ttc+0.5*max_a_x*required_ttc*required_ttc
        dist[dist<0]=0.01
        # ttc = range/vel_directed
        safe_score=range/dist
        min_safe_score_idx = np.argmin(safe_score)
        net_min_safe_score_idx = min_safe_score_idx + right_idx
        min_safe_score = safe_score[min_safe_score_idx]
        
        if (min_safe_score < 1 ) or (range[min_safe_score_idx] <0.4):
            # print("idx: ", net_min_safe_score_idx)
            # print("range: ",range[min_safe_score_idx])
            # print("safe_score: ",min_safe_score)
            # print("vel: ",self.cur_velocity)
            # print("dist:",dist[min_safe_score_idx])
            self.ttc_stop_signal = True
        else:
             self.ttc_stop_signal = False

    def brake(self, current):
            brake_msg = Float64()
            brake_msg.data = current
            self.brake_pub.publish(brake_msg) 
            
def main(args):
    rospy.init_node("brake_governer", anonymous=False)
    BG = Break_governer()
    rospy.sleep(0.02)
    rospy.spin()

if __name__=='__main__':
	main(sys.argv)
