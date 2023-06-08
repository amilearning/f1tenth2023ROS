#!/usr/bin/env python3
import rospy
from std_msgs.msg import String, Float64, Header
from vesc_msgs.msg import VescStateStamped
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np
import enum


class Collector:

    def __init__(self):
      
        
        self.time_count = 0
        self.tar_vel = 4.0
        self.target_delta = 0.2
        self.delta_rate = 0.02
        self.delta_limit = 0.43
        self.delta_sign = 1.0
        self.cur_delta = 0.0
        self.dt = 0.1  # seconds
        self.cmd_publisher_ = rospy.Publisher('/vesc/ackermann_cmd', AckermannDriveStamped, queue_size=10)

        self.cmd_msg = AckermannDriveStamped()

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'vesc'
        self.cmd_msg.header = header

        self.timer = rospy.Timer(rospy.Duration(self.dt), self.timer_callback)

        self.subscription = rospy.Subscriber(
            '/vesc/sensors/servo_position_command',
            Float64,
            self.vehiclestate_callback)

  

    def vehiclestate_callback(self, msg):
        self.cur_delta = msg.data

    def timer_callback(self, event):
        cmd_msg = AckermannDriveStamped()

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'vesc'
        cmd_msg.header = header
        if self.target_delta is None:
            self.target_delta = 0.0
        cmd_msg.drive.steering_angle = float(-self.target_delta)
        if self.target_delta is None:
            self.tar_vel = 0.0
        cmd_msg.drive.speed = float(self.tar_vel)

        cmd_msg.drive.steering_angle_velocity = 0.0
        cmd_msg.drive.acceleration = 0.0
        cmd_msg.drive.jerk = 0.0

        self.target_delta = self.target_delta + (self.delta_rate * self.dt * self.delta_sign)
        self.target_delta = np.clip([self.target_delta], [-self.delta_limit], [self.delta_limit])
        if np.abs(self.target_delta) >= self.delta_limit:
            self.delta_sign = -1 * self.delta_sign

        self.cmd_publisher_.publish(cmd_msg)


def main():
    rospy.init_node('data_collect', anonymous=True)
    data_collect = Collector()

   
    rospy.spin()
 

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    rospy.signal_shutdown("Node has been shut down.")


if __name__ == '__main__':
    main()
