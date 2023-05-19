#include <ros/ros.h>
#include <network_analysis/NetworkDelay.h>
#include <geometry_msgs/PoseStamped.h>

ros::Publisher pose_pub;

void customMsgCallback(const network_analysis::NetworkDelay::ConstPtr& msg)
{
  // Callback function for the custom message type subscriber
  geometry_msgs::PoseStamped pose_stamped;

  // Do some processing to get the pose_stamped message
  // For this example, let's assume x, y, z values are taken from the custom message
  pose_stamped.header.stamp = ros::Time::now();
  pose_stamped.pose.position.x = msg->delay;
  pose_stamped.pose.position.y = msg->alive;

  // Publish the pose_stamped message
  pose_pub.publish(pose_stamped);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "network_delay_converted");
  ros::NodeHandle nh;

  // Subscriber for the custom message type
  ros::Subscriber custom_msg_sub = nh.subscribe<network_analysis::NetworkDelay>(
    "network_analysis/network_delay", 10, customMsgCallback);

  // Publisher for the converted message type
  pose_pub = nh.advertise<geometry_msgs::PoseStamped>(
    "network_analysis/network_delay_converted", 10);

  // Spin to wait for messages
  ros::spin();

  return 0;
}
