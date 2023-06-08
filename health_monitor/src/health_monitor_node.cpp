#include "ros/ros.h"
#include "imu_health_checker.h"
#include "vesc_health_checker.h"
#include "lidar_health_checker.h"
int main(int argc, char** argv)
{
  // Initialize the ROS node
  ros::init(argc, argv, "health_monitor_node");
  ros::NodeHandle nh("~");
  // Create an instance of the ImuHealthChecker class

  ImuHealthChecker imu_checker(nh);
  LidarHealthChecker lidar_checker(nh);
  VescHealthChecker vesc_checker(nh);

  // Start the ROS spin loop
  ros::spin();

  return 0;
}
