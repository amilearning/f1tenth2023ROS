
//   Copyright (c) 2022 Ulsan National Institute of Science and Technology (UNIST)
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

//   Authour : Hojin Lee, hojinlee@unist.ac.kr


#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <cmath>
#include <cstdlib>
#include <chrono>


#include <boost/algorithm/string.hpp>
#include <boost/thread/thread.hpp>
#include <vector>

#include <sstream>
#include <string>
#include <list>
#include <queue>
#include <mutex> 
#include <thread> 
#include <numeric>
#include <boost/thread/thread.hpp>

#include <ros/ros.h>
#include <ros/time.h>
#include <std_msgs/Float64.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <eigen3/Eigen/Geometry>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>


#include <ackermann_msgs/AckermannDriveStamped.h>
#include <hmcl_msgs/VehicleStatus.h>
#include <hmcl_msgs/VehicleSCC.h>
#include <hmcl_msgs/VehicleSteering.h>
#include <hmcl_msgs/Lane.h>
#include <hmcl_msgs/CtrlVehicleStates.h>
#include <hmcl_msgs/VehicleWheelSpeed.h>
#include <dynamic_reconfigure/server.h>
#include <highspeed_ctrl/testConfig.h>
#include <std_msgs/UInt8MultiArray.h>

#include "lowpass_filter.h"
#include "trajectory_manager.h"
#include "utils.h"
#include "polyfit.h"
#include "state.h"

#include "FORCESNLPsolver.h"
#include "FORCESNLPsolver_memory.h"


#define PI 3.14159265358979323846264338


class Ctrl 
{  
private:
ros::NodeHandle nh_ctrl_, nh_traj_;


VehicleState cur_state, prev_state; //< @brief vehicle status

bool my_steering_ok_,my_position_ok_, my_odom_ok_;
std::mutex mtx_;
ros::Subscriber  waypointSub,  odomSub;
ros::Publisher  ackmanPub, global_traj_marker_pub, local_traj_marker_pub;




TrajectoryManager traj_manager;
hmcl_msgs::Lane current_waypoints_;

dynamic_reconfigure::Server<highspeed_ctrl::testConfig> srv;
dynamic_reconfigure::Server<highspeed_ctrl::testConfig>::CallbackType f;

std::vector<double> delta_buffer;
int path_filter_moving_ave_num_, curvature_smoothing_num_, path_smoothing_times_;


double error_deriv_lpf_curoff_hz;
std::string vel_cmd_topic, control_topic, pose_topic, vehicle_states_topic, waypoint_topic, odom_topic, status_topic, simstatus_topic, steer_cmd_topic;

double lookahead_path_length, wheelbase, lf, lr, mass, dt;


bool config_switch;


public:
Ctrl(ros::NodeHandle& nh_ctrl, ros::NodeHandle& nh_traj);
~Ctrl();
void ControlLoop();

void odomToVehicleState(VehicleState & vehicle_state, const nav_msgs::Odometry & odom);
// void callbackPose(const geometry_msgs::PoseStampedConstPtr& msg);

void odomCallback(const nav_msgs::OdometryConstPtr& msg);

void callbackRefPath(const hmcl_msgs::Lane::ConstPtr &msg);

void dyn_callback(highspeed_ctrl::testConfig& config, uint32_t level);



};



