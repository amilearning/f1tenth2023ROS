
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

#include <hmcl_msgs/obstacle.h>
#include <hmcl_msgs/TrackArray.h>
#include <hmcl_msgs/Track.h>

#include "lowpass_filter.h"
#include "trajectory_manager.h"
#include "pure_pursuit.h"   
#include "utils.h"
#include "polyfit.h"
#include "state.h"
#include "trajectory.h"
#include "vehicle_dynamics.h"


#include "FORCESNLPsolver.h"
#include "FORCESNLPsolver_memory.h"


#define PI 3.14159265358979323846264338


class Ctrl 
{  
private:
ros::NodeHandle nh_ctrl_, nh_traj_, nh_state_, nh_p;


VehicleState cur_state, prev_state, obstacle_state; //< @brief vehicle status

bool my_steering_ok_,my_position_ok_, my_odom_ok_;
std::mutex odom_mtx, imu_mtx, pose_mtx, vesc_mtx, lidar_mtx;
ros::Subscriber  waypointSub,  odomSub, poseSub, imuSub, obstacleSub, vesodomSub, lidarSub;


std::vector<double> start_line_time;

bool first_traj_received;
bool first_pose_received;
Butterworth2dFilter x_vel_filter, y_vel_filter;

ros::Publisher est_odom_pub, keypts_info_pub, fren_pub, centerlin_info_pub, pred_traj_marker_pub, target_pointmarker_pub, ackmanPub, global_traj_marker_pub, local_traj_marker_pub;
ros::Publisher speed_target_pointmarker_pub, closest_obj_marker_pub;
double manual_target_vel;


PurePursuit pp_ctrl;
TrajectoryManager traj_manager;
Trajectory local_traj;
hmcl_msgs::Lane current_waypoints_;

dynamic_reconfigure::Server<highspeed_ctrl::testConfig> srv;
dynamic_reconfigure::Server<highspeed_ctrl::testConfig>::CallbackType f;

std::vector<double> delta_buffer;
int path_filter_moving_ave_num_, curvature_smoothing_num_, path_smoothing_times_;
double max_a_lat;
double odom_pose_diff_threshold;
double error_deriv_lpf_curoff_hz;
std::string pp_lookup_table_file_name, vel_cmd_topic, control_topic, pose_topic, vehicle_states_topic, waypoint_topic, odom_topic, status_topic, simstatus_topic, steer_cmd_topic;

double curv_lookahead_path_length, lookahead_path_length, wheelbase, lf, lr, mass, dt;
double manual_weight_ctrl;
bool manual_velocity, manual_lookahed_switch,  manual_speed_lookahed_switch;
bool config_switch;
double x_vel_filter_cutoff, y_vel_filter_cutoff;

double manual_lookahead, manual_speed_lookahead;

// hmcl_msgs::obstacle cur_obstacles;
hmcl_msgs::TrackArray obstacles;


VehicleDynamics ego_vehicle;

int ctrl_select;

geometry_msgs::PoseStamped cur_pose, prev_pose; 
nav_msgs::Odometry cur_odom, prev_odom;
bool first_odom_received;
sensor_msgs::Imu cur_imu;
bool imu_received;
bool is_odom_used;

sensor_msgs::LaserScan::ConstPtr cur_scan;

public:
Ctrl(ros::NodeHandle& nh_ctrl, ros::NodeHandle& nh_traj,ros::NodeHandle& nh_state, ros::NodeHandle& nh_p_);
~Ctrl();
void ControlLoop();
void filter_given_TargetClearance(ackermann_msgs::AckermannDriveStamped prev_cmd,visualization_msgs::Marker obst_marker);
void odomToVehicleState(VehicleState & vehicle_state, const nav_msgs::Odometry & odom,const bool & odom_twist_in_local);
// void callbackPose(const geometry_msgs::PoseStampedConstPtr& msg);
void lidarCallback(const sensor_msgs::LaserScan::ConstPtr &msg);
void obstacleCallback(const hmcl_msgs::TrackArrayConstPtr& msg);
void odomCallback(const nav_msgs::OdometryConstPtr& msg);
void vescodomCallback(const nav_msgs::OdometryConstPtr& msg);
void poseCallback(const geometry_msgs::PoseStampedConstPtr& msg);
void imuCallback(const sensor_msgs::Imu::ConstPtr& msg);
void callbackRefPath(const visualization_msgs::MarkerArray::ConstPtr &msg);
void trackToMarker(const hmcl_msgs::Track& state, visualization_msgs::Marker & marker);
bool odom_close_to_pose(const geometry_msgs::PoseStamped & pos, const nav_msgs::Odometry& odom);

void dyn_callback(highspeed_ctrl::testConfig& config, uint32_t level);

visualization_msgs::MarkerArray PathPrediction(const VehicleState state, int n_step);


};



