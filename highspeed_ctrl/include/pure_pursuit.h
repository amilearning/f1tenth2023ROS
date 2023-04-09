// Copyright 2023 HMCL, High-assurance Mobility and Control Laboratory in UNIST 
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef PURE_PURSUIT_PURE_PURSUIT_HPP_
#define PURE_PURSUIT_PURE_PURSUIT_HPP_

#include <ros/package.h>
#include <ros/ros.h>
#include <ros/time.h>

#include "std_srvs/Trigger.h"
#include <vector>
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <utility>
#include <ackermann_msgs/AckermannDriveStamped.h>
#include "std_msgs/ColorRGBA.h"
#include "state.h"
#include "trajectory.h"

#include <visualization_msgs/Marker.h>


/// \brief Given a trajectory and the current state, compute the control command
class PurePursuit
{
public:  
  PurePursuit(const ros::NodeHandle& nh_ctrl);
  void update_vehicleState(const VehicleState & state);
  void update_ref_traj(const Trajectory & ref);
  
  ackermann_msgs::AckermannDriveStamped compute_command();
  PathPoint get_target_point();
   visualization_msgs::Marker getTargetPointhMarker();



private:
  
  ros::NodeHandle ctrl_nh;
  ros::ServiceServer update_param_srv;    
  bool updateParamCallback(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res);
  
  double compute_target_speed();
  void compute_lookahead_distance(const double current_velocity);

  bool compute_target_point(const double & lookahead_distance, PathPoint & target_point_, int & near_idx);

  double compute_points_distance_squared(
    const PathPoint & point1,
    const PathPoint & point2);
  /// \brief Compute the relative y coordinate distance between two points
  /// \param[in] current The point with x and y position, and 2D heading information
  /// \param[in] target The point with x and y position, and 2D heading information
  /// \return the pair of relative x and y distances
   std::pair<double, double> compute_relative_xy_offset(
    const PathPoint & current,
    const PathPoint & target) const;
  /// \brief Compute the steering angle (radian) using the current pose and the target point
  /// \param[in] current_point The current position and velocity information
  /// \return the computed steering angle (radian)
   double compute_steering_rad();
  

  double m_lookahead_distance;
  PathPoint m_target_point;
  ackermann_msgs::AckermannDriveStamped cmd_msg;
  
  const double dt;
  Trajectory local_traj;
  VehicleState cur_state;
  


     double    minimum_lookahead_distance,
                maximum_lookahead_distance,
                speed_to_lookahead_ratio,    
                emergency_stop_distance,
                speed_thres_traveling_direction,
                max_acceleration,
                distance_front_rear_wheel,
                vel_lookahead_ratio;


     bool is_interpolate_lookahead_point, is_delay_compensation;
    


};  // class PurePursuit

#endif  // CONTROLLER_HPP_
