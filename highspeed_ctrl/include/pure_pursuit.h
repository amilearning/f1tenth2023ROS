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

#include <vector>
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <utility>
#include <ackermann_msgs/AckermannDriveStamped.h>
#include "state.h"


/// \brief Given a trajectory and the current state, compute the control command
class PurePursuit
{
public:  
  PurePursuit(const ros::NodeHandle& nh_traj);
  void update_vehicleState(const VehicleState & state);
  void update_ref_traj(const Trajectory & ref);
  void update_ref_path(const Trajectory & ref);
  ackermann_msgs::msg::AckermannDriveStamped compute_command();
  PathPoint get_target_point();
  void pathToTrajectory(
    const Path & ref_path, 
    Trajectory & ref_traj, 
    const std::vector<double>& ref_speed
);



private:
  
  rclcpp::Node::SharedPtr nh_;

  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr update_param_srv;
  void updateParamCallback(const std::shared_ptr<std_srvs::srv::Empty::Request> req, std::shared_ptr<std_srvs::srv::Empty::Response> res);
  double compute_target_speed();
  /// \brief Compute the lookahead distance using the current velocity and the conversion ratio
  ///        from the velocity to distance
  /// \param[in] current_velocity The current vehicle velocity
  void compute_lookahead_distance(const double current_velocity);
  /// \brief Whether the point is in the traveling direction.
  ///        If the value of the target velocity computed the previous update is positive,
  ///        the current target should be in the forward direction
  /// \param[in] current_point The current position and velocity information
  /// \param[in] point The candidate point in the trajectory
  /// \return True if the point is in the traveling direction
  bool in_traveling_direction(
    const PathPoint & current_point,
    const PathPoint & point) const;
  /// \brief Get the interpolated target point by computing the intersection between
  ///        line from the candidate target point to its previous point, and the circle.
  ///        The center of a circle is the current pose and the lookahead distance is a radius
  /// \param[in] current_point The current position and velocity information
  /// \param[in] target_point The candidate target point
  /// \param[in] idx The candidate target index in the trajectory
  void compute_interpolate_target_point(
    const PathPoint & current_point,
    const PathPoint & target_point,
    const uint32_t idx);
  /// \brief Compute the target point using the current pose and the trajectory
  /// \param[in] current_point The current position and velocity information
  /// \return True if the controller get the current target point
  //  bool compute_target_point(const PathPoint & current_point);
  bool compute_target_point(const double & lookahead_distance, PathPoint & target_point_, int & near_idx);
  /// \brief Compute the 2D distance between given two points
  /// \param[in] point1 The point with x and y position information
  /// \param[in] point2 The point with x and y position information
  /// \return the distance between given two points
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
  ackermann_msgs::msg::AckermannDriveStamped cmd_msg;
  
  const double dt;
  Trajectory ref_traj;
  Path ref_path;
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
