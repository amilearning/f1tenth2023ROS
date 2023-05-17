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
#include <tf/tf.h>
#include "std_srvs/Trigger.h"
#include <vector>
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <utility>
#include <ackermann_msgs/AckermannDriveStamped.h>
#include <sensor_msgs/LaserScan.h>
#include "std_msgs/ColorRGBA.h"
#include "state.h"
#include "trajectory.h"
#include "utils.h"

#include "lowpass_filter.h"
#include <visualization_msgs/Marker.h>

#include <iostream>
#include <fstream>
#include <string>


enum RaceMode { Race = 0, Overtaking = 1, Following = 2 };
// 




class BicubicSplineLookupTable{
public:
    BicubicSplineLookupTable() : spline(nullptr), xacc(nullptr), yacc(nullptr), nx(0), ny(0), is_ready(false) {}

    void setValues(std::vector<double>& x, std::vector<double>& y, std::vector<double>& z) {
        const gsl_interp2d_type *T = gsl_interp2d_bicubic;
        nx = x.size();
        ny = y.size();

        if (spline != nullptr) {
            gsl_spline2d_free(spline);
        }
        if (xacc != nullptr) {
            gsl_interp_accel_free(xacc);
        }
        if (yacc != nullptr) {
            gsl_interp_accel_free(yacc);
        }

        spline = gsl_spline2d_alloc(T, nx, ny);
        xacc = gsl_interp_accel_alloc();
        yacc = gsl_interp_accel_alloc();

        gsl_spline2d_init(spline, x.data(), y.data(), z.data(), nx, ny);
        ROS_INFO("PP lookuptable ready");
        is_ready = true;
    }

    double eval(double xval, double yval) {
      if(!is_ready){
        ROS_WARN("Lookuptable is not ready" );        
        return 0.0;
      }
        return gsl_spline2d_eval(spline, xval, yval, xacc, yacc);
    }

 

bool read_dictionary_file(const std::string& filename, std::vector<double>& x, std::vector<double>& y, std::vector<double>& z) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Warning: unable to open file " << filename << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        size_t pos = line.find("=");
        if (pos == std::string::npos) {
            continue;
        }
        std::string field = line.substr(0, pos);
        std::string values_str = line.substr(pos+1);
        std::istringstream iss(values_str);

        if (field == "alat") {
            double value;
            while (iss >> value) {
                x.push_back(value);
                if (iss.peek() == ',') {
                    iss.ignore();
                }
            }
        }
        else if (field == "vx") {
            double value;
            while (iss >> value) {
                y.push_back(value);
                if (iss.peek() == ',') {
                    iss.ignore();
                }
            }
        }
        else if (field == "delta") {
            double value;
            while (iss >> value) {
                z.push_back(value);
                if (iss.peek() == ',') {
                    iss.ignore();
                }
            }
        }
    }

    file.close();

    if (z.size() != x.size() * y.size()) {
        std::cerr << "Warning: size of extracted z array is not equal to the multiplication of the size of x and y" << std::endl;
        return false;
    }else{
        auto vx_min_max = std::minmax_element(y.begin(), y.end());
        auto alat_min_max = std::minmax_element(x.begin(), x.end());
        auto delta_min_max = std::minmax_element(z.begin(), z.end());
        vx_min = *vx_min_max.first;
        vx_max = *vx_min_max.second;
        alat_min = *alat_min_max.first;
        alat_max = *alat_min_max.second;
        delta_min = *delta_min_max.first;
        delta_max = *delta_min_max.second;
        std::cout << "vx_min  = " << vx_min << std::endl;
        std::cout << "vx_max  = " << vx_max << std::endl;
        std::cout << "alat_min  = " << alat_min << std::endl;
        std::cout << "alat_max  = " << alat_max << std::endl;
        std::cout << "delta_min  = " << delta_min << std::endl;
        std::cout << "delta_max  = " << delta_max << std::endl;
        
      return true;
    }
    
}

   ~BicubicSplineLookupTable() {
        if (spline != nullptr) {
            gsl_spline2d_free(spline);
        }
        if (xacc != nullptr) {
            gsl_interp_accel_free(xacc);
        }
        if (yacc != nullptr) {
            gsl_interp_accel_free(yacc);
        }
    }

    bool is_ready;
      double vx_min, vx_max;
    double alat_min, alat_max;
    double delta_min, delta_max;
private:
    gsl_spline2d *spline;
    gsl_interp_accel *xacc;
    gsl_interp_accel *yacc;
    size_t nx;
    size_t ny;     
    
};

// std::vector<double> x = {-2.0, -1.0, 0.0, 1.0, 2.0};
// std::vector<double> y = {-2.0, -1.0, 0.0, 1.0, 2.0};
// std::vector<double> z = {0.0, 3.0, 4.0, 3.0, 0.0, -3.0, 0.0, 1.0, 0.0, -3.0, -4.0, -1.0, 0.0, -1.0, -4.0, -3.0, 0.0, 1.0, 0.0, -3.0, 0.0, 3.0, 4.0, 3.0, 0.0};

// BiCubicSpline spline;
// spline.setValues(x, y, z);
// double val = spline.eval(-1.5, 1.0);



  ///////////////

/// \brief Given a trajectory and the current state, compute the control command
class PurePursuit
{
public:  
  PurePursuit(const ros::NodeHandle& nh_ctrl);
  void update_vehicleState(const VehicleState & state);
  void update_obstacleState(const VehicleState & state);
  void update_ref_traj(const Trajectory & ref);
  void readLookuptable(const std::string& filename);
  bool is_there_obstacle;
  
  ackermann_msgs::AckermannDriveStamped compute_command();
  ackermann_msgs::AckermannDriveStamped compute_model_based_command();
  ackermann_msgs::AckermannDriveStamped compute_lidar_based_command(bool & is_straight, const sensor_msgs::LaserScan::ConstPtr laser_data);
  bool getOvertakingStatus();
  PathPoint get_target_point();
  PathPoint get_speed_target_point();
  visualization_msgs::Marker getTargetPointhMarker(int point_idx);
    void set_manual_lookahead(const bool target_switch, const bool speed_switch, const double dist_lookahead,const double speed_lookahead,const double max_a_lat_);
    double max_acceleration;

private:
  
  ros::NodeHandle ctrl_nh;
  ros::ServiceServer update_param_srv;    
  bool updateParamCallback(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res);

    bool ObstacleAvoidance(PathPoint & target_point_, int near_idx);
  void vel_clip_accel(double & ref_vel);
  double compute_target_speed(double vel_lookahead_ratio_);
  void compute_lookahead_distance(const double reference_velocity);

  double refine_target_vel_via_curvature(const double init_vel, const int & target_wp_idx);
  bool compute_target_point(const double & lookahead_distance, PathPoint & target_point_, int & near_idx);
  bool getLookupTablebasedDelta(double& delta, const double&  diff_angel, const double& lookahead_dist, const double& vx, const double& vy);
  double getAngleDiffToTargetPoint();
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

 
   ros::Publisher debug_pub;
  
    
    RaceMode race_mode;
    bool obstacle_avoidance_activate;
  double m_lookahead_distance;
  PathPoint m_target_point, m_speed_target_point;
  ackermann_msgs::AckermannDriveStamped cmd_msg;
  Butterworth2dFilter lookahead_dist_filter;
  double filt_lookahead;
  const double dt;
  Trajectory local_traj;
  VehicleState cur_state, cur_obstacle;
  BicubicSplineLookupTable lookup_tb;
    bool manual_target_lookahead, manual_speed_lookahead;
    double manual_target_lookahead_value, manual_speed_lookahead_value;
    double max_a_lat;
     double    minimum_lookahead_distance,
                maximum_lookahead_distance,
                speed_to_lookahead_ratio,    
                emergency_stop_distance,
                speed_thres_traveling_direction,                
                distance_front_rear_wheel,                
                vel_lookahead_ratio,
                speed_minimum_lookahead_distance,
                speed_maximum_lookahead_distance;


     bool is_interpolate_lookahead_point, is_delay_compensation;
    


};  // class PurePursuit

#endif  // CONTROLLER_HPP_
