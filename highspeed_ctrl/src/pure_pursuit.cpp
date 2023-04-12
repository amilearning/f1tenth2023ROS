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



#include <algorithm>
#include <limits>
#include <utility>
#include "pure_pursuit.h"


////////////////////////////////////////////////////////////////////////////////
PurePursuit::PurePursuit(const ros::NodeHandle& nh_ctrl) : ctrl_nh(nh_ctrl), dt(0.05){
    
    update_param_srv = ctrl_nh.advertiseService("/pure_param_update", &PurePursuit::updateParamCallback, this);
    
  
    ctrl_nh.param<double>("/Pminimum_lookahead_distance",minimum_lookahead_distance, 0.5);
    ctrl_nh.param<double>("/Pmaximum_lookahead_distance", maximum_lookahead_distance,2.0);
    ctrl_nh.param<double>("/Pspeed_to_lookahead_ratio", speed_to_lookahead_ratio,1.2);
    ctrl_nh.param<double>("/Pemergency_stop_distance", emergency_stop_distance,0.0);
    // ctrl_nh.param<double>("Pspeed_thres_traveling_direction", 0.0);
    ctrl_nh.param<double>("/Pmax_acceleration", max_acceleration, 50.0);
    ctrl_nh.param<double>("/Pdistance_front_rear_wheel",distance_front_rear_wheel,  0.33);
    ctrl_nh.param<double>("/vel_lookahead_ratio",vel_lookahead_ratio,  1.0);
  



}


bool PurePursuit::updateParamCallback(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res)
{   
    ctrl_nh.getParam("vel_lookahead_ratio",vel_lookahead_ratio);    
    ctrl_nh.getParam("Pminimum_lookahead_distance",minimum_lookahead_distance);    
    ctrl_nh.getParam("Pmaximum_lookahead_distance",maximum_lookahead_distance);    
    ctrl_nh.getParam("Pspeed_to_lookahead_ratio",speed_to_lookahead_ratio);    
    ctrl_nh.getParam("Pemergency_stop_distance",emergency_stop_distance);    
    // nh_->get_parameter("Pspeed_thres_traveling_direction",speed_thres_traveling_direction);    
    ctrl_nh.getParam("Pmax_acceleration",max_acceleration);    
    ctrl_nh.getParam("Pdistance_front_rear_wheel",distance_front_rear_wheel);    
    ROS_INFO("Param has been updated");
    
    res.success = true;
    return true;  
}
////////////////////////////////////////////////////////////////////////////////


void PurePursuit::update_ref_traj(const Trajectory & ref){
  local_traj = ref;
 }

void PurePursuit::update_vehicleState(const VehicleState & state){
    cur_state = state;
}

void PurePursuit::readLookuptable(const std::string& filename){
  
  std::vector<double> x_data, y_data, z_data;
  if(lookup_tb.read_dictionary_file(filename, x_data,y_data,z_data))
    { 
      lookup_tb.setValues(x_data,y_data,z_data);}
    
}

double PurePursuit::getAngleDiffToTargetPoint(){
  double target_heading = std::atan2((m_target_point[1] -cur_state.pose.position.y),
  (m_target_point[0] -cur_state.pose.position.x));
  double velocity_heading;

  tf::Quaternion q(cur_state.pose.orientation.x,
                  cur_state.pose.orientation.y,
                  cur_state.pose.orientation.z,
                  cur_state.pose.orientation.w);
  q.normalize();
  // Extract the yaw angle from the quaternion object    
  double roll, pitch, yaw;
  tf::Matrix3x3(q).getRPY(roll, pitch, yaw);  
  bool odom_twist_in_local = true;
  if(odom_twist_in_local){
  Eigen::Matrix2d Rbg;
  Rbg << cos(-1*yaw), -sin(-1*yaw),
                        sin(-1*yaw), cos(-1*yaw);
  Eigen::Vector2d local_vx_vy;
  local_vx_vy << cur_state.vx, cur_state.vy; 
  Eigen::Vector2d global_vx_vy = Rbg*local_vx_vy;
   velocity_heading = std::atan2(global_vx_vy[1],global_vx_vy[0]);    
  }else{
    velocity_heading = std::atan2(cur_state.vy,cur_state.vx);
  }
    target_heading = normalizeRadian(target_heading);
    velocity_heading = normalizeRadian(velocity_heading);
    return normalizeRadian(target_heading-velocity_heading);
}



bool PurePursuit::getLookupTablebasedDelta(double& delta, const double&  diff_angel, const double& lookahead_dist, const double& vx, const double& vy){
  if(lookup_tb.is_ready){    
  double vt = sqrt(vx*vx + vy*vy);
  double desired_alat = 2*vt*vt*sin(diff_angel)/(lookahead_dist+1e-7);
  delta = lookup_tb.eval(desired_alat, vt);
  return true;
  }
  else{
    return false;
  }  
}

ackermann_msgs::AckermannDriveStamped PurePursuit::compute_model_based_command(){

  if (local_traj.size() < 2 && local_traj.x.size() < 2){
    // ROS_INFO("local traj size not enough");
    return cmd_msg; 
  }
  const auto start = std::chrono::system_clock::now();
//   TrajectoryPoint current_point = current_pose.state;  // copy 32bytes
  int near_idx;
  compute_lookahead_distance(cur_state.vx);  // update m_lookahead_distance 
  const auto is_success = compute_target_point(m_lookahead_distance, m_target_point, near_idx); // update target_point, near_idx
  
  if (is_success) {
    // m_command.long_accel_mps2 = compute_command_accel_mps(current_point, false);
        cmd_msg.header.stamp = ros::Time::now();
        // ackermann_msg.drive.speed =  opt_vel/5.0;
        cmd_msg.drive.speed =  compute_target_speed();
        
        if(lookup_tb.is_ready){
            double angle_diff = getAngleDiffToTargetPoint();
            double target_delta;
            getLookupTablebasedDelta(target_delta, angle_diff, m_lookahead_distance,cur_state.vx, cur_state.vy);
             cmd_msg.drive.steering_angle = target_delta;
        }else{
          cmd_msg.drive.steering_angle = compute_steering_rad();
        }        

  } else {
        cmd_msg.header.stamp = ros::Time::now();        
        cmd_msg.drive.speed =  0.0;
        cmd_msg.drive.steering_angle =cur_state.delta;
    
  }
  return cmd_msg;
}

ackermann_msgs::AckermannDriveStamped PurePursuit::compute_command()
{ 
  if (local_traj.size() < 2 && local_traj.x.size() < 2){
    // ROS_INFO("local traj size not enough");
    return cmd_msg; 
  }
  const auto start = std::chrono::system_clock::now();
//   TrajectoryPoint current_point = current_pose.state;  // copy 32bytes
  int near_idx;
  compute_lookahead_distance(cur_state.vx);  // update m_lookahead_distance 
  const auto is_success = compute_target_point(m_lookahead_distance, m_target_point, near_idx); // update target_point, near_idx
  
  if (is_success) {
    // m_command.long_accel_mps2 = compute_command_accel_mps(current_point, false);
        cmd_msg.header.stamp = ros::Time::now();
        cmd_msg.drive.speed =  compute_target_speed();
          cmd_msg.drive.steering_angle = compute_steering_rad();
 
  } else {
        cmd_msg.header.stamp = ros::Time::now();        
        cmd_msg.drive.speed =  0.0;
        cmd_msg.drive.steering_angle =cur_state.delta;
    
  }
  

  return cmd_msg;
}


visualization_msgs::Marker PurePursuit::getTargetPointhMarker(){
  
  std_msgs::ColorRGBA  lookahead_point_color;        
  lookahead_point_color.b = 1.0;
  lookahead_point_color.a = 1.0;
  visualization_msgs::Marker marker;
  marker.header.frame_id = "map";
  marker.header.stamp = ros::Time::now();
  marker.type = visualization_msgs::Marker::SPHERE;
  marker.action = visualization_msgs::Marker::ADD;
  // Set the pose of the marker to the position of the point
  marker.pose.position.x = m_target_point[0];
  marker.pose.position.y = m_target_point[1];
  marker.pose.position.z = 0.0;
  // Set the scale of the marker
  marker.scale.x = 0.3;
  marker.scale.y = 0.3;
  marker.scale.z = 0.1;

  // Set the color of the marker
  marker.color = lookahead_point_color;
  return marker;

 
 }

PathPoint PurePursuit::get_target_point(){
    return m_target_point;
} 

double PurePursuit::compute_target_speed(){
     PathPoint target_point;
     int near_idx;
     double vel_lookahed_dist = fabs(cur_state.vx*vel_lookahead_ratio);
    compute_target_point(vel_lookahed_dist, target_point, near_idx); 
    if (near_idx >= local_traj.size()-1){
        near_idx = local_traj.size()-1;
    }
    double target_speed = local_traj.vx[near_idx];
    
    return target_speed;
}
////////////////////////////////////////////////////////////////////////////////
void PurePursuit::compute_lookahead_distance(const double current_velocity)
{
  const double lookahead_distance = fabs(current_velocity * speed_to_lookahead_ratio);
  m_lookahead_distance =
    std::max(minimum_lookahead_distance,
    std::min(lookahead_distance, maximum_lookahead_distance));
}
////////////////////////////////////////////////////////////////////////////////




bool PurePursuit::compute_target_point(const double & lookahead_distance, PathPoint & target_point_, int & near_idx)
{   near_idx =0;
    PathPoint current_point;
    current_point << cur_state.pose.position.x , cur_state.pose.position.y;
    
  int idx = 0;
//   uint32_t last_idx_for_noupdate = 0U;
  int last_idx_for_noupdate = 0;
  
  for (idx = 0; idx <   local_traj.size(); ++idx) {
    PathPoint target_point_tmp;
    target_point_tmp << local_traj.x[idx], local_traj.y[idx];
    
    last_idx_for_noupdate = idx;

      // Search the closest point over the lookahead distance
      if (compute_points_distance_squared(current_point, target_point_tmp) >=lookahead_distance)
      {
        target_point_ = target_point_tmp;
        
        break;
      }
   
  }

  bool is_success = true;
  // If all points are within the distance threshold,
  if (idx == local_traj.size()) {
      // use the farthest target index in the traveling direction      
      target_point_ << local_traj.x[last_idx_for_noupdate], local_traj.y[last_idx_for_noupdate];      
  }

    // if the target point is too close to stop within 0.2sec 
  if(compute_points_distance_squared(current_point, target_point_) < cur_state.vx*0.2){
    is_success = false;
  }
  near_idx = last_idx_for_noupdate;

  return is_success;
}
////////////////////////////////////////////////////////////////////////////////
double PurePursuit::compute_points_distance_squared(
  const PathPoint & point1,
  const PathPoint & point2)
{
    return sqrt((point1[0] - point2[0])*(point1[0] - point2[0]) + (point1[1] - point2[1])*(point1[1] - point2[1]) );
    
}
////////////////////////////////////////////////////////////////////////////////
std::pair<double, double> PurePursuit::compute_relative_xy_offset(
  const PathPoint & current,
  const PathPoint & target) const
{
  const auto diff_x = target[0] - current[0];
  const auto diff_y = target[1] - current[1];
  const auto yaw = cur_state.yaw;  
  const auto cos_pose = std::cos(yaw);
  const auto sin_pose = std::sin(yaw);
  const auto relative_x = static_cast<double>((cos_pose * diff_x) + (sin_pose * diff_y));
  const auto relative_y = static_cast<double>((-sin_pose * diff_x) + (cos_pose * diff_y));
  const std::pair<double, double> relative_xy(relative_x, relative_y);
  return relative_xy;
}
////////////////////////////////////////////////////////////////////////////////
double PurePursuit::compute_steering_rad()
{

    PathPoint current_point;
        current_point << cur_state.pose.position.x, cur_state.pose.position.y;
        
  // Compute the steering angle by arctan(curvature * wheel_distance)
  // link: https://www.ri.cmu.edu/pub_files/2009/2/
  //       Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf
  const double denominator = compute_points_distance_squared(current_point, m_target_point);
  
  const double numerator = compute_relative_xy_offset(current_point, m_target_point).second;
  
  constexpr double epsilon = 0.0001F;
  // equivalent to (2 * y) / (distance * distance) = (2 * sin(th)) / distance
  const double curvature = (denominator > epsilon) ? ((2.0F * numerator) / denominator) : 0.0F;
  
  
  const double steering_angle_rad = atanf(curvature * distance_front_rear_wheel);
  return steering_angle_rad;
}
////////////////////////////////////////////////////////////////////////////////

