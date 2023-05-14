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


double clamp(double value, double min_val, double max_val) {
    if (value < min_val) {
        return min_val;
    }
    else if (value > max_val) {
        return max_val;
    }
    else {
        return value;
    }
}


////////////////////////////////////////////////////////////////////////////////
PurePursuit::PurePursuit(const ros::NodeHandle& nh_ctrl) : ctrl_nh(nh_ctrl), dt(0.05), is_there_obstacle(false), obstacle_avoidance_activate(false),race_mode(RaceMode::Race){
    
    update_param_srv = ctrl_nh.advertiseService("/pure_param_update", &PurePursuit::updateParamCallback, this);
    
    double lookahead_filter_cutoff = 1;
    lookahead_dist_filter.initialize(0.05, lookahead_filter_cutoff);

    debug_pub = ctrl_nh.advertise<geometry_msgs::PoseStamped>("/pp_debug",1);
    ctrl_nh.param<double>("Pminimum_lookahead_distance",minimum_lookahead_distance, 0.5);
    ctrl_nh.param<double>("Pmaximum_lookahead_distance", maximum_lookahead_distance,2.0);
    // ctrl_nh.param<double>("Pspeed_to_lookahead_ratio", speed_to_lookahead_ratio,1.2);
    ctrl_nh.param<double>("Pemergency_stop_distance", emergency_stop_distance,0.0);
    
    ctrl_nh.param<double>("speed_minimum_lookahead_distance", speed_minimum_lookahead_distance,0.0);
    ctrl_nh.param<double>("speed_maximum_lookahead_distance", speed_maximum_lookahead_distance,0.4);
    
    // ctrl_nh.param<double>("Pspeed_thres_traveling_direction", 0.0);
    ctrl_nh.param<double>("Pmax_acceleration", max_acceleration, 50.0);
    ctrl_nh.param<double>("Pdistance_front_rear_wheel",distance_front_rear_wheel,  0.33);
    ctrl_nh.param<double>("vel_lookahead_ratio",vel_lookahead_ratio,  1.0);
  



}


bool PurePursuit::updateParamCallback(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res)
{   
    ctrl_nh.getParam("vel_lookahead_ratio",vel_lookahead_ratio);    
    ctrl_nh.getParam("Pminimum_lookahead_distance",minimum_lookahead_distance);    
    ctrl_nh.getParam("Pmaximum_lookahead_distance",maximum_lookahead_distance);    
    ctrl_nh.getParam("speed_minimum_lookahead_distance",speed_minimum_lookahead_distance);    
    ctrl_nh.getParam("speed_maximum_lookahead_distance",speed_maximum_lookahead_distance);  
    // ctrl_nh.getParam("Pspeed_to_lookahead_ratio",speed_to_lookahead_ratio);    
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

void PurePursuit::update_obstacleState(const VehicleState & state){
    cur_obstacle = state;
}

void PurePursuit::readLookuptable(const std::string& filename){
  
  std::vector<double> x_data, y_data, z_data;
  // alat, vx, delta 

  if(lookup_tb.read_dictionary_file(filename, x_data,y_data,z_data))
    { 
      lookup_tb.setValues(x_data,y_data,z_data);
      }
    
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
  yaw = normalizeRadian(yaw);
  double speed = sqrt(cur_state.vx*cur_state.vx + cur_state.vy*cur_state.vy);
  // if(speed < 0.2){
  //   // speed is too low, replace with yaw angle 
  //   velocity_heading = yaw;
  // }else{
      bool odom_twist_in_local = true;
      if(odom_twist_in_local){
      Eigen::Matrix2d Rbg;
      Rbg << cos(yaw), -sin(yaw),
                            sin(yaw), cos(yaw);
      Eigen::Vector2d local_vx_vy;
      local_vx_vy << cur_state.vx, cur_state.vy; 
      Eigen::Vector2d global_vx_vy = Rbg*local_vx_vy;

      velocity_heading = std::atan2(global_vx_vy[1],global_vx_vy[0]);    
      }else{
        velocity_heading = std::atan2(cur_state.vy,cur_state.vx);
      }
  // }
  
 

  
  
  
    target_heading = normalizeRadian(target_heading);
    velocity_heading = normalizeRadian(velocity_heading);

  /////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////// check if the velocity vector angle is far away from yaw .... (Assumption: we do not dfirt!!!)
  // double diff_yaw_to_vel = normalizeRadian(velocity_heading - yaw);
  // double max_diff_angle = 30 * 3.14195 / 180.0;
  // if(diff_yaw_to_vel > max_diff_angle){
  //   std::cout << "diff too much " << std::endl;
  //   velocity_heading = yaw + max_diff_angle;
  // }else if( diff_yaw_to_vel < -1*max_diff_angle){
  //   std::cout << "diff too less " << std::endl;
  //   velocity_heading = yaw - max_diff_angle;
  // }
  // velocity_heading = normalizeRadian(velocity_heading);
  /////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  double diff_heading = normalizeRadian(target_heading-velocity_heading);

  
  // geometry_msgs::PoseStamped debug_msg;
  // debug_msg.header.stamp = ros::Time::now();
  // debug_msg.pose.position.x = target_heading*180/M_PI;
  // debug_msg.pose.position.y = velocity_heading*180/M_PI;
  // debug_msg.pose.position.z = diff_heading*180/M_PI;
  // debug_pub.publish(debug_msg); 

    return diff_heading;
}



bool PurePursuit::getLookupTablebasedDelta(double& delta, const double&  diff_angel, const double& lookahead_dist, const double& vx, const double& vy){
  if(lookup_tb.is_ready){ 
       
  double vt = sqrt(vx*vx + vy*vy);
  // if (vt < 1.0) {
  //   vt = 0.01;
  // }
  //  if (vt > 1.5){
  //   vt = 1.5;
  // }
  
  // unsigned alat 
  
  double signed_desired_alat = 2*vt*vt*sin(diff_angel)/(lookahead_dist+1e-10);
  double desired_alat = abs(signed_desired_alat);
  // std::cout << " unsigned desired_alat = " << desired_alat << std::endl;
  
  vt = clamp(vt, lookup_tb.vx_min, lookup_tb.vx_max);
  
  desired_alat = clamp(desired_alat, lookup_tb.alat_min, lookup_tb.alat_max);
  
   
  ////////////// GGUMZZIKHAE
  // std::cout << " clamped vt = " << vt << std::endl;
  // std::cout << " clamped desired_alat = " << desired_alat << std::endl;

  double unsigned_delta = lookup_tb.eval(desired_alat, vt);
  
  
  if (signed_desired_alat  < 0){
    delta = abs(unsigned_delta);

  }else{
    delta = -1*abs(unsigned_delta);
  }


  // geometry_msgs::PoseStamped debug_msg;
  // debug_msg.header.stamp = ros::Time::now();
  // debug_msg.pose.position.x = vt;
  // if (signed_desired_alat > 0){
  //   debug_msg.pose.position.y = signed_desired_alat;
  // }else{
  //   debug_msg.pose.position.y = -1*signed_desired_alat;
    
  // }  
  // debug_msg.pose.position.z = delta;
  // debug_msg.pose.orientation.x = diff_angel;
  // debug_pub.publish(debug_msg);
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
        cmd_msg.drive.speed =  compute_target_speed(vel_lookahead_ratio);
        
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

bool PurePursuit::getOvertakingStatus(){
  return obstacle_avoidance_activate;
}



ackermann_msgs::AckermannDriveStamped PurePursuit::compute_command()
{ 
  if (local_traj.size() < 2 && local_traj.x.size() < 2){
    // ROS_INFO("local traj size not enough");
    return cmd_msg; 
  }
  const auto start = std::chrono::system_clock::now();
  // Compute the initial guess of target velocity from the path info 
       // given initial velocity guess, lookahead distance is computed for steering(independent of current velocity) 
       // Given lookahead distance, we extract the curvature of that lookahead point 
       // if the lookahead distance is greater than some threshold, and the curavture at that point is great 
        // we decress the velocity from initial guess for fast decelleration

  //   TrajectoryPoint current_point = current_pose.state;  // copy 32bytes
  int near_idx;
  double adaptive_vel_lookahead_ratio;
  if(fabs(cur_state.k) < 0.2 ){
    // if we drive on straight line
    adaptive_vel_lookahead_ratio = 2*  vel_lookahead_ratio;
  }else{
     // if we drive on curvy road 
    adaptive_vel_lookahead_ratio = vel_lookahead_ratio/2;
  }
  
  

  double target_vel_init_guess = compute_target_speed(adaptive_vel_lookahead_ratio);
  compute_lookahead_distance(target_vel_init_guess);  // update m_lookahead_distance  for target speed reference 
  // compute_lookahead_distance(cur_state.vx); 
  
  auto is_success = compute_target_point(m_lookahead_distance, m_target_point, near_idx); // update target_point, near_idx
  

  if (is_success) {
      double target_vel = refine_target_vel_via_curvature(target_vel_init_guess, near_idx);
      if (target_vel != target_vel_init_guess){
      }

        // filter vx if ey is high 
    ///  TODO: or we can reduce speed to certain value if such case.. 
     // if accel , limit vel via maximum acceleration 
     if(cur_state.ey > 0.2 || cur_state.epsi > 20*3.14195/180.0){
      // ROS_INFO("ey or epsi increased");
      ///  TODO: or we can reduce speed to certain value if such case.. 
      if(target_vel > cur_state.vx ){
        double cliped_vel_cmd = cur_state.vx+max_acceleration;      
        target_vel  = std::min(cliped_vel_cmd, target_vel);
        // std::cout << "limit vel " << cliped_vel_cmd << std::endl;
      }
     }

    // hard constraint for recovery 
     if(cur_state.ey > 1.0 ){
      target_vel  =1.5;
     }else if(cur_state.ey > 0.5 && cur_state.ey < 1.0){
      target_vel  =2.0;
     }


      compute_lookahead_distance(cur_state.vx);                                          // true lookahead for steering
      filt_lookahead = lookahead_dist_filter.filter(m_lookahead_distance);
      double cur_speed = sqrt(cur_state.vx*cur_state.vx+cur_state.vy*cur_state.vy);
      if(cur_speed > target_vel){
        m_lookahead_distance = filt_lookahead;
      }

    //   geometry_msgs::PoseStamped debug_msg;
    // debug_msg.header.stamp = ros::Time::now();
    // debug_msg.pose.position.x = m_lookahead_distance;
    // debug_pub.publish(debug_msg);


      is_success = compute_target_point(m_lookahead_distance, m_target_point, near_idx); // update target_point, near_idx
                                                                                         // m_command.long_accel_mps2 = compute_command_accel_mps(current_point, false);

      obstacle_avoidance_activate = ObstacleAvoidance(m_target_point,near_idx);                                                                                         
      if (obstacle_avoidance_activate){
          target_vel = 0.0;
      }
      cmd_msg.header.stamp = ros::Time::now();
      cmd_msg.drive.speed = target_vel;
      cmd_msg.drive.steering_angle = compute_steering_rad();
 
  } else {
        cmd_msg.header.stamp = ros::Time::now();        
        cmd_msg.drive.speed =  0.0;
        cmd_msg.drive.steering_angle =cur_state.delta;
    
  }
  
//
    
      

  return cmd_msg;
}

ackermann_msgs::AckermannDriveStamped PurePursuit::compute_lidar_based_command(bool & is_straight, const sensor_msgs::LaserScan::ConstPtr laser_data){
    // get the closest distance lidar index on the right side
    // until the middle point of laser scan find the lookahead point given distance      
    // reactive obstacle avoidance if there is any
    int middle_idx = int(laser_data->ranges.size()/2);
    int right_normal_idx = middle_idx + int(90/laser_data->angle_increment);
     // right side laser scan
     double dist_tmp = 1e4;
     double min_idx;
    for(int i=middle_idx; i < right_normal_idx; i++){
          if(dist_tmp > laser_data->ranges[i]){
            min_idx = i;
            dist_tmp = laser_data->ranges[i];
          }
    }

    // lookahead distance computed given the current speed 
    // minimum of 1.0m/s
     double lookahead_distance = cur_state.vx *1.8-2.9;
      lookahead_distance =
      std::max(1.0,
      std::min(lookahead_distance, maximum_lookahead_distance));
      if(manual_target_lookahead){
        lookahead_distance = manual_target_lookahead_value;
      }

    // find the index which is used for line searching
      dist_tmp = 0;
      double min_angle = (laser_data->angle_increment)*min_idx+laser_data->angle_min;
      double x_min = laser_data->ranges[min_idx]*cos(min_angle);
      double y_min = laser_data->ranges[min_idx]*sin(min_angle);
      double target_idx = middle_idx;
      for(int i=min_idx; i >middle_idx; i--){
        double angle_tmp = (laser_data->angle_increment)*i+laser_data->angle_min;
        double x_tmp = laser_data->ranges[i]*cos(angle_tmp);
        double y_tmp = laser_data->ranges[i]*sin(angle_tmp);
        double dist_tmp = sqrt((x_min-x_tmp)*(x_min-x_tmp) + (y_min-y_tmp)*(y_min-y_tmp));
        if(dist_tmp > lookahead_distance*2){
          target_idx = i;
          break;
        } 
      }

      // fit lines 
      std::vector<double> x_laser, y_laser;
      for(int i=min_idx; i > target_idx ; i--){
       double angle_tmp = (laser_data->angle_increment)*i+laser_data->angle_min;
       double x_ = laser_data->ranges[i]*cos(angle_tmp+cur_state.yaw);
       double y_ = laser_data->ranges[i]*sin(angle_tmp+cur_state.yaw);
       x_laser.push_back(x_);
       y_laser.push_back(y_);
      }

      is_straight = false;
      double  cost,  slope,  y_intercept;
      estimateLineEquation(cost,  slope,  y_intercept,x_laser,y_laser);
      std::cout << "Line equation: y = " << slope << "x + " << y_intercept << std::endl;
      std::cout << "Fit cost: " << cost << std::endl;

}


bool PurePursuit::ObstacleAvoidance(PathPoint & target_point_, int near_idx){
  if(!is_there_obstacle){
    
    return false;
    
  }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////Obstacle avoidance refinement//////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  double min_dist_to_local_path = 1e3;
  int min_idx = 0;
  for(int k=0; k < local_traj.x.size(); k++){
      double tmp_dist = sqrt((cur_obstacle.pose.position.x-local_traj.x[k])*(cur_obstacle.pose.position.x - local_traj.x[k]) + (cur_obstacle.pose.position.y-local_traj.y[k])*(cur_obstacle.pose.position.y-local_traj.y[k]));
      if(tmp_dist < min_dist_to_local_path){
        min_dist_to_local_path = tmp_dist;
        min_idx = k;
      }
  }
  double width_safe_dist = 0.3;
  race_mode = RaceMode::Race;
  int refined_idx = std::min(near_idx,min_idx); // closer index is set as taret idx

      
  geometry_msgs::PoseStamped debug_msg;
  debug_msg.header.stamp = ros::Time::now();
  debug_msg.pose.position.x = cur_obstacle.s;
  debug_msg.pose.position.y = -1*local_traj.ey_r[refined_idx];
  debug_msg.pose.position.z = local_traj.ey_l[refined_idx];
  debug_msg.pose.orientation.x = cur_obstacle.ey ;
  debug_pub.publish(debug_msg); 


    // Obstacle is inside of track width --> overtaking actiavted 
     double target_ey = 0;
      if(cur_obstacle.ey > 0){ // obstacle is in the left side of centerline 
              target_ey = cur_obstacle.ey -width_safe_dist*2;              
              double track_right_cosntaint = local_traj.ey_r[refined_idx]-width_safe_dist;
              track_right_cosntaint = std::max(track_right_cosntaint, 0.0);
              target_ey = std::max(std::min(target_ey, 0.0), -1*track_right_cosntaint);                
      }else{
        // obstacle is in the right side of centerline 
        target_ey = cur_obstacle.ey+width_safe_dist*2;
        double track_left_constraint = local_traj.ey_l[refined_idx] - width_safe_dist;
        track_left_constraint = std::max(track_left_constraint, 0.0);
        target_ey = std::max(std::min(target_ey, track_left_constraint), 0.0);                
      } 
      
      
    // if(abs(cur_obstacle.ey) > width_safe_dist){
      
      race_mode = RaceMode::Overtaking;
      // Aggresive Overtaking Action!!
      ROS_WARN("OVVertaking !!");
    // }else{      
    //   // Timid Following Action !!! 
    //   race_mode = RaceMode::Following;
    //   // ROS_INFO("Following");  
    //   target_ey = std::max(std::min(target_ey, 0.1), -0.1);          
    // }

    double yaw_on_centerline = local_traj.yaw[refined_idx];
    double new_x, new_y;
      if(target_ey >= 0){
      new_x = local_traj.x[refined_idx]+ fabs(target_ey)*cos(M_PI/2.0+yaw_on_centerline); 
      new_y = local_traj.y[refined_idx]+ fabs(target_ey)*sin(M_PI/2.0+yaw_on_centerline);
      
      }else{
      new_x = local_traj.x[refined_idx]+ fabs(target_ey)*cos(-M_PI/2.0+yaw_on_centerline); 
      new_y = local_traj.y[refined_idx]+ fabs(target_ey)*sin(-M_PI/2.0+yaw_on_centerline);
      
      }
      
      // target_point_ << new_x, new_y;

  return true;
    
  
    

     
}

visualization_msgs::Marker PurePursuit::getTargetPointhMarker(int point_idx){
  double x_, y_;
    std_msgs::ColorRGBA  lookahead_point_color;
  if (point_idx > 0){
    x_ = m_target_point[0];
    y_ = m_target_point[1];
  
  lookahead_point_color.b = 1.0;
  lookahead_point_color.a = 1.0;
  }else{
    x_ = m_speed_target_point[0];
    y_ = m_speed_target_point[1];          
  lookahead_point_color.r = 1.0;
  lookahead_point_color.b = 1.0;
  lookahead_point_color.a = 1.0;
  }

  visualization_msgs::Marker marker;
  marker.header.frame_id = "map";
  marker.header.stamp = ros::Time::now();
  marker.type = visualization_msgs::Marker::SPHERE;
  marker.action = visualization_msgs::Marker::ADD;
  // Set the pose of the marker to the position of the point
  marker.pose.position.x = x_;
  marker.pose.position.y = y_;
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

PathPoint PurePursuit::get_speed_target_point(){
    return m_speed_target_point;
} 

void PurePursuit::set_manual_lookahead(const bool target_switch, const bool speed_switch, const double dist_lookahead,const double speed_lookahead, const double max_a_lat_){
   manual_target_lookahead = target_switch;
   manual_speed_lookahead = speed_switch;
   manual_target_lookahead_value = dist_lookahead;
   manual_speed_lookahead_value = speed_lookahead;
   max_a_lat = max_a_lat_;
}

double PurePursuit::compute_target_speed(double vel_lookahead_ratio_){
    //  PathPoint target_point;
     int near_idx;
     
     double vel_lookahed_dist = fabs(cur_state.vx*vel_lookahead_ratio_);
       vel_lookahed_dist =
    std::max(speed_minimum_lookahead_distance,
    std::min(vel_lookahed_dist, speed_maximum_lookahead_distance));
    
    if(manual_speed_lookahead){
     vel_lookahed_dist =  manual_speed_lookahead_value;
    }
    compute_target_point(vel_lookahed_dist, m_speed_target_point, near_idx); 
    if (near_idx >= local_traj.size()-1){
        near_idx = local_traj.size()-1;
    }
    double target_speed = local_traj.vx[near_idx];
    
    return target_speed;
}
////////////////////////////////////////////////////////////////////////////////
void PurePursuit::compute_lookahead_distance(const double reference_velocity)
{ 
  double target_vel = cur_state.vx;
  if (fabs(cur_state.ey) > 0.2){
    target_vel = std::max(cur_state.vx, reference_velocity);
  }
  
  // const double lookahead_distance = fabs(current_velocity * speed_to_lookahead_ratio);
  const double lookahead_distance = target_vel *1.8-2.9;
  m_lookahead_distance =
    std::max(minimum_lookahead_distance,
    std::min(lookahead_distance, maximum_lookahead_distance));
   
    if(manual_target_lookahead){
      m_lookahead_distance = manual_target_lookahead_value;

    }
    
}
////////////////////////////////////////////////////////////////////////////////


double PurePursuit::refine_target_vel_via_curvature(const double init_vel, const int & target_wp_idx){
  double target_wp_curvature = fabs(local_traj.k[target_wp_idx]);  
  if(m_lookahead_distance > 0.0){ // 
      double max_vel = sqrt(max_a_lat / (abs(target_wp_curvature)+1e-5));
      double refined_vel = std::max(0.0,
                            std::min(init_vel, max_vel));
      return refined_vel;
  } 
  
  return init_vel;
} 

bool PurePursuit::compute_target_point(const double & lookahead_distance, PathPoint & target_point_, int & near_idx)
{   near_idx =0;
    PathPoint current_point;
    current_point << cur_state.pose.position.x , cur_state.pose.position.y;
    
  int idx = 0;
//   uint32_t last_idx_for_noupdate = 0U;
  int last_idx_for_noupdate = 0;
  bool find_within_lap = false;
  double cum_dist = 0;
  for (idx = 0; idx <   local_traj.size(); ++idx) {
    PathPoint target_point_tmp;
    target_point_tmp << local_traj.x[idx], local_traj.y[idx];
    
    last_idx_for_noupdate = idx;

      // Search the closest point over the lookahead distance
      cum_dist = compute_points_distance_squared(current_point, target_point_tmp); 
      if (cum_dist >=lookahead_distance)
      {
        target_point_ = target_point_tmp;
        find_within_lap = true;
        break;
      }
   
  }

  bool is_success = true;

  // THiS CODE IS NOT CORRECT ... ShOULD BE DONE while extracing local trajectory from path manager 
        // if we meet the end of trajectory recompute from the initial 
  // if(!find_within_lap){
  //   // ROS_INFO("finish line close");
   
  //   PathPoint init_point;
  //   init_point <<  local_traj.x[0], local_traj.y[0];
  //   for(int iidx = 0;iidx <  local_traj.size(); ++iidx){
  //     PathPoint target_point_tmp;
  //     target_point_tmp << local_traj.x[iidx], local_traj.y[iidx];
  //     last_idx_for_noupdate = iidx;
  //     if (compute_points_distance_squared(init_point, target_point_tmp) >=lookahead_distance-cum_dist)
  //     {
  //       target_point_ = target_point_tmp;        
  //       break;
  //     }

  //   }
    
  // }

  


  // If all points are within the distance threshold,
  // if (idx == local_traj.size()) {
  //     // use the farthest target index in the traveling direction      
  //     target_point_ << local_traj.x[last_idx_for_noupdate], local_traj.y[last_idx_for_noupdate];      
  // }

    // if the target point is too close to stop within 0.2sec 
  // if(compute_points_distance_squared(current_point, target_point_) < cur_state.vx*0.2){
  //   is_success = false;
  // }
  near_idx = last_idx_for_noupdate;
  target_point_ << local_traj.x[near_idx], local_traj.y[near_idx];
  // ROS_INFO("near_idx = %d", near_idx);
  // ROS_INFO("target_point_ = %f,   %f", target_point_[near_idx], target_point_[near_idx]);
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
  return -1*steering_angle_rad;
}
////////////////////////////////////////////////////////////////////////////////

