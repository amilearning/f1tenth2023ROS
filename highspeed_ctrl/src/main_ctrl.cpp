
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


#include "main_ctrl.h"


using namespace std;

Ctrl::Ctrl(ros::NodeHandle& nh_ctrl, ros::NodeHandle& nh_traj,ros::NodeHandle& nh_state, ros::NodeHandle& nh_p_):
  nh_p(nh_p_),
  nh_ctrl_(nh_ctrl),
  nh_traj_(nh_traj),
  nh_state_(nh_state),
  traj_manager(nh_traj),
  pp_ctrl(nh_p_),
  my_steering_ok_(false),
  my_position_ok_(false),
  ego_vehicle(0.1),
  ctrl_select(0),
  first_pose_received(false),
  first_odom_received(false),
  is_odom_used(false),
  imu_received(false),
  first_traj_received(false)
{
  
  
  nh_p.param<double>("odom_pose_diff_threshold", odom_pose_diff_threshold, 1.0);
  nh_p.param<double>("x_vel_filter_cutoff", x_vel_filter_cutoff, 10.0);
  nh_p.param<double>("y_vel_filter_cutoff", y_vel_filter_cutoff, 10.0);


    start_line_time.push_back(ros::Time::now().toSec());

    double filter_dt = 0.01;
    
    x_vel_filter.initialize(filter_dt, x_vel_filter_cutoff);
    y_vel_filter.initialize(filter_dt, y_vel_filter_cutoff);
    


  nh_p.param<std::string>("status_topic", status_topic, "/vehicle_status");  
  nh_p.param<std::string>("control_topic", control_topic, "/vesc/ackermann_cmd");
  nh_p.param<std::string>("waypoint_topic", waypoint_topic, "/local_traj");
  // nh_p.param<std::string>("odom_topic", odom_topic, "/odom");   
  nh_p.param<std::string>("odom_topic", odom_topic, "/pose_estimate");   
  nh_p.param<int>("path_smoothing_times_", path_smoothing_times_, 1);
  
  nh_p.param<int>("path_filter_moving_ave_num_", path_filter_moving_ave_num_, 35);  


  nh_p.param<double>("lookahead_path_length", lookahead_path_length, 5.0);
  
  nh_p.param<double>("wheelbase", wheelbase, 0.33);
  nh_p.param<double>("lf", lf, 0.165);
  nh_p.param<double>("lr", lr, 0.165);
  nh_p.param<double>("mass", mass, 2.0);    
  nh_p.param<double>("dt", dt, 0.04); 

  nh_p.param<std::string>("pp_lookup_table_file_name", pp_lookup_table_file_name, "lookuptb.txt");   

  std::string package_path = ros::package::getPath("highspeed_ctrl");
  std::string lookuptable = package_path + "/path/" +pp_lookup_table_file_name;
  pp_ctrl.readLookuptable(lookuptable);
 
  
  waypointSub = nh_traj.subscribe(waypoint_topic, 2, &Ctrl::callbackRefPath, this);
  

  odomSub = nh_state_.subscribe(odom_topic, 2, &Ctrl::odomCallback, this);  
  vesodomSub = nh_state_.subscribe("/vesc/odom", 2, &Ctrl::vescodomCallback, this);  
  poseSub  = nh_ctrl.subscribe("/tracked_pose", 2, &Ctrl::poseCallback, this);  
  imuSub = nh_ctrl.subscribe("/imu/data", 2, &Ctrl::imuCallback, this);  
  obstacleSub = nh_traj.subscribe("/datmo/box_kf", 2, &Ctrl::obstacleCallback, this);  
  lidarSub = nh_traj.subscribe("/scan", 2, &Ctrl::lidarCallback, this);  

  fren_pub = nh_ctrl.advertise<geometry_msgs::PoseStamped>("/fren_pose",1);
  global_traj_marker_pub = nh_traj.advertise<visualization_msgs::Marker>("global_traj", 1);
  centerlin_info_pub = nh_traj.advertise<visualization_msgs::Marker>("/centerline_info", 1);
  
  closest_obj_marker_pub = nh_traj.advertise<visualization_msgs::Marker>("/closest_obj", 1);

  // keypts_info_pub = nh_traj.advertise<visualization_msgs::Marker>("/keypts_info", 1);
  target_pointmarker_pub = nh_traj.advertise<visualization_msgs::Marker>("target_point", 1);
  speed_target_pointmarker_pub = nh_traj.advertise<visualization_msgs::Marker>("speed_target_point", 1);
  local_traj_marker_pub = nh_traj.advertise<visualization_msgs::MarkerArray>("local_traj", 1);
  pred_traj_marker_pub = nh_traj.advertise<visualization_msgs::MarkerArray>("predicted_traj", 1);
  ackmanPub = nh_ctrl.advertise<ackermann_msgs::AckermannDriveStamped>(control_topic, 2);    

  est_odom_pub = nh_ctrl.advertise<nav_msgs::Odometry>("/est_odom", 2);    


  boost::thread ControlLoopHandler(&Ctrl::ControlLoop,this);   

  f = boost::bind(&Ctrl::dyn_callback,this, _1, _2);
	srv.setCallback(f);
}

Ctrl::~Ctrl()
{}



void Ctrl::obstacleCallback(const hmcl_msgs::TrackArrayConstPtr& msg){
  if(!first_traj_received){
    return;
  }
  obstacles = *msg;
  
  std::vector<VehicleState> obstacles_vehicleState;
  
  for (int i=0; i<obstacles.tracks.size(); i++){
    VehicleState tmp_state; 
    tmp_state.pose = obstacles.tracks[i].odom.pose.pose;
    tmp_state.yaw = normalizeRadian(tf2::getYaw(obstacles.tracks[i].odom.pose.pose.orientation));
    tmp_state.vx = sqrt((obstacles.tracks[i].odom.twist.twist.linear.x*obstacles.tracks[i].odom.twist.twist.linear.x) 
                        + 
                        (obstacles.tracks[i].odom.twist.twist.linear.y*obstacles.tracks[i].odom.twist.twist.linear.y));
    computeFrenet(tmp_state, traj_manager.getglobalPath());      
      ///////////////////////////////////////////////////////////////////////////
      //check if obstacle is within track boundary
     if(local_traj.size() > 0) {
          double min_dist_to_local_path = 1e3;
          int min_idx = 0;
          for(int k=0; k < local_traj.x.size(); k++){
              double tmp_dist = sqrt((tmp_state.pose.position.x-local_traj.x[k])*(tmp_state.pose.position.x - local_traj.x[k]) + (tmp_state.pose.position.y-local_traj.y[k])*(tmp_state.pose.position.y-local_traj.y[k]));
              if(tmp_dist < min_dist_to_local_path){
                min_dist_to_local_path = tmp_dist;
                min_idx = k;
              }
          }
  
          double conserve_track_width  = -0.15 + std::min(local_traj.ey_l[min_idx], local_traj.ey_r[min_idx]);
          conserve_track_width  = std::max(0.0, conserve_track_width);
          if(fabs(tmp_state.ey) < conserve_track_width){              
            obstacles_vehicleState.push_back(tmp_state);
          }
     }

  
  }

  if(obstacles_vehicleState.size() < 1){
    ROS_INFO("Obstacles are all outside of track");
    pp_ctrl.is_there_obstacle = false;   
    return;
  }

  // check if the obstacles are within the track bound
  // get the most closest obstacle to the current ego vehicle 
  int best_idx = 0;
  double min_dist = 1000.0;
  double track_length = traj_manager.getTrackLength();
  bool any_front_obstacle = false;  
    
  for(int i=0; i < obstacles.tracks.size(); i++){    

              if(!traj_manager.is_s_front(cur_state.s, obstacles_vehicleState[i].s)){
                  any_front_obstacle = true;
                  double dist_tmp = traj_manager.get_s_diff(cur_state.s, obstacles_vehicleState[i].s);
                      if(dist_tmp< min_dist){
                              min_dist = dist_tmp;
                              best_idx = i;
                          }
              }else{
                  continue; // targets are behind of track
              }

    }
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  double dist_for_trigger = 100.0;
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  if(any_front_obstacle){
    std::cout << "any front obstacle   = true" << std::endl;
     if (min_dist < dist_for_trigger && min_dist > -0.5){
      pp_ctrl.is_there_obstacle = true;
      visualization_msgs::Marker closest_obstacle_marker;
      trackToMarker(obstacles.tracks[best_idx], closest_obstacle_marker);
    closest_obj_marker_pub.publish(closest_obstacle_marker);
    pp_ctrl.update_obstacleState(obstacles_vehicleState[best_idx]);
     return;
     }
  }
  pp_ctrl.is_there_obstacle = false;      
  return;
}

void Ctrl::trackToMarker(const hmcl_msgs::Track& track, visualization_msgs::Marker & marker){

    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();
    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose = track.odom.pose.pose;    
    marker.scale.x = track.length;
    marker.scale.y = track.width;
    marker.scale.z = 0.1;
    // Set the color of the marker
    marker.color.g = 1.0;
    marker.color.a = 1.0;
    


}



bool Ctrl::odom_close_to_pose(const geometry_msgs::PoseStamped & pos, const nav_msgs::Odometry& odom){
  double dist_tmp = (pos.pose.position.x - odom.pose.pose.position.x)*(pos.pose.position.x - odom.pose.pose.position.x)+(pos.pose.position.y - odom.pose.pose.position.y)*(pos.pose.position.y - odom.pose.pose.position.y);

  ros::Time pose_time = pos.header.stamp;
  ros::Time odom_time = odom.header.stamp;
  ros::Duration diff = odom_time - pose_time;
  double msg_diff_sec = diff.toSec();


  if(dist_tmp > odom_pose_diff_threshold || msg_diff_sec < -0.5){
    // incorrect odom data.. use pose instead 
    // std::cout << "ms_diff_sec = " << msg_diff_sec << std::endl;
    return false; 
  }else{
    return true;
  }

  
}

void Ctrl::poseCallback(const geometry_msgs::PoseStampedConstPtr& msg){
  std::lock_guard<std::mutex> lock(pose_mtx);


  if(!imu_received){
    return;
  }
  if(!first_pose_received){
    prev_pose = *msg;
    first_pose_received = true;
    return;
  }


  //If odom is available and close to current pose, then use odom instead-- check if the current position is close to current odom 
  if(first_odom_received && odom_close_to_pose(cur_pose,cur_odom)){    
      is_odom_used = true;      
  }else{
    is_odom_used = false;
  }

  ros::Time cur_time = msg->header.stamp;
  ros::Time prev_time = prev_pose.header.stamp;
  ros::Duration diff = cur_time - prev_time;
  double dt_sec = fabs(diff.toSec());

  if (dt_sec > 0.02){    
  cur_pose = *msg;
  cur_state.pose = cur_pose.pose;  
  cur_state.yaw = normalizeRadian(tf2::getYaw(cur_pose.pose.orientation));

    tf::Quaternion q( cur_pose.pose.orientation.x,cur_pose.pose.orientation.y,cur_pose.pose.orientation.z,cur_pose.pose.orientation.w);
    double yaw = tf::getYaw(q);    
    double vx_local_tmp, vy_local_tmp;
    vx_local_tmp = x_vel_filter.filter((cur_pose.pose.position.x - prev_pose.pose.position.x) / dt_sec);
    vy_local_tmp = y_vel_filter.filter((cur_pose.pose.position.y - prev_pose.pose.position.y) / dt_sec);
       
    // Compute Local velocity
    double vx_car = vx_local_tmp * cos(yaw) + vy_local_tmp * sin(yaw);
    double vy_car = -vx_local_tmp * sin(yaw) + vy_local_tmp * cos(yaw);
    vy_car = 0.0;
    //  Set the current odom message 
    nav_msgs::Odometry odom_msg;
    odom_msg.header = msg->header;
    odom_msg.pose.pose = msg->pose;    
    double local_vx, local_vy; 
    if(is_odom_used){
      // get the velocity from factor odom 
    odom_msg.twist.twist.linear.x = cur_state.vx;
    odom_msg.twist.twist.linear.y = cur_state.vy;
    }else{
      odom_msg.twist.twist.linear.x = cur_state.vx;
      odom_msg.twist.twist.linear.y = 0.0;
    }    
    odom_msg.twist.twist.linear.z = 0.0;
    odom_msg.twist.twist.angular.x = cur_imu.angular_velocity.x;
    odom_msg.twist.twist.angular.y = cur_imu.angular_velocity.y;
    odom_msg.twist.twist.angular.z = cur_imu.angular_velocity.z;    
    est_odom_pub.publish(odom_msg);    
    //
      cur_state.accel = 0.0;
      pp_ctrl.update_vehicleState(cur_state);
      traj_manager.log_odom(odom_msg); 
      
      
//// Publish frenet pose
      if(traj_manager.getRefTrajSize() > 5 && !traj_manager.is_recording()){         
            computeFrenet(cur_state, traj_manager.getglobalPath());
            geometry_msgs::PoseStamped fren_pose_;
            fren_pose_.header = msg->header;
            fren_pose_.pose.position.x = cur_state.s;
            fren_pose_.pose.position.y = cur_state.ey;
            fren_pose_.pose.position.z = cur_state.epsi;
            fren_pose_.pose.orientation.x = cur_state.vx;
            fren_pose_.pose.orientation.y = cur_state.vy;
            fren_pose_.pose.orientation.z = cur_state.wz;
            fren_pose_.pose.orientation.w = cur_state.delta;            
            fren_pub.publish(fren_pose_);
            traj_manager.frenet_ready = true;
          } 


      if(traj_manager.getRefTrajSize() > 5 && !traj_manager.is_recording() && traj_manager.frenet_ready){
        
          
          visualization_msgs::MarkerArray pred_marker = PathPrediction(cur_state,10);
          pred_traj_marker_pub.publish(pred_marker);
        }

    prev_pose = cur_pose;

    // check the laptime 

    prev_state = cur_state;

    // if (prev_state.s > cur_state.s):
    //   start_line_time.push_back(ros::Time::now().toSec());
    //   start_line_time[start_line_time.size()-1]
    //   ROS_INFO("Laptime = ")
  }
  
      
}


void Ctrl::imuCallback(const sensor_msgs::Imu::ConstPtr& msg){
  std::lock_guard<std::mutex> lock(imu_mtx);
  if(!imu_received){
    imu_received = true;
  }
  cur_imu = *msg;
}


void Ctrl::vescodomCallback(const nav_msgs::OdometryConstPtr& msg){
   std::lock_guard<std::mutex> lock(vesc_mtx);
  // if odom is used, then no update  --> priority 1. factor, 2. vesc/odom
  if(is_odom_used){
    return;    
  }
  
    cur_state.vx = msg->twist.twist.linear.x;
    cur_state.vy = 0.0;
return; 
}


void Ctrl::odomCallback(const nav_msgs::OdometryConstPtr& msg){
    std::lock_guard<std::mutex> lock(odom_mtx);
    if(!imu_received){
      return;
    }
    if(!first_odom_received){
      prev_odom = *msg;
      first_odom_received = true;
      return;
    }

  cur_odom = *msg;
  ros::Time cur_time = msg->header.stamp;
  ros::Time prev_time = prev_odom.header.stamp;
  ros::Duration diff = cur_time - prev_time;
  double dt_sec = fabs(diff.toSec());
  
  if (dt_sec > 0.02){    
    // if odom is far away from pose, then no update
    if(is_odom_used){        
          bool odom_twist_in_local = false;
                double yaw_ = cur_state.yaw;           
                if(odom_twist_in_local){
                  cur_state.vx = cur_odom.twist.twist.linear.x;
                  cur_state.vy = cur_odom.twist.twist.linear.y;          
                }else{            
                  double global_x  = cur_odom.twist.twist.linear.x;
                  double global_y  = cur_odom.twist.twist.linear.y;
                  cur_state.vx = fabs(global_x*cos(-1*yaw_) - global_y*sin(-1*yaw_)); 
                  cur_state.vy = global_x*sin(-1*yaw_) + global_y*cos(-1*yaw_);             
                }
                if(cur_state.vx < 0.1){
                  cur_state.vx = 0.1;
                }
                cur_state.wz = cur_odom.twist.twist.angular.z;    
      
     }
    prev_odom = cur_odom;
  }

  return;   
 
}

visualization_msgs::MarkerArray Ctrl::PathPrediction(const VehicleState state, int n_step){
  std::vector<VehicleState> propogated_states = ego_vehicle.dynamics_propogate(state,n_step);
  auto global_centerline = traj_manager.getglobalPath();
  frenToCarticians(propogated_states,global_centerline);

   visualization_msgs::MarkerArray marker_array;
  // Add arrows to the MarkerArray
  for (int i = 0; i < propogated_states.size(); i++) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();
    marker.id = i;
    marker.type = visualization_msgs::Marker::ARROW;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position = propogated_states[i].pose.position;    
    marker.pose.orientation = propogated_states[i].pose.orientation;
    marker.scale.x = 0.2;
    marker.scale.y = 0.05;
    marker.scale.z = 0.01;
    marker.color.a = 1.0;
    marker.color.r = 1.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    marker_array.markers.push_back(marker);
  }
    return marker_array;
}


void Ctrl::lidarCallback(const sensor_msgs::LaserScan::ConstPtr &msg)
{
  std::lock_guard<std::mutex> lock(lidar_mtx);
  cur_scan = msg;
  
};



void Ctrl::ControlLoop()
{ 
    double hz = 20;
    double ctrl_dt = 1/hz;
    ros::Rate loop_rate(20); // rate  
    
         

    while (ros::ok()){         
        
        auto start = std::chrono::steady_clock::now();        
        
        traj_manager.updatelookaheadPath(cur_state,lookahead_path_length);
      
        visualization_msgs::Marker global_traj_marker = traj_manager.getGlobalPathMarker();
        visualization_msgs::Marker centerline_info_marker = traj_manager.getCenterLineInfo();
        visualization_msgs::MarkerArray local_traj_marker = traj_manager.getLocalPathMarkerArray();
        global_traj_marker_pub.publish(global_traj_marker);
        local_traj_marker_pub.publish(local_traj_marker);
        if(centerline_info_marker.points.size() > 5){
            centerlin_info_pub.publish(centerline_info_marker);
        }
        
    

        pp_ctrl.set_manual_lookahead(manual_lookahed_switch, manual_speed_lookahed_switch, manual_lookahead, manual_speed_lookahead, max_a_lat);



          ackermann_msgs::AckermannDriveStamped pp_cmd;
          visualization_msgs::Marker targetPoint_marker, speed_targetPoint_marker;
          

          // if(is_odom_used){
          //   // factor estimation is enabled ... we can do model pp 
          //   ctrl_select = 2;
          // }
          // else{
          //   ctrl_select = 1;
          // }



        if(ctrl_select == 1){ 
                  // Purepursuit Computation 
        local_traj = traj_manager.getlookaheadPath();
        pp_ctrl.update_ref_traj(traj_manager.getlookaheadPath());
        bool is_straight; 
        // pp_cmd = pp_ctrl.compute_lidar_based_command(is_straight, cur_scan);
        pp_cmd = pp_ctrl.compute_command();
        targetPoint_marker = pp_ctrl.getTargetPointhMarker(1);
        target_pointmarker_pub.publish(targetPoint_marker);
        speed_targetPoint_marker = pp_ctrl.getTargetPointhMarker(-1);
        speed_target_pointmarker_pub.publish(speed_targetPoint_marker);
        
        }else if(ctrl_select ==2){
                // Model based Purepursuit Computation 
        local_traj = traj_manager.getlookaheadPath();
        pp_ctrl.update_ref_traj(traj_manager.getlookaheadPath());
         targetPoint_marker = pp_ctrl.getTargetPointhMarker(1); // 1 for steering lookahead 
        target_pointmarker_pub.publish(targetPoint_marker);
         speed_targetPoint_marker = pp_ctrl.getTargetPointhMarker(-1); // -1 for speed lookahead
        speed_target_pointmarker_pub.publish(speed_targetPoint_marker);
          pp_cmd = pp_ctrl.compute_model_based_command();
        }
        
        // pp_cmd.drive.speed = 0.0;
        // pp_cmd.drive.steering_angle = 0.0;

        if(!traj_manager.is_recording() && ctrl_select > 0){
          
         


           


          if (manual_velocity){
            // ROS_INFO("cmd vel = %f", pp_cmd.drive.speed);  
            pp_cmd.drive.speed = manual_target_vel;
            }         
            // multiply weight on vel
            pp_cmd.drive.speed = pp_cmd.drive.speed*manual_weight_ctrl;

        
          


          //    if(pp_cmd.drive.speed > 0.0 && pp_cmd.drive.speed < 1.0){
          //   pp_cmd.drive.speed = 1.5;
          // }         

               //////////////////// Check if obstacle within the line to the tareget point 
        bool is_overtaking = pp_ctrl.getOvertakingStatus();
        // if(is_overtaking){
        //   filter_given_TargetClearance(pp_cmd, targetPoint_marker);
        // } 

          ackmanPub.publish(pp_cmd);
          cur_state.delta = pp_cmd.drive.steering_angle;          
        }
        
        
        prev_state = cur_state;

      auto end = std::chrono::steady_clock::now();     
     loop_rate.sleep();
     std::chrono::duration<double> elapsed_seconds = end-start;
     if ( elapsed_seconds.count() > dt){
       ROS_ERROR("computing control gain takes too much time");
       std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
     }
      
    }
}

void Ctrl::filter_given_TargetClearance(ackermann_msgs::AckermannDriveStamped prev_cmd,visualization_msgs::Marker obst_marker){
  double target_x = obst_marker.pose.position.x;
  double target_y = obst_marker.pose.position.y;
  double del_x = target_x-cur_state.pose.position.x;
  double del_y = target_x-cur_state.pose.position.x;
  
  double local_x =  del_x*cos(-1*cur_state.yaw) - del_y*sin(-1*cur_state.yaw);              
  double local_y = del_y*sin(-1*cur_state.yaw) + del_y*cos(-1*cur_state.yaw);    

  double local_angle_to_target = std::atan2(local_y,local_x);
  double local_dist_to_target = sqrt(local_x*local_x + local_y+local_y);
  std::cout << "angle = " << local_angle_to_target*180/3.14195 << std::endl;
  std::cout << "dist = " << local_dist_to_target << std::endl;
  
  int scan_idx = int((normalizeRadian(local_angle_to_target)-cur_scan->angle_min)/cur_scan->angle_increment);
  if( scan_idx < 0 || scan_idx > cur_scan->ranges.size()){
    std::cout << "target point is out of current scan data range" <<  std::endl;
    return;
  }
  
  
}

void Ctrl::dyn_callback(highspeed_ctrl::testConfig &config, uint32_t level)
{
  ROS_INFO("Dynamiconfigure updated");  
  max_a_lat = config.max_a_lat;
  ctrl_select = config.ctrl_switch_param;
  manual_target_vel = config.manual_target_vel;
  manual_velocity = config.manual_velocity;
  manual_weight_ctrl = config.manual_weight_ctrl;
  manual_lookahead = config.manual_lookahead;
  manual_speed_lookahead = config.manual_speed_lookahead; 
  manual_lookahed_switch = config.manual_lookahed_switch;
  manual_speed_lookahed_switch = config.manual_speed_lookahed_switch;  
  return ;
  // ROS_INFO("Dynamiconfigure updated");
  // config_switch = config.config_switch;
  // if(config_switch){
  // Q_ey = config.Q_ey;
  // Q_eydot = config.Q_eydot
  // } 
}



void Ctrl::callbackRefPath(const visualization_msgs::MarkerArray::ConstPtr &msg)
{ 
  
 if(!first_traj_received){
  traj_manager.path_logger.updataPath(*msg);
  ROS_INFO("Trajectory has been updated by subscription");
  first_traj_received = true;
 }
 return;
  
};



int main (int argc, char** argv)
{


  

  ros::init(argc, argv, "HighSpeedCtrl");
  ros::NodeHandle nh_private("~");
  ros::NodeHandle nh_ctrl, nh_traj, nh_state;
  Ctrl Ctrl_(nh_ctrl, nh_traj, nh_state, nh_private);

  ros::CallbackQueue callback_queue_ctrl, callback_queue_traj, callback_queue_state;
  nh_ctrl.setCallbackQueue(&callback_queue_ctrl);
  nh_traj.setCallbackQueue(&callback_queue_traj);
  nh_state.setCallbackQueue(&callback_queue_state);
  

  std::thread spinner_thread_ctrl([&callback_queue_ctrl]() {
    ros::SingleThreadedSpinner spinner_ctrl;
    spinner_ctrl.spin(&callback_queue_ctrl);
  });


  std::thread spinner_thread_traj([&callback_queue_traj]() {
    ros::SingleThreadedSpinner spinner_traj;
    spinner_traj.spin(&callback_queue_traj);
  });

   std::thread spinner_thread_state([&callback_queue_state]() {
    ros::SingleThreadedSpinner spinner_state;
    spinner_state.spin(&callback_queue_state);
  });


    ros::spin();

    spinner_thread_ctrl.join();
    spinner_thread_traj.join();
    spinner_thread_state.join();


  return 0;

}
