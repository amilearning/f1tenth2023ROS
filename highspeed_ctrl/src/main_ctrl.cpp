
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

int test_count = 0;
using namespace std;

Ctrl::Ctrl(ros::NodeHandle& nh_ctrl, ros::NodeHandle& nh_traj, ros::NodeHandle& nh_p_):
  nh_p(nh_p_),
  nh_ctrl_(nh_ctrl),
  nh_traj_(nh_traj),
  traj_manager(nh_traj),
  pp_ctrl(nh_p_),
  my_steering_ok_(false),
  my_position_ok_(false),
  my_odom_ok_(false),
  ego_vehicle(0.1),
  ctrl_select(0),
  first_pose_received(false),
  imu_received(false),
  first_traj_received(false)
{
  
  
  nh_p.param<double>("x_vel_filter_cutoff", x_vel_filter_cutoff, 10.0);
  nh_p.param<double>("y_vel_filter_cutoff", y_vel_filter_cutoff, 10.0);

    double filter_dt = 0.01;
    
    x_vel_filter.initialize(filter_dt, x_vel_filter_cutoff);
    y_vel_filter.initialize(filter_dt, y_vel_filter_cutoff);


  nh_p.param<std::string>("status_topic", status_topic, "/vehicle_status");  
  nh_p.param<std::string>("control_topic", control_topic, "/vesc/ackermann_cmd");
  nh_p.param<std::string>("waypoint_topic", waypoint_topic, "/local_traj");
  // nh_p.param<std::string>("odom_topic", odom_topic, "/odom");   
  nh_p.param<std::string>("odom_topic", odom_topic, "/tracked_odom");   
  nh_p.param<int>("path_smoothing_times_", path_smoothing_times_, 1);
  nh_p.param<int>("curvature_smoothing_num_", curvature_smoothing_num_, 35);
  nh_p.param<int>("path_filter_moving_ave_num_", path_filter_moving_ave_num_, 35);  


  nh_p.param<double>("lookahead_path_length", lookahead_path_length, 5.0);
  nh_p.param<double>("curv_lookahead_path_length", curv_lookahead_path_length, 10.0);
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
  

  odomSub = nh_traj.subscribe(odom_topic, 2, &Ctrl::odomCallback, this);  
  poseSub  = nh_ctrl.subscribe("/tracked_pose", 2, &Ctrl::poseCallback, this);  
  imuSub = nh_ctrl.subscribe("/imu/data", 2, &Ctrl::imuCallback, this);  
  obstacleSub = nh_traj.subscribe("/datmo/obstacle", 2, &Ctrl::obstacleCallback, this);  

  fren_pub = nh_ctrl.advertise<geometry_msgs::PoseStamped>("/fren_pose",1);
  global_traj_marker_pub = nh_traj.advertise<visualization_msgs::Marker>("global_traj", 1);
  centerlin_info_pub = nh_traj.advertise<visualization_msgs::Marker>("/centerline_info", 1);
  keypts_info_pub = nh_traj.advertise<visualization_msgs::Marker>("/keypts_info", 1);
  target_pointmarker_pub = nh_traj.advertise<visualization_msgs::Marker>("target_point", 1);
  speed_target_pointmarker_pub = nh_traj.advertise<visualization_msgs::Marker>("speed_target_point", 1);
  local_traj_marker_pub = nh_traj.advertise<visualization_msgs::Marker>("local_traj", 1);
  pred_traj_marker_pub = nh_traj.advertise<visualization_msgs::MarkerArray>("predicted_traj", 1);
  ackmanPub = nh_ctrl.advertise<ackermann_msgs::AckermannDriveStamped>(control_topic, 2);    

  est_odom_pub = nh_ctrl.advertise<nav_msgs::Odometry>("/est_odom", 2);    


  boost::thread ControlLoopHandler(&Ctrl::ControlLoop,this);   

  f = boost::bind(&Ctrl::dyn_callback,this, _1, _2);
	srv.setCallback(f);
}

Ctrl::~Ctrl()
{}



void Ctrl::obstacleCallback(const hmcl_msgs::obstacleConstPtr& msg){
  cur_obstacles = *msg;
  obstacle_state.pose.position.x = cur_obstacles.x;
  obstacle_state.pose.position.y = cur_obstacles.y;
  obstacle_state.yaw = cur_obstacles.theta;

  tf2::Quaternion quat;
  quat.setRPY(0,0,obstacle_state.yaw);
  quat.normalize();
  obstacle_state.pose.orientation.x = quat.getX();
  obstacle_state.pose.orientation.y = quat.getY();
  obstacle_state.pose.orientation.z = quat.getZ();
  obstacle_state.pose.orientation.w = quat.getW();


  obstacle_state.vx = cur_obstacles.vx;
  obstacle_state.vy = cur_obstacles.vy;  
  computeFrenet(obstacle_state, traj_manager.getglobalPath());
  pp_ctrl.update_obstacleState(obstacle_state);

}


void Ctrl::odomToVehicleState(VehicleState & vehicle_state, const nav_msgs::Odometry & odom){
           vehicle_state.pose = odom.pose.pose;
           double yaw_ = normalizeRadian(tf2::getYaw(odom.pose.pose.orientation));
           vehicle_state.yaw = yaw_;
           bool odom_twist_in_local = true;
           if(odom_twist_in_local){
            vehicle_state.vx = odom.twist.twist.linear.x;
            vehicle_state.vy = odom.twist.twist.linear.y;          
           }else{            
            double global_x  = odom.twist.twist.linear.x;
            double global_y  = odom.twist.twist.linear.y;
            vehicle_state.vx = fabs(global_x*cos(-1*yaw_) - global_y*sin(-1*yaw_)); 
            vehicle_state.vy = global_x*sin(-1*yaw_) + global_y*cos(-1*yaw_);             
           }
          vehicle_state.wz = odom.twist.twist.angular.z;    
          if(traj_manager.getRefTrajSize() > 5 && !traj_manager.is_recording()){
         
            computeFrenet(vehicle_state, traj_manager.getglobalPath());
            geometry_msgs::PoseStamped fren_pose_;
            fren_pose_.header = odom.header;
            fren_pose_.pose.position.x = vehicle_state.s;
            fren_pose_.pose.position.y = vehicle_state.ey;
            fren_pose_.pose.position.z = vehicle_state.epsi;
            fren_pose_.pose.orientation.x = vehicle_state.vx;
            fren_pose_.pose.orientation.y = vehicle_state.vy;
            fren_pose_.pose.orientation.z = vehicle_state.wz;
            fren_pose_.pose.orientation.w = vehicle_state.delta;            
            fren_pub.publish(fren_pose_);
            traj_manager.frenet_ready = true;
          }      
          // std:cout << "yaw" << vehicle_state.yaw <<" epsi= " <<  vehicle_state.epsi << "s = " <<vehicle_state.s << std::endl;
          
}



void Ctrl::poseCallback(const geometry_msgs::PoseStampedConstPtr& msg){
  if(!imu_received){
    return;
  }
  if(!first_pose_received){
    prev_pose = *msg;
    first_pose_received = true;
    return;
  }
  cur_pose = *msg;
  ros::Time cur_time = msg->header.stamp;
  ros::Time prev_time = prev_pose.header.stamp;
  ros::Duration diff = cur_time - prev_time;
  double dt_sec = fabs(diff.toSec());

  if (dt_sec > 0.05){    
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
    odom_msg.twist.twist.linear.x = vx_car;
    odom_msg.twist.twist.linear.y = vy_car;
    odom_msg.twist.twist.linear.z = 0.0;
    odom_msg.twist.twist.angular.x = cur_imu.angular_velocity.x;
    odom_msg.twist.twist.angular.y = cur_imu.angular_velocity.y;
    odom_msg.twist.twist.angular.z = cur_imu.angular_velocity.z;
    
    est_odom_pub.publish(odom_msg);
      odomToVehicleState(cur_state,odom_msg);
      cur_state.accel = 0.0;
      pp_ctrl.update_vehicleState(cur_state);

      traj_manager.log_odom(odom_msg); 
      my_odom_ok_ = true;
      if(traj_manager.getRefTrajSize() > 5 && !traj_manager.is_recording() && traj_manager.frenet_ready){
        
          
          visualization_msgs::MarkerArray pred_marker = PathPrediction(cur_state,10);
          pred_traj_marker_pub.publish(pred_marker);
        }

    prev_pose = cur_pose;
  }
  
      
}

void Ctrl::imuCallback(const sensor_msgs::Imu::ConstPtr& msg){
  if(!imu_received){
    imu_received = true;
  }
  cur_imu = *msg;
}


void Ctrl::odomCallback(const nav_msgs::OdometryConstPtr& msg){
    odomToVehicleState(cur_state,*msg);
    
    cur_state.accel = 0.0;
    
    pp_ctrl.update_vehicleState(cur_state);

    traj_manager.log_odom(*msg); 
    my_odom_ok_ = true;
    if(traj_manager.getRefTrajSize() > 5 && !traj_manager.is_recording() && traj_manager.frenet_ready){
      
        
        visualization_msgs::MarkerArray pred_marker = PathPrediction(cur_state,10);
        pred_traj_marker_pub.publish(pred_marker);
      }
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


void Ctrl::ControlLoop()
{
    ros::Rate loop_rate(20); // rate  

    while (ros::ok()){         
        auto start = std::chrono::steady_clock::now();        
        
        traj_manager.updatelookaheadPath(cur_state,lookahead_path_length, curv_lookahead_path_length);
        KeyPoints curv_key_pts;        
        if(traj_manager.getCurvatureKeypoints(curv_key_pts)){
          visualization_msgs::Marker key_pts_marker = traj_manager.keyptsToMarker(curv_key_pts);
          keypts_info_pub.publish(key_pts_marker);
        }
        visualization_msgs::Marker global_traj_marker = traj_manager.getGlobalPathMarker();
        visualization_msgs::Marker centerline_info_marker = traj_manager.getCenterLineInfo();
        visualization_msgs::Marker local_traj_marker = traj_manager.getLocalPathMarker();
        global_traj_marker_pub.publish(global_traj_marker);
        local_traj_marker_pub.publish(local_traj_marker);
        if(centerline_info_marker.points.size() > 5){
            centerlin_info_pub.publish(centerline_info_marker);
        }
        
    

        pp_ctrl.set_manual_lookahead(manual_lookahed_switch, manual_speed_lookahed_switch, manual_lookahead, manual_speed_lookahead);



          ackermann_msgs::AckermannDriveStamped pp_cmd;
        if(ctrl_select == 1){
                  // Purepursuit Computation 
        pp_ctrl.update_ref_traj(traj_manager.getlookaheadPath());
        visualization_msgs::Marker targetPoint_marker = pp_ctrl.getTargetPointhMarker(1);
        target_pointmarker_pub.publish(targetPoint_marker);
        visualization_msgs::Marker speed_targetPoint_marker = pp_ctrl.getTargetPointhMarker(-1);
        speed_target_pointmarker_pub.publish(speed_targetPoint_marker);
        pp_cmd = pp_ctrl.compute_command();
        }else if(ctrl_select ==2){
                // Model based Purepursuit Computation 
        pp_ctrl.update_ref_traj(traj_manager.getlookaheadPath());
        visualization_msgs::Marker targetPoint_marker = pp_ctrl.getTargetPointhMarker(1);
        target_pointmarker_pub.publish(targetPoint_marker);
        visualization_msgs::Marker speed_targetPoint_marker = pp_ctrl.getTargetPointhMarker(-1);
        speed_target_pointmarker_pub.publish(speed_targetPoint_marker);
          pp_cmd = pp_ctrl.compute_model_based_command();
        }
        
        // pp_cmd.drive.speed = 0.0;
        // pp_cmd.drive.steering_angle = 0.0;

        if(!traj_manager.is_recording() && ctrl_select > 0){
          if (manual_velocity){
            ROS_INFO("cmd vel = %f", pp_cmd.drive.speed);
            pp_cmd.drive.speed = manual_target_vel;}         
            // put weight on vel
            pp_cmd.drive.speed = pp_cmd.drive.speed*manual_weight_ctrl;

          //    if(pp_cmd.drive.speed > 0.0 && pp_cmd.drive.speed < 1.0){
          //   pp_cmd.drive.speed = 1.5;
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



void Ctrl::dyn_callback(highspeed_ctrl::testConfig &config, uint32_t level)
{
  ROS_INFO("Dynamiconfigure updated");  
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
  ros::NodeHandle nh_ctrl, nh_traj;
  Ctrl Ctrl_(nh_ctrl, nh_traj, nh_private);

  ros::CallbackQueue callback_queue_ctrl, callback_queue_traj;
  nh_ctrl.setCallbackQueue(&callback_queue_ctrl);
  nh_traj.setCallbackQueue(&callback_queue_traj);
  

  std::thread spinner_thread_ctrl([&callback_queue_ctrl]() {
    ros::SingleThreadedSpinner spinner_ctrl;
    spinner_ctrl.spin(&callback_queue_ctrl);
  });

  std::thread spinner_thread_traj([&callback_queue_traj]() {
    ros::SingleThreadedSpinner spinner_traj;
    spinner_traj.spin(&callback_queue_traj);
  });

 

    ros::spin();

    spinner_thread_ctrl.join();
    spinner_thread_traj.join();


  return 0;

}
