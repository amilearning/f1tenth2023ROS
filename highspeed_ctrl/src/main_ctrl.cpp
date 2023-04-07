
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

Ctrl::Ctrl(ros::NodeHandle& nh_ctrl, ros::NodeHandle& nh_traj):  
  nh_ctrl_(nh_ctrl),
  nh_traj_(nh_traj),
  traj_manager(nh_traj),
  my_steering_ok_(false),
  my_position_ok_(false),
  my_odom_ok_(false)
{
  
  nh_traj.param<std::string>("status_topic", status_topic, "/vehicle_status");  
  nh_traj.param<std::string>("control_topic", control_topic, "/drive");
  nh_traj.param<std::string>("waypoint_topic", waypoint_topic, "/local_traj");
  nh_traj.param<std::string>("odom_topic", odom_topic, "/odom");
  
  
  nh_traj.param<int>("path_smoothing_times_", path_smoothing_times_, 1);
  nh_traj.param<int>("curvature_smoothing_num_", curvature_smoothing_num_, 35);
  nh_traj.param<int>("path_filter_moving_ave_num_", path_filter_moving_ave_num_, 35);
  
  nh_traj.param<double>("lookahead_path_length", lookahead_path_length, 5.0);
  nh_traj.param<double>("wheelbase", wheelbase, 0.33);
  nh_traj.param<double>("lf", lf, 0.165);
  nh_traj.param<double>("lr", lr, 0.165);
  nh_traj.param<double>("mass", mass, 2.0);    
  nh_traj.param<double>("dt", dt, 0.04); 
  
  waypointSub = nh_traj.subscribe(waypoint_topic, 2, &Ctrl::callbackRefPath, this);
  
  odomSub = nh_traj.subscribe(odom_topic, 2, &Ctrl::odomCallback, this);  
  global_traj_marker_pub = nh_traj.advertise<visualization_msgs::Marker>("global_traj", 1);
  local_traj_marker_pub = nh_traj.advertise<visualization_msgs::Marker>("local_traj", 1);
  ackmanPub = nh_ctrl.advertise<ackermann_msgs::AckermannDriveStamped>("/drive", 2);    


  boost::thread ControlLoopHandler(&Ctrl::ControlLoop,this);   

  f = boost::bind(&Ctrl::dyn_callback,this, _1, _2);
	srv.setCallback(f);
}

Ctrl::~Ctrl()
{}

void Ctrl::odomToVehicleState(VehicleState & vehicle_state, const nav_msgs::Odometry & odom){
           vehicle_state.pose = odom.pose.pose;
           double yaw_ = normalizeRadian(tf2::getYaw(odom.pose.pose.orientation));
           vehicle_state.yaw = yaw_;
           bool odom_twist_in_local = false;
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
           
}

void Ctrl::odomCallback(const nav_msgs::OdometryConstPtr& msg){
    odomToVehicleState(cur_state,*msg);
    traj_manager.log_odom(*msg); 
    my_odom_ok_ = true;
}




void Ctrl::ControlLoop()
{
    ros::Rate loop_rate(20); // rate  

    while (ros::ok()){         
        auto start = std::chrono::steady_clock::now();        
        
        traj_manager.updatelookaheadPath(cur_state,lookahead_path_length);
        visualization_msgs::Marker global_traj_marker = traj_manager.getGlobalPathMarker();
        visualization_msgs::Marker local_traj_marker = traj_manager.getLocalPathMarker();
        global_traj_marker_pub.publish(global_traj_marker);
        local_traj_marker_pub.publish(local_traj_marker);

        prev_state = cur_state;
        ///////////////////////////////////////////////////////

        // Prepare current State for state feedback control 
      //   if(!stateSetup()){
      //     ROS_WARN("Path is not close to the current position");
      //     debug_msg.header.stamp = ros::Time::now();    
      //     debugPub.publish(debug_msg);
      //      loop_rate.sleep();
      //     continue;
      //   }
      //   VehicleModel_.setState(Xk,Cr);
      //   double current_speed = max(vehicle_status_.twist.linear.x ,1.5); // less than 1m/s takes too much time for riccati solver
      //   // double current_speed = max(wheel_speed,1.0); // less than 1m/s takes too much time for riccati solver
      //   //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
         
      //   // Computes control gains 
      //   ///////////////////////////////////////////////////////
       
           
      //   VehicleModel_.computeMatrices(current_speed);  

        
      //   bool riccati_solved = VehicleModel_.solveRiccati();
      //   if(!riccati_solved){
      //     ROS_WARN("solution not found ~~~!!!!!!! control lost"); 
      //      loop_rate.sleep();
      //     continue;
      //   }
      //   double delta_cmd = VehicleModel_.computeGain(); 
        
        
      //   // if(fabs(delta_cmd) > 0.5){
      //   //    loop_rate.sleep();
      //   //   continue;
      //   // }                  


      //   // concatenate via angular limit
      //   double diff_delta = delta_cmd-prev_delta_cmd;
        
        
        
      //   if( fabs(diff_delta)/dt > angle_rate_limit ){
      //     ROS_WARN("rate limit reached!!! angle_rate_limit = %f",angle_rate_limit);
      //     if(diff_delta>0){
      //         delta_cmd = prev_delta_cmd + angle_rate_limit*dt;
      //     }else{
      //         delta_cmd = prev_delta_cmd - angle_rate_limit*dt;
      //     }
      //   }
        
      //   // extract target velocity  (avg of local segments)
      //   double avg_speed_sum = std::accumulate(std::begin(traj_.vx), std::end(traj_.vx), 0.0);
      //   double avg_speed =  avg_speed_sum / traj_.vx.size();
      //   // curvature and maximum lateral accerlation filtering 
      //   double alat_lim = 0.2;
      //   double kappa = 0;
      //   double cur_lim_vel = sqrt(alat_lim/kappa);
      //   double target_speed = std::min(avg_speed,cur_lim_vel);
      // //////////////////////////////////////////////////
      // ////////////////////////////////////////////////// Lateral error filter 
      //   // filter by error , 0.3 m is margin 
      //   // 1m error -> reduce to 20%
      //   // 0.5m error -> reduce to 60%
      //   // Y = -0.8X + 1
      // //////////////////////////////////////////////////
      //   // double l_lim= 0.2;        
      //   // double filter_ratio = -0.8*std::max(l_lim, Xk(0))+1;
      //   // filter_ratio = std::max(l_lim,std::min(filter_ratio,1.0));
      //   // target_speed = target_speed*filter_ratio;
      // //////////////////////////////////////////////////
      //   // ROS_INFO("target_speed = %f", target_speed);
      //   std_msgs::Float64 vel_msg;
      //   vel_msg.data = target_speed;
      //   // velPub.publish(vel_msg);
      //   debug_msg.pose.position.x = delta_cmd;    
      //   delta_cmd = steer_filter.filter(delta_cmd);            
      //   debug_msg.pose.position.y = delta_cmd;    
        
      //   //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
      //   ackermann_msgs::AckermannDrive ctrl_msg;
      //   ctrl_msg.acceleration = 1.0;        
       
        
      //   ctrl_msg.steering_angle = delta_cmd;
        
      //   ackmanPub.publish(ctrl_msg);
        
   
          

      //   ///////////
        
        
        
      //   hmcl_msgs::VehicleSteering steer_msg;
        
      //   steer_msg.header.stamp = ros::Time::now();
      //   steer_msg.steering_angle = delta_cmd;        

      //   steerPub.publish(steer_msg);
      //   ///////////////////////////////////////////////////////
      //   // record control inputs 
      //   delta_buffer.push_back(delta_cmd);
      //   if(delta_buffer.size()>delay_step){
      //     delta_buffer.erase(delta_buffer.begin());
      //   }
      //   //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

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
  return ;
  // ROS_INFO("Dynamiconfigure updated");
  // config_switch = config.config_switch;
  // if(config_switch){
  // Q_ey = config.Q_ey;
  // Q_eydot = config.Q_eydot;
  // Q_epsi = config.Q_epsi ;
  // Q_epsidot = config.Q_epsidot;
  // R_weight = config.R_weight;  
  // angle_rate_limit = config.angle_rate_limit;
  // lag_tau = config.lag_tau;
  // delay_in_sec = config.delay_in_sec;
  // delay_step = (int)(delay_in_sec/dt);
  // std::vector<double> Qweight = {Q_ey, Q_eydot, Q_epsi, Q_epsidot};
      // double tmp_q = -0.1*speed*3.6 + 7;
      // tmp_q = std::min(std::max(tmp_q,3.0),7.0);
      // double tmp_r = 80*speed*3.6 + 500;
      // tmp_r = std::min(std::max(tmp_r,1000.0),4000.0);

      // Qweight[0] = tmp_q;
      // R_weight = tmp_r;
      // VehicleModel_.setWeight( Qweight, R_weight);


  
  // std::vector<double> Qweight_ = {Q_ey, Q_eydot, Q_epsi, Q_epsidot};
  // VehicleModel_.setWeight( Qweight_, R_weight);
  // VehicleModel_.setDelayStep(delay_step);  
  // VehicleModel_.setLagTau(lag_tau);
  // }

  
}



void Ctrl::callbackRefPath(const hmcl_msgs::Lane::ConstPtr &msg)
{
  // current_waypoints_ = *msg;  
 
  // Trajectory traj;
  // /* calculate relative time */
  // std::vector<double> relative_time;
  // calcPathRelativeTime(current_waypoints_, relative_time);  

  // /* resampling */
  // double traj_resample_dist_ = dt;
  // convertWaypointsToMPCTrajWithDistanceResample(current_waypoints_, relative_time, traj_resample_dist_, traj);
  // convertEulerAngleToMonotonic(traj.yaw);
  // /* path smoothing */
  // bool enable_path_smoothing_ =true;
  // if (enable_path_smoothing_)
  // {
  //   for (int i = 0; i < path_smoothing_times_; ++i)
  //   {
  //     if (!MoveAverageFilter::filt_vector(path_filter_moving_ave_num_, traj.x) ||
  //         !MoveAverageFilter::filt_vector(path_filter_moving_ave_num_, traj.y) ||
  //         !MoveAverageFilter::filt_vector(path_filter_moving_ave_num_, traj.yaw) ||
  //         !MoveAverageFilter::filt_vector(path_filter_moving_ave_num_, traj.vx))
  //     {
  //       ROS_WARN("path callback: filtering error. stop filtering");
  //       return;
  //     }
  //   }
  // }

  // /* calculate yaw angle */
  // bool enable_yaw_recalculation_=true;
  // if (enable_yaw_recalculation_)
  // {
  //   calcTrajectoryYawFromXY(traj);
  //   convertEulerAngleToMonotonic(traj.yaw);
  // }

  // /* calculate curvature */
  // calcTrajectoryCurvature(traj, curvature_smoothing_num_);
  // const double max_k = *max_element(traj.k.begin(), traj.k.end());
  // const double min_k = *min_element(traj.k.begin(), traj.k.end());  

  // traj_ = traj;

  // /* publish trajectory for visualize */
  // visualization_msgs::Marker markers;
  // convertTrajToMarker(traj, markers, "ref_traj", 0.0, 0.5, 1.0, 0.05);
  // pub_debug_filtered_traj_.publish(markers);
  
};



int main (int argc, char** argv)
{
  ros::init(argc, argv, "HighSpeedCtrl");
  
  ros::NodeHandle nh_ctrl, nh_traj;
  Ctrl Ctrl_(nh_ctrl, nh_traj);

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
