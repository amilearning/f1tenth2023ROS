#include "trajectory_manager.h"

TrajectoryManager::TrajectoryManager(const ros::NodeHandle& nh_traj) : traj_nh(nh_traj), path_logger(0.05), frenet_ready(false), is_first_lookahead(true){
    // create publishers and subscribers
    is_recording_ = false;
    path_record_init_srv = traj_nh.advertiseService("/path_record_init_srv", &TrajectoryManager::startRecordingCallback, this);
    path_record_stop_srv = traj_nh.advertiseService("/path_record_stop_srv", &TrajectoryManager::stopRecordingCallback, this);
    path_save_srv = traj_nh.advertiseService("/path_save_srv", &TrajectoryManager::savePathCallback, this);
    path_read_srv = traj_nh.advertiseService("/path_read_srv", &TrajectoryManager::readPathCallback, this);
    
    
}

visualization_msgs::Marker TrajectoryManager::getCenterLineInfo(){
    visualization_msgs::Marker marker;
      if (path_logger.get_size() > 0){
        Trajectory local_tmp_traj; 
        path_logger.getPath(local_tmp_traj);
        std_msgs::ColorRGBA color_;
        color_.a = 0.2;
        color_.g = 1.0;
         // Create marker message
            
            marker.header.frame_id = "map";
            marker.header.stamp = ros::Time::now();
            //   marker.ns = "trajectory";
            marker.type = visualization_msgs::Marker::LINE_STRIP;
            marker.action = visualization_msgs::Marker::ADD;
            marker.pose.orientation.w = 1.0;
            marker.scale.x = 0.1;
            marker.color = color_;

            // Get Trajectory data

            // fill traj with data

            // Loop through Trajectory x and y vectors
            for (size_t i = 0; i < local_tmp_traj.x.size(); ++i)
            {
                // Create Point message
                geometry_msgs::Point point;
                point.x = local_tmp_traj.s[i];
                point.y = local_tmp_traj.k[i];
                double track_width = 0.5;
                // TODO: having a varying track width 
                point.z = track_width;
                // Add Point to Marker message
                marker.points.push_back(point);
            }
    }
    return marker;
}

visualization_msgs::Marker TrajectoryManager::getGlobalPathMarker(){
    visualization_msgs::Marker tmp_marker;
    if (path_logger.get_size() > 0){
        Trajectory local_tmp_traj; 
        path_logger.getPath(local_tmp_traj);
        std_msgs::ColorRGBA color_;
        color_.a = 0.2;
        color_.g = 1.0;
        tmp_marker = traj_to_marker(local_tmp_traj,color_);        
    }
    return tmp_marker;    
}


visualization_msgs::Marker TrajectoryManager::getLocalPathMarker(){
    visualization_msgs::Marker tmp_marker;
    if (lookahead_traj.size() > 0){
        std_msgs::ColorRGBA color_;
        color_.a = 0.5;
        color_.r = 1.0;
        tmp_marker = traj_to_marker(lookahead_traj,color_);        
    }
    return tmp_marker;    
}

visualization_msgs::Marker TrajectoryManager::keyptsToMarker(const KeyPoints & key_pts){
visualization_msgs::Marker marker;
marker.header.frame_id = "map";
marker.header.stamp = ros::Time::now();
//   marker.ns = "trajectory";
  marker.type = visualization_msgs::Marker::LINE_STRIP;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.1;
std_msgs::ColorRGBA color_;
color_.a = 0.5;
color_.r = 0.8; 
  marker.color = color_;
  for (size_t i = 0; i < key_pts.s_curv.rows(); ++i)
  {    
    geometry_msgs::Point point;
    point.x = key_pts.s_curv(i,0);
    point.y = key_pts.s_curv(i,1);    
    marker.points.push_back(point);
  }
  return marker;
}



bool TrajectoryManager::is_s_front(const double &s, const double &s_target){
    double diff = s-s_target;
    double track_length = getTrackLength(); 
    
    if(abs(diff) > track_length/2){
        double unwraped_s_target = s_target+track_length;
        diff = unwraped_s_target - s;
                if(diff > 0){
                return false;
            }else{
                return true;
            }
    }else{
                if(diff > 0){
                return true;
            }else{
                return false;
            }    

     }
    
}


double TrajectoryManager::get_s_diff(const double &ego_s1, const double &tar_s2){
        // diff-s = tar_s - ego_s 

        // tar -- > ego 
         //  0.1    5.9   =  -5.8
         //  0.1    0.3  = -0.2
         // 5.9     0.1 = 5.8 
         // 5.7    5.9 = -0.2 
        double diff_s = tar_s2-ego_s1 ;
        if (fabs(diff_s) > getTrackLength()/2){
            if(diff_s < 0){
                diff_s = diff_s + getTrackLength();
            }else{
                diff_s = diff_s - getTrackLength();
            }
        }


        // if(ego_s1 > getTrackLength()/2 && tar_s2 < getTrackLength()/2 )
        // {   
            
        // }
        // double bigger_s = std::max(s1,s2);
        // double smaller_s = std::min(s1,s2);
        // double diff_s = bigger_s - smaller_s;
        // double track_length = getTrackLength(); 
        // if (abs(diff_s) > track_length/2.0){
        //     smaller_s  = smaller_s +  track_length;            
        //     diff_s = smaller_s - bigger_s; 
        //     wrapTrack(diff_s);
        // }
        return diff_s;
}


void TrajectoryManager::wrapTrack(double & s){
    double track_length = getTrackLength();
    while (s > track_length){
        s = s- track_length; 
    }
    while ( s < 0 ){
        s = s+track_length;
    }
}

double TrajectoryManager::getTrackLength(){
    path_logger.getPath(tmp_ref_traj);
    return tmp_ref_traj.s[tmp_ref_traj.s.size()-1];
}


Trajectory TrajectoryManager::getlookaheadPath(){
    return lookahead_traj;
}

Trajectory TrajectoryManager::getglobalPath(){
    path_logger.getPath(tmp_ref_traj);
    return tmp_ref_traj;
}



unsigned int TrajectoryManager::getClosestIdx(double& closest_dist, const Trajectory & traj_, const VehicleState& vehicle_state){
         closest_dist = std::numeric_limits<double>::infinity();
        unsigned int closest_idx = 0;
        for (int i = 0; i < traj_.size(); ++i) {
        double dist = std::sqrt(std::pow(traj_.x[i] - vehicle_state.pose.position.x, 2.0) + std::pow(traj_.y[i] - vehicle_state.pose.position.y, 2.0));

            if (dist < closest_dist) {
            closest_dist = dist;
            closest_idx = i;
            }
        }
        return closest_idx;
}  


// Delete the driven traj
void TrajectoryManager::updatelookaheadPath(const VehicleState& vehicle_state, const double& length)
{
    
        path_logger.getPath(global_ref_traj);    
    
        
    
    if(global_ref_traj.size() < 2 ){return;}
     
    
    

    // Find the closest point on the path to the current position
    double closest_dist = std::numeric_limits<double>::infinity();
   if(is_first_lookahead){
        
        local_ref_traj_idx_in_global = getClosestIdx(closest_dist, global_ref_traj,vehicle_state);        
        is_first_lookahead = false;
        
        
    }else{
        auto closest_idx_in_local = getClosestIdx(closest_dist,lookahead_traj,vehicle_state);
        if(closest_dist > 0.4 || local_ref_traj_idx_in_global > global_ref_traj.size()){
            // vehicle status is in outside of  local path
            local_ref_traj_idx_in_global = getClosestIdx(closest_dist, global_ref_traj,vehicle_state);
            
        }else{
            local_ref_traj_idx_in_global = closest_idx_in_local+local_ref_traj_idx_in_global;
            
        }
    }
   
    unsigned int closest_idx = local_ref_traj_idx_in_global;
    // Delete the past trajectory -- TODO: need to update at completion of 1 lap
    // if(!is_recording_){
    //     ROS_INFO("trim up to %d", closest_idx);
    //     path_logger.trimrajectory(closest_idx);
    // }
    

    // if (closest_idx < ref_traj.size()-4){
    //     closest_idx = closest_idx +3;
    // }

    // Determine the start and end indices of the segment
    double total_dist = 0.0;
    int start_idx = closest_idx;
    int end_idx = closest_idx;
    int max_count = 0;
    while (total_dist < length) {
        ++max_count;
        if (end_idx < global_ref_traj.size() - 1) {
            double dist = std::sqrt(std::pow(global_ref_traj.x[end_idx] - global_ref_traj.x[end_idx+1], 2.0) +std::pow(global_ref_traj.y[end_idx] - global_ref_traj.y[end_idx+1], 2.0));
            if (total_dist + dist < length) {
                total_dist += dist;
                ++end_idx;
            } else {
                
                break;
            }
        }else{
            // loop again to begin of track
            end_idx = 0;
            // break;
        }
        if(max_count > 1e6){
            break;
        }
    }
    // ROS_INFO("start_idx = %d", start_idx);
    // ROS_INFO("end_idx = %d", end_idx);
   
    lookahead_traj = global_ref_traj.get_segment(start_idx,end_idx);
    

 

    
}


int TrajectoryManager::getRefTrajSize(){        
        return path_logger.get_size();    
}


void TrajectoryManager::log_odom(const nav_msgs::Odometry& odom){
    
        path_logger.logPath(odom,is_recording_);
    
}

bool TrajectoryManager::startRecordingCallback(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res){

    is_recording_ = true;
    ROS_INFO("Path Recording Init");
res.success = true;
    return true;  
    
}

bool TrajectoryManager::stopRecordingCallback(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res){
    is_recording_ = false;
      ROS_INFO("Path Recording Stop");
    res.success = true;
    return true;  
}

bool TrajectoryManager::readPathCallback(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res){
    ROS_INFO("Path read from file");
    std::string package_path = ros::package::getPath("highspeed_ctrl");
    std::string path_file = package_path + "/path/path.txt";
    ROS_INFO("Trying Path read from file: %s", path_file.c_str());
    path_logger.readPathFromFile(path_file);
    
    ROS_INFO("Path read from file: %s", path_file.c_str());
    res.success = true;
    return true;    
}

// save the path as center line in global coordinate
bool TrajectoryManager::savePathCallback(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res){
   
    std::string package_path = ros::package::getPath("highspeed_ctrl");
    std::string path_file = package_path + "/path/path.txt";
    path_logger.savePathToFile(path_file);
    
    ROS_INFO("Path saved to file: %s", path_file.c_str());
        res.success = true;
    return true;  
}

bool TrajectoryManager::is_recording()
{
    return is_recording_;
}
visualization_msgs::Marker TrajectoryManager::traj_to_marker(const Trajectory & traj, const std_msgs::ColorRGBA & color_){
  // Create marker message
  visualization_msgs::Marker marker;
  marker.header.frame_id = "map";
  marker.header.stamp = ros::Time::now();
//   marker.ns = "trajectory";
  marker.type = visualization_msgs::Marker::LINE_STRIP;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.1;
  marker.color = color_;

  // Get Trajectory data

  // fill traj with data

  // Loop through Trajectory x and y vectors
  for (size_t i = 0; i < traj.x.size(); ++i)
  {
    // Create Point message
    geometry_msgs::Point point;
    point.x = traj.x[i];
    point.y = traj.y[i];
    point.z = traj.k[i];

    // Add Point to Marker message
    marker.points.push_back(point);
  }

 return marker;
}
