#include "trajectory_manager.h"

TrajectoryManager::TrajectoryManager(const ros::NodeHandle& nh_traj) : traj_nh(nh_traj), path_logger(0.1){
    // create publishers and subscribers
    is_recording_ = false;
    path_record_init_srv = traj_nh.advertiseService("/path_record_init_srv", &TrajectoryManager::startRecordingCallback, this);
    path_record_stop_srv = traj_nh.advertiseService("/path_record_stop_srv", &TrajectoryManager::stopRecordingCallback, this);
    path_save_srv = traj_nh.advertiseService("/path_save_srv", &TrajectoryManager::savePathCallback, this);
    path_read_srv = traj_nh.advertiseService("/path_read_srv", &TrajectoryManager::readPathCallback, this);
    
    
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


Trajectory TrajectoryManager::getlookaheadPath(){
    return lookahead_traj;
}

// Delete the driven traj
void TrajectoryManager::updatelookaheadPath(const VehicleState& vehicle_state, const double& length)
{
    path_logger.getPath(tmp_ref_traj);
    if(tmp_ref_traj.size() < 2 ){return;}
    // Find the closest point on the path to the current position
    double closest_dist = std::numeric_limits<double>::infinity();
    int closest_idx = -1;
    for (int i = 0; i < tmp_ref_traj.size(); ++i) {
        double dist = std::sqrt(std::pow(tmp_ref_traj.x[i] - vehicle_state.pose.position.x, 2.0) + std::pow(tmp_ref_traj.y[i] - vehicle_state.pose.position.y, 2.0));
        if (dist < closest_dist) {
            closest_dist = dist;
            closest_idx = i;
        }
    }
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
    while (total_dist < length) {
        
        if (end_idx < tmp_ref_traj.size() - 1) {
            double dist = std::sqrt(std::pow(tmp_ref_traj.x[end_idx] - tmp_ref_traj.x[end_idx+1], 2.0) +std::pow(tmp_ref_traj.y[end_idx] - tmp_ref_traj.y[end_idx+1], 2.0));
            if (total_dist + dist < length) {
                total_dist += dist;
                ++end_idx;
            } else {
                break;
            }
        }else{
            break;
        }
    }
    // ROS_INFO("start_idx = %d", start_idx);
    // ROS_INFO("end_idx = %d", end_idx);
    lookahead_traj = tmp_ref_traj.get_segment(start_idx,end_idx);

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

visualization_msgs::Marker TrajectoryManager::traj_to_marker(const Trajectory & traj, const std_msgs::ColorRGBA & color_){
  // Create marker message
  visualization_msgs::Marker marker;
  marker.header.frame_id = "map";
  marker.header.stamp = ros::Time::now();
  marker.ns = "trajectory";
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
    point.z = traj.z[i];

    // Add Point to Marker message
    marker.points.push_back(point);
  }

 return marker;
}
