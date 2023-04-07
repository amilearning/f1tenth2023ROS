#ifndef TRAJECTORY_MANAGER_HPP_
#define TRAJECTORY_MANAGER_HPP_

#include <ros/package.h>
#include <ros/ros.h>
#include <ros/time.h>

#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include <nav_msgs/Odometry.h>


#include <fstream>
#include <mutex>
#include <vector> 
#include <cmath>

#include <tuple>
#include <algorithm>

#include <boost/filesystem.hpp>

#include "state.h"

#include <tf/tf.h>


// #include "std_msgs/String.hpp"
#include "std_msgs/String.h"
#include "std_srvs/Trigger.h"
#include "geometry_msgs/Twist.h"
#include "geometry_msgs/Point.h"
#include "sensor_msgs/LaserScan.h"
#include "sensor_msgs/Imu.h"
#include <thread>
#include <sstream>
#include <string>
#include "geometry_msgs/PoseStamped.h"
#include "std_msgs/ColorRGBA.h"
#include "lowpass_filter.h"
#include "trajectory.h"

class PathLogger {
public:
    PathLogger(double threshold) : threshold_(threshold) {

        double dt = 0.1;
        double lpf_cutoff_hz = 0.1;
        x_filter.initialize(dt, lpf_cutoff_hz);
        y_filter.initialize(dt, lpf_cutoff_hz);
        yaw_filter.initialize(dt, lpf_cutoff_hz);
        ref_traj.clear();
    }


void logPath(const nav_msgs::Odometry& odom, bool is_record) {
    double dx = odom.pose.pose.position.x - last_odom.pose.pose.position.x;
    double dy = odom.pose.pose.position.y - last_odom.pose.pose.position.y;
    double distance = std::sqrt(dx*dx + dy*dy);
    
    tf::Quaternion q(odom.pose.pose.orientation.x,
                        odom.pose.pose.orientation.y,
                        odom.pose.pose.orientation.z,
                        odom.pose.pose.orientation.w);
    q.normalize();
    // Extract the yaw angle from the quaternion object    
    double roll, pitch, yaw;
    tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
        x_tmp = x_filter.filter(odom.pose.pose.position.x);
        y_tmp = y_filter.filter(odom.pose.pose.position.y); 
        yaw_tmp = yaw_filter.filter(yaw);

        if (distance >= threshold_ && is_record) {
            ref_traj.push_back(x_tmp,y_tmp, 0.0, y_tmp, 0.0, 0.0, 0.0, 0.0);
            
            // path_.push_back(std::make_tuple(odom.pose.pose.position.x, odom.pose.pose.position.y, yaw));
            last_odom = odom;
        }
        
    }

    int get_size(){
        if(ref_traj.x.size() < 1){
            return 0;
        }else{
            return ref_traj.x.size();
        }        
    }

    void getPath(Trajectory & tmp_traj) {
        tmp_traj = ref_traj;
        
    }
    
    void trimrajectory(int closest_idx){
        if(closest_idx < 1 || closest_idx > ref_traj.size()-1){
            return;
        }        
        ref_traj.erase_to(closest_idx);
       
    }

    //////////////////////////////////////////////////////////////////////
    
    void savePathToFile(const std::string& file_path) {
    // Open the file for writing
    unsigned int trajectory_size = ref_traj.size();
    
    if (trajectory_size < 2){
        return ;
    }

    std::size_t pos = file_path.find_last_of("/\\");
    std::string directory_path = file_path.substr(0, pos);
    if (!boost::filesystem::exists(directory_path)) {
        boost::filesystem::create_directory(directory_path);
    }


    std::ofstream ofs(file_path, std::ios::out);

    // Write the path to the file
    for (unsigned int i = 0; i < trajectory_size; ++i) {
        
        ofs << ref_traj.x[i] << " " << ref_traj.y[i] << " " << ref_traj.yaw[i] << "\n";
    }

    // Close the file
    ofs.close();
    }

    void readPathFromFile(const std::string& file_path) {
    // Open the file for reading
    std::ifstream ifs(file_path, std::ios::in);
    if (ifs.fail()) {        
        return;
    }
    // Clear the current path
    ref_traj.clear();
    // Read the path from the file
    double x, y,  yaw;
    while (ifs >> x >> y >>yaw ) {
        ref_traj.push_back(x,y,0.0, yaw, 0.0, 0.0, 0.0, 0.0);
        
        
    }
    std::cout << "ref_traj size = " << ref_traj.size() << std::endl;
    // Close the file
    ifs.close();

    }



    double threshold_;
    Trajectory ref_traj;    
    nav_msgs::Odometry last_odom;
    Butterworth2dFilter x_filter,y_filter, yaw_filter;    
    double x_tmp, y_tmp, yaw_tmp; 

};


class TrajectoryManager {
public:

    TrajectoryManager(const ros::NodeHandle& nh_traj);

    // save the path as center line in global coordinate
    

    bool savePathCallback(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res);
    bool startRecordingCallback(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res);
    bool stopRecordingCallback(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res);
    bool readPathCallback(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res);
    
    void log_odom(const nav_msgs::Odometry& odom);
    visualization_msgs::Marker traj_to_marker(const Trajectory & traj, const std_msgs::ColorRGBA & color_);
    
    void updatelookaheadPath(const VehicleState& vehicle_state, const double& length);
    Trajectory getlookaheadPath(); 
    visualization_msgs::Marker getLocalPathMarker();
    
    visualization_msgs::Marker getGlobalPathMarker();
    int getRefTrajSize(); 
    // // extract a lookahead path in frenet coordinate given the current position (odometry)
    // void extractLookaheadPath(const nav_msgs::msg::Odometry& odom, nav_msgs::msg::Path& lookahead_path);

    // // save and load a path file
    // void savePathToFile(const nav_msgs::msg::Path& path);
    // nav_msgs::msg::Path loadPathFromFile();

    // Service callback functions
    
    PathLogger path_logger;

private:
    // ROS node and publishers/subscribers
    
    ros::NodeHandle traj_nh;
    ros::ServiceServer path_record_init_srv, path_record_stop_srv, path_save_srv,path_read_srv ;
    

    // Path recording variables
    bool is_recording_;
    
    // Mutex for thread safety
    std::mutex mutex_;
    Trajectory lookahead_traj;
    Trajectory tmp_ref_traj;
    
    
};







#endif  // TRAJECTORY_MANAGER_HPP_
