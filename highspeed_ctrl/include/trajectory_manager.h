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
#include "utils.h"


class PathLogger {
public:
    PathLogger(double threshold) : threshold_(threshold) {

        double dt = 0.1;
        double lpf_cutoff_hz = 0.1;
        x_filter.initialize(dt, lpf_cutoff_hz);
        y_filter.initialize(dt, lpf_cutoff_hz);
        yaw_filter.initialize(dt, lpf_cutoff_hz);
        vx_filter.initialize(dt, lpf_cutoff_hz);
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
        // yaw_tmp = yaw_filter.filter(yaw);
        
        vx_tmp = vx_filter.filter(odom.twist.twist.linear.x);
        yaw_tmp = normalizeRadian(yaw);
        if (distance >= threshold_ && is_record) {
            double lkh=0.0;
            ref_traj.push_back(x_tmp,y_tmp, 0.0, yaw_tmp, vx_tmp, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0,lkh);  // fake distance for left and right wall
            
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
        
        ofs << ref_traj.x[i] << " " << ref_traj.y[i] << " " << ref_traj.yaw[i] << " " << ref_traj.vx[i] <<  "\n";
    }

    // Close the file
    ofs.close();
    }
    
    void sampleUniformPath(Trajectory & traj,const double minDist){
        std::cout << "Uniform path sampling init" <<std::endl;
        std::vector<std::vector<double>> resampled_path(traj.path.size());
        double last_x = traj.path[0].front();
        double last_y = traj.path[1].front();
        // initialize waypoints
        for(int j=0;  j< traj.path.size(); ++j){
                    resampled_path[j].push_back(traj.path[j][0]); 
        } 


        for(int i = 1;  i < traj.path[0].size(); ++i){
            double dx = traj.path[0][i] - last_x;
            double dy = traj.path[1][i] - last_y;
            double dist = std::sqrt(dx*dx + dy*dy);
            if (dist > minDist) {
                for(int j=0;  j< traj.path.size(); ++j){
                    resampled_path[j].push_back(traj.path[j][i]); 
                }            
            last_x = traj.path[0][i];
            last_y = traj.path[1][i];
            }
        } 
        
        traj.encode_from_path_to_traj(resampled_path);
        std::cout << "Sample path size =  " << traj.size() << std::endl;
    }

    void updataPath(const visualization_msgs::MarkerArray& marker_data){
        ref_traj.clear();
        // Read the path from the file
      
        double x, y,  vx, yaw, roll, pitch;
        for(int i=0; i < marker_data.markers.size(); i++){
            x = marker_data.markers[i].pose.position.x;
            y = marker_data.markers[i].pose.position.y;
            vx = marker_data.markers[i].pose.position.z;
            
            // orientation.x -->  curvature 
            // orientation.y -->  left wall width
            // orientation.z --> right wall width 
            
            if(i < marker_data.markers.size()-3){
                double diff_x = marker_data.markers[i+3].pose.position.x - marker_data.markers[i].pose.position.x;
                double diff_y = marker_data.markers[i+3].pose.position.y - marker_data.markers[i].pose.position.y;
                yaw = atan2(diff_y, diff_x);                
            }
            // tf::Quaternion q(marker_data.markers[i].pose.orientation.x, 

            //                 marker_data.markers[i].pose.orientation.y,
            //                 marker_data.markers[i].pose.orientation.z,
            //                 marker_data.markers[i].pose.orientation.w);                            
            // q.normalize();
            // Extract the yaw angle from the quaternion object       
            double ey_l = marker_data.markers[i].pose.orientation.y;
            double ey_r = marker_data.markers[i].pose.orientation.z;
            double lkh = marker_data.markers[i].color.r;
            // tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
              ref_traj.push_back(x,y,0.0, yaw, vx, 0.0, 0.0, 0.0, 0.0, ey_l, ey_r, lkh);
        }
       
        // encode frenet coordinate 
             // update s 
        ref_traj.s.clear();
        double accumulated_dist = 0.0;
        ref_traj.s.push_back(0.0);
        for (int i = 1; i < marker_data.markers.size(); i++) {
            accumulated_dist +=     dist(ref_traj.x[i],ref_traj.y[i], ref_traj.x[i-1], ref_traj.y[i-1]);
            ref_traj.s.push_back(accumulated_dist);
        }
        //
            // Update curvature  
        ref_traj.k.clear();
        for (int i =   0; i < marker_data.markers.size(); i++) {        
        ref_traj.k.push_back(marker_data.markers[i].pose.orientation.x);
        }

        std::cout << "ref_traj size = " << ref_traj.size() << std::endl;
        ref_traj.encode_traj_to_path_info();
        double traj_waypoints_min_dist = 0.1;
        sampleUniformPath(ref_traj, traj_waypoints_min_dist);
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
    double x, y,  yaw, vx;
    double lkh = 0.0;
    while (ifs >> x >> y >>yaw >> vx ) {
        
        ref_traj.push_back(x,y,0.0, yaw, vx, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0,lkh); // fake distance for ey_l, ey_r
        
        
    }
    // encode frenet coordinate 
    calcTrajectoryFrenet(ref_traj, 5);
    
    std::cout << "ref_traj size = " << ref_traj.size() << std::endl;
    // Close the file
    ifs.close();

    }



    double threshold_;
    Trajectory ref_traj;    
    nav_msgs::Odometry last_odom;
    Butterworth2dFilter x_filter,y_filter, yaw_filter,  vx_filter;    
    double x_tmp, y_tmp, yaw_tmp, vx_tmp; 

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
    visualization_msgs::MarkerArray traj_to_markerArray(const Trajectory & traj, const std_msgs::ColorRGBA & color_);
    unsigned int getClosestIdx(double& closest_dist, const Trajectory & traj_, const VehicleState& vehicle_state);
    void updatelookaheadPath( const VehicleState& vehicle_state, const double& length);
    Trajectory getlookaheadPath(); 
    Trajectory getglobalPath();
    bool getCurvatureKeypoints(KeyPoints & key_pts);
    visualization_msgs::Marker getLocalPathMarker();
    visualization_msgs::MarkerArray getLocalPathMarkerArray();
    
    visualization_msgs::Marker getGlobalPathMarker();
    visualization_msgs::Marker getCenterLineInfo();
    visualization_msgs::Marker keyptsToMarker(const KeyPoints & key_pts);
    int getRefTrajSize(); 
    double getTrackLength();
    void wrapTrack(double & s);
    double get_s_diff(const double &s1, const double &s2);
    bool is_s_front(const double &s, const double &s_target);
    bool is_recording();
    // // extract a lookahead path in frenet coordinate given the current position (odometry)
    // void extractLookaheadPath(const nav_msgs::msg::Odometry& odom, nav_msgs::msg::Path& lookahead_path);

    // // save and load a path file
    // void savePathToFile(const nav_msgs::msg::Path& path);
    // nav_msgs::msg::Path loadPathFromFile();

    // Service callback functions
    
    PathLogger path_logger;
        bool frenet_ready;
    

private:
    // ROS node and publishers/subscribers
    
    ros::NodeHandle traj_nh;
    ros::ServiceServer path_record_init_srv, path_record_stop_srv, path_save_srv,path_read_srv ;

    bool is_recording_;
    // Path recording variables
    bool is_first_lookahead;
    
    // Mutex for thread safety
    std::mutex mutex_;
    Trajectory lookahead_traj; // local lookahead centerline (mainly used py pp_ctrl)
    Trajectory tmp_ref_traj;  // global centerline
    Trajectory global_ref_traj;
    Trajectory local_ref_traj;
    unsigned int local_ref_traj_idx_in_global;
    Trajectory curv_info_traj; // local lookahead centerline for curvature estimation
    
    
};







#endif  // TRAJECTORY_MANAGER_HPP_
