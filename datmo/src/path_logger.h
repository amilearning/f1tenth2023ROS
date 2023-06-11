#ifndef TRAJECTORY_MANAGER_HPP_
#define TRAJECTORY_MANAGER_HPP_

#include <ros/package.h>
#include <ros/ros.h>
#include <ros/time.h>

#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include <mutex>
#include <vector> 
#include <cmath>

#include <tuple>
#include <algorithm>


#include "state.h"
#include <tf/tf.h>
#include <thread>
#include <sstream>
#include <string>
#include "trajectory.h"





class PathLogger {
public:
    PathLogger(double threshold) : threshold_(threshold) {

        ref_traj.clear();
    }



    double dist(double x1, double y1, double x2, double y2)
    {
            return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
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
    

    void updataPath(const visualization_msgs::MarkerArray& marker_data){
        ref_traj.clear();
        // Read the path from the file
      
        double x, y,  vx, yaw, roll, pitch;
        double prev_x  = marker_data.markers[0].pose.position.x;
        double prev_y  = marker_data.markers[0].pose.position.y;

        
        ///////////////////////////////////////
        double dist_sample_thres = 0.1;
        std::vector<int> sampled_idx;
        ///////////////////////////////////////
        for(int i=0; i < marker_data.markers.size(); i++){
            x = marker_data.markers[i].pose.position.x;
            y = marker_data.markers[i].pose.position.y;
            
           

            double dist_tmp = sqrt((x-prev_x)*(x-prev_x)+(y-prev_y)*(y-prev_y));
            if  ( dist_tmp > dist_sample_thres ){
                    prev_x = x;
                    prev_y= y;
                    sampled_idx.push_back(i);
                } else{
                    continue;
                }
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
            double s = marker_data.markers[i].pose.orientation.x;  
            double ey_l = marker_data.markers[i].pose.orientation.y;
            double ey_r = marker_data.markers[i].pose.orientation.z;
            // tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
            ref_traj.push_back(x,y,0.0, yaw, vx, 0.0, 0.0, 0.0, s, ey_l, ey_r);
        }
       
        // encode frenet coordinate 
             // update s 
        ref_traj.s.clear();
        double accumulated_dist = 0.0;
        ref_traj.s.push_back(0.0);
        for (int i = 1; i < ref_traj.size(); i++) {
            accumulated_dist +=     dist(ref_traj.x[i],ref_traj.y[i], ref_traj.x[i-1], ref_traj.y[i-1]);
            ref_traj.s.push_back(accumulated_dist);
        }
        //
            // Update curvature  
        ref_traj.k.clear();
        for (int i =   0; i < ref_traj.size(); i++) {        
        ref_traj.k.push_back(marker_data.markers[sampled_idx[i]].pose.orientation.x);
        }

        // std::cout << "ref_traj size = " << ref_traj.size() << std::endl;
    }
    
    
   

    double threshold_;
    Trajectory ref_traj;      
    double x_tmp, y_tmp, yaw_tmp, vx_tmp; 

};





#endif  // TRAJECTORY_MANAGER_HPP_
