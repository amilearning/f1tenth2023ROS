
#ifndef VEHICLESTATE_H
#define VEHICLESTATE_H

#include <std_msgs/Header.h>
#include <geometry_msgs/Pose.h>
#include <nav_msgs/Odometry.h>
#include <cmath>
// #include "state_and_types.h"
// #include "utils.h"
#include "track.h"
// Include other necessary headers...


struct VehicleState {
    double t;
    std_msgs::Header header;
    geometry_msgs::Pose pose;  // Pose information (position and orientation)
    FrenPose p; 
    double yaw;                // Yaw angle
    double vx;                 // Velocity in local x direction
    double vy;                 // Velocity in local y direction
    double wz;                 // Angular velocity in local z direction
    double accel;              // Acceleration
    double delta;              // Steering angle    
    double curv;               // Curvature at the closest point on the centerline
    double lookahead_curv;     // Curvature at the lookahead point

    void fromOdometry(const nav_msgs::Odometry& odom);
    void updateFrenet(const Track& track); 
};

#endif // VEHICLESTATE_H


// // VehicleState structure for representing the state of a vehicle
// struct VehicleState {
//     double t;
//     std_msgs::Header header;
//     geometry_msgs::Pose pose;  // Pose information (position and orientation)
//     FrenPose p; 
//     double yaw;                // Yaw angle
//     double vx;                 // Velocity in local x direction
//     double vy;                 // Velocity in local y direction
//     double wz;                 // Angular velocity in local z direction
//     double accel;              // Acceleration
//     double delta;              // Steering angle    
//     double curv;                  // Curvature at the closest point on the centerline
//     double lookahead_curv;                  // Curvature at the closest point on the centerline
        
//     void fromOdometry(const nav_msgs::Odometry& odom) {
//         this->t = odom.header.stamp.toSec();
//         this->header = odom.header;
//         this->pose = odom.pose.pose;        
//         // tf::Vector3 euler_angles = quaternionToEuler(odom.pose.pose.orientation);
//         // this->yaw = euler_angles.z();  // Calculate or extract the yaw angle from the pose
//         // double vx_local = odom.twist.twist.linear.x * cos(euler_angles.z()) + odom.twist.twist.linear.y * sin(euler_angles.z());
//         // double vy_local = -odom.twist.twist.linear.x * sin(euler_angles.z()) + odom.twist.twist.linear.y * cos(euler_angles.z());
//         // this->vx = vx_local;
//         // this->vy = vy_local;
//         // this->wz = 0.0;       
//     }

//     // void updateFrenet(const Track& track){
//     //     Pose tmp_pose = {this->pose.position.x, this->pose.position.y, this->pose.position.z};
//     //     FrenPose fren_pose = track.globalToLocal(tmp_pose);
//     //     this->p = fren_pose;
//     // }    
// };