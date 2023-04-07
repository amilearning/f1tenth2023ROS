#ifndef STATE_HPP_
#define STATE_HPP_

#include <vector>
#include <Eigen/Dense>
#include <chrono>
#include <cmath>

#include <geometry_msgs/Pose.h>

// Path (x,y,psi)

// Trajectory includes Speed reference (x,y,psi,speed)

using PathPoint = Eigen::Vector2d;
struct VehicleState {
    geometry_msgs::Pose pose;
    double yaw;
    double vx;  // vel in local 
    double vy; // vel in local
    double wz; // vel in local
    double accel;
    double delta;  
};


struct ControlInput {
    double accel;
    double delta;
    
};





#endif  // CONTROLLER_HPP_
