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
    double s; // progress in frenet 
    double ey; // deviation to centerline 
    double epsi; // deviation angle to centerline heading
    double k; // curvature of the cloeset centerline
};


struct ControlInput {
    double accel;
    double delta;
    
};

struct KeyPoints{
    Eigen::MatrixXd s_curv; 
};



#endif  // CONTROLLER_HPP_
