#pragma once

#include <utility> // For std::pair
#include <vector>
#include <geometry_msgs/PoseStamped.h>  // Assuming you have this dependency

// Define a Point as a pair of doubles (x, y coordinates)
typedef std::pair<double, double> Point;



struct Pose {
    double x;
    double y;
    double psi;
};

struct FrenPose{
    double s;
    double ey;
    double epsi;
};

// Segment structure for track segments
struct KeyPoint {
    double x;
    double y;
    double psi;
    double cum_length;
    double segment_length;
    double curvature;
};

// VehicleState structure for representing the state of a vehicle
struct VehicleState {
    geometry_msgs::Pose pose;  // Pose information (position and orientation)
    double yaw;                // Yaw angle
    double vx;                 // Velocity in local x direction
    double vy;                 // Velocity in local y direction
    double wz;                 // Angular velocity in local z direction
    double accel;              // Acceleration
    double delta;              // Steering angle
    double s;                  // Progress along the track in Frenet coordinates
    double ey;                 // Lateral deviation from the centerline
    double epsi;               // Orientation deviation from the centerline
    double k;                  // Curvature at the closest point on the centerline
};
