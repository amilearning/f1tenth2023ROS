#ifndef STATESTYPES
#define STATESTYPES



#include <utility> // For std::pair
#include <vector>
#include <geometry_msgs/PoseStamped.h>  // Assuming you have this dependency
#include <torch/torch.h>
#include <torch/script.h>

// Define a Point as a pair of doubles (x, y coordinates)
typedef std::pair<double, double> Point;



struct ModelConfig {
    int batch_size;
    torch::Device device;
    int input_dim;
    int n_time_step;
    int latent_dim;
    int gp_output_dim;
    int inducing_points;
    double dt;

    ModelConfig() 
        : batch_size(100), // Assuming last 'batch_size' value from Python code
          device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
          input_dim(9),
          n_time_step(10),
          latent_dim(4),
          gp_output_dim(4),
          inducing_points(300),
          dt(0.1)
    {
        // Constructor body, if needed
    }
};

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
    double s;                  // Progress along the track in Frenet coordinates
    double ey;                 // Lateral deviation from the centerline
    double epsi;               // Orientation deviation from the centerline
    double curv;                  // Curvature at the closest point on the centerline
    double lookahead_curv;                  // Curvature at the closest point on the centerline

    
};

#endif