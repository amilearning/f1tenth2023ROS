#ifndef TRACKS
#define TRACKS

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "state_and_types.h"
#include "utils.h"

class Track {
private:
    std::vector<KeyPoint> key_pts;    
    std::vector<std::pair<double, double>> segments; 
    double cumulative_length = 0.0;
    double track_width;
    double slack;
    double initial_x, initial_y, initial_psi;   
    double track_length; 

public:
    Track(double initial_x = 0, double initial_y = 0, double initial_psi = 0, double track_width = 1.6, double slack = 0.1);

    void test_local_and_global();
    std::tuple<double, double, double> nextPositionOrientation(double x, double y, double psi, double curvature, double segment_length);
    void getKeyPts();
    FrenPose globalToLocal(const Pose& pos_cur) const;
    Pose localToGlobal(const FrenPose& cl_coord);
    double get_curvature(const double& s);
    torch::Tensor torch_wrap_s_single(const torch::Tensor& s_, double track_length);
    
};

#endif