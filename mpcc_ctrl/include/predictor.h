#ifndef PREDICTOR
#define PREDICTOR

#include <deque>
#include <ros/ros.h>
#include <ros/package.h>
#include "track.h"
#include "utils.h"
#include "state_and_types.h"
#include <vector>
#include <string>
#include "vehicleState.h"

// Include other necessary headers such as for ROS and any custom classes

class CovGPPredictor {
public:
    explicit CovGPPredictor(const Track &track_);
    ~CovGPPredictor();
    
    
    bool append_vehicleState(const VehicleState& ego_state, const VehicleState& tar_state);
    
    
    
   

    
    
    

private:    
    
    std::deque<std::vector<double>> input_buffer; // Buffer declaration    
    ModelConfig config;
    Track track;       
    // Other member variables...
};

// Define other necessary methods and utilities

#endif 