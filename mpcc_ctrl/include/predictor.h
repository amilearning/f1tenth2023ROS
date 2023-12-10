#ifndef PREDICTOR
#define PREDICTOR


#include <ros/ros.h>
#include <ros/package.h>
#include "track.h"
#include "utils.h"
#include "state_and_types.h"
#include <vector>
#include <string>


// Include other necessary headers such as for ROS and any custom classes

class CovGPPredictor {
public:
    explicit CovGPPredictor(const Track &track_);
    ~CovGPPredictor();
    
    

    
    bool append_vehicleState(const VehicleState& ego_state, const VehicleState& tar_state);
    torch::Tensor states_to_encoder_input_torch(const VehicleState& tar_st, const VehicleState& ego_st);
    std::vector<VehicleState> get_constant_vel_prediction_par(const VehicleState& target_state);
    torch::Tensor sample_traj_covGPNN(const torch::Tensor& input_buffer,
                                const VehicleState& ego_state,
                                const VehicleState& target_state,
                                const std::vector<VehicleState>& ego_prediction,
                                int M);

    void insert_to_end(torch::Tensor& roll_input, 
                            const torch::Tensor& tar_state, 
                            const torch::Tensor& tar_curv, 
                            const torch::Tensor& ego_state);

    torch::Tensor outputToReal(const torch::Tensor& output);
    void readJsonModel();
    

private:    
    torch::Tensor input_buffer;
    torch::Device device;     
    torch::jit::script::Module module;    
    torch::Tensor test_x; 
    torch::jit::IValue output;
     int sample_num;
    double cov_scale;
    int horizon; 
    ModelConfig config;
    Track track;    
    torch::Tensor means_y, stds_y;
    // Other member variables...
};

// Define other necessary methods and utilities

#endif 