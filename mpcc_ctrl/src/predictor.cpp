#include "predictor.h"

CovGPPredictor::CovGPPredictor(const Track &track_) : track(track_)
{   
    
    config = ModelConfig();    
    for (size_t i = 0; i < config.n_time_step; ++i) {
        input_buffer.push_back(std::vector<double>(config.input_dim));
    }

    
}



CovGPPredictor::~CovGPPredictor()
{}





bool CovGPPredictor::append_vehicleState(const VehicleState& ego_state, const VehicleState& tar_state) {
        // Rolling into buffer 
        std::vector<double> input_data(9);
        input_data[0] = tar_state.p.s - ego_state.p.s,
        input_data[1] = tar_state.p.ey,
        input_data[2] = tar_state.p.epsi,
        input_data[3] = tar_state.vx,
        input_data[4] = tar_state.curv,
        input_data[5] = tar_state.lookahead_curv,
        input_data[6] = ego_state.p.ey,
        input_data[7] = ego_state.p.epsi, 
        input_data[8] = ego_state.vx;
    
        input_buffer.push_back(input_data); // Add new array at the back
        if (input_buffer.size() > config.n_time_step) {
            input_buffer.pop_front(); // Remove the first array to maintain the size
            return true;
        }
        else{
            return false;
        }
    
    }







