#include "predictor.h"

CovGPPredictor::CovGPPredictor(const Track &track_) : track(track_), 
      device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) // Initialize 'device' here
// CovGPPredictor::CovGPPredictor(const Track & track_)
//     :track(track_)      
{   
    readJsonModel();


    config = ModelConfig();
    if (torch::cuda::is_available()) {
      std::cout << "CUDA is available! Training on GPU." << std::endl;
      device = torch::kCUDA;
    }

    try {
        module = torch::jit::load("/home/hjpc/torch_pkg/001_example/traced_model.pt");
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";        
    }
    module.to(device);
    test_x= torch::randn({50, 8}).to(device);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(test_x);
    output = module.forward(inputs);        

    
    input_buffer = torch::zeros({config.input_dim, config.n_time_step}).to(device);
    
}



CovGPPredictor::~CovGPPredictor()
{}

void CovGPPredictor::readJsonModel(){
    std::string packagePath = ros::package::getPath("mpcc_ctrl");        
    std::string jsonFilePath = packagePath + "/models/data.json";    
    std::tuple<torch::Tensor, torch::Tensor> tensors =jsonFileToTensor(jsonFilePath);
    means_y = std::get<0>(tensors);
    stds_y = std::get<1>(tensors);
}


torch::Tensor CovGPPredictor::states_to_encoder_input_torch(const VehicleState& tar_st, const VehicleState& ego_st) {
    double tar_s = tar_st.p.s;
    double ego_s = ego_st.p.s;
    double delta_s = tar_s - ego_s;
    // Create a tensor from the input data
    torch::Tensor input_data = torch::tensor({delta_s,
                                              tar_st.p.ey,
                                              tar_st.p.epsi,
                                              tar_st.vx,
                                              tar_st.curv,
                                              tar_st.lookahead_curv,
                                              ego_st.p.ey,
                                              ego_st.p.epsi, 
                                              ego_st.vx});

    return input_data;
}



bool CovGPPredictor::append_vehicleState(const VehicleState& ego_state, const VehicleState& tar_state) {
        // Rolling into buffer
        auto tmp = input_buffer.clone();
        input_buffer.slice(1, 0, -1) = tmp.slice(1, 1); // Shift left
        auto new_column = states_to_encoder_input_torch(tar_state, ego_state); 
        input_buffer.slice(1, -1) = new_column;
    }



std::vector<VehicleState> CovGPPredictor::get_constant_vel_prediction_par(const VehicleState& target_state) {
        
        std::vector<VehicleState> predictions;
        VehicleState roll_state = target_state; // Assuming a copy constructor exists
        predictions.push_back(roll_state);
        for (int i = 0; i < config.n_time_step; ++i) {
            VehicleState tmp_state = roll_state;            
            double vel_global_x = tmp_state.vx * cos(tmp_state.yaw) - tmp_state.vy * sin(tmp_state.yaw);
            double vel_global_y = tmp_state.vx * sin(tmp_state.yaw) + tmp_state.vy * cos(tmp_state.yaw);
            tmp_state.pose.position.x = roll_state.pose.position.x + vel_global_x * config.dt;
            tmp_state.pose.position.y = roll_state.pose.position.y + vel_global_y * config.dt;                        
            predictions.push_back(tmp_state);
        }


        return predictions;
}




torch::Tensor CovGPPredictor::sample_traj_covGPNN(const torch::Tensor& input_buffer,
                                const VehicleState& ego_state,
                                const VehicleState& target_state,
                                const std::vector<VehicleState>& ego_prediction,
                                int M){
                                
            
        torch::Tensor roll_input = input_buffer.repeat({M, 1, 1}).to(device);        
        
        torch::Tensor tar_state_tensor = torch::tensor({target_state.p.s, target_state.p.ey, target_state.p.epsi, target_state.vx});
        torch::Tensor roll_tar_state = tar_state_tensor.repeat({M,1}).to(device);        
        
        torch::Tensor tar_curv = torch::tensor({target_state.curv, target_state.lookahead_curv});        
        torch::Tensor roll_tar_curv = tar_curv.repeat({M,1}).to(device);

        torch::Tensor ego_state_tensor = torch::tensor({ego_state.p.s, ego_state.p.ey, ego_state.p.epsi, ego_state.vx});
        torch::Tensor roll_ego_state = ego_state_tensor.repeat({M,1}).to(device);        
        
        std::vector<torch::jit::IValue> inputs_to_trace;
        for (int i=0; i<config.n_time_step; ++i){
            insert_to_end(roll_input, roll_tar_state, roll_tar_curv, roll_ego_state);                                      
            inputs_to_trace.push_back(roll_input);
            auto output = module.forward(inputs_to_trace);                   
            auto output_tuple = output.toTuple();
            torch::Tensor mean = output_tuple->elements()[0].toTensor();
            torch::Tensor variance = output_tuple->elements()[1].toTensor();
            torch::Tensor stddev = torch::sqrt(variance);            
            torch::Tensor samples = at::normal(mean, stddev);
            torch::Tensor residual_tmp = outputToReal(samples);
            roll_tar_state.slice(1, 0, 1) += residual_tmp.slice(1,0,1);
            roll_tar_state.slice(1, 1, 2) += residual_tmp.slice(1,1,2); 
            roll_tar_state.slice(1, 2, 3) += residual_tmp.slice(1,2,3);
            roll_tar_state.slice(1, 3, 4) += residual_tmp.slice(1,3,4);  
            // roll_tar_curv.slice(1, 0, 1) = get_curvature_from_keypts_torch(pred_delta[:,0].clone().detach(),track)
            // roll_tar_curv.slice(1, 1, 2) = get_curvature_from_keypts_torch(pred_delta[:,0].clone().detach()+target_state.lookahead.dl*2,track)                                     
            roll_ego_state.slice(1, 0, 1) = torch::full({M, 1}, ego_prediction[i].p.s);
            roll_ego_state.slice(1, 1, 2) =torch::full({M, 1}, ego_prediction[i].p.ey);  
            roll_ego_state.slice(1, 2, 3) =torch::full({M, 1}, ego_prediction[i].p.epsi);   
            roll_ego_state.slice(1, 3, 4) =torch::full({M, 1}, ego_prediction[i].vx);

       }        

}



torch::Tensor CovGPPredictor::outputToReal(const torch::Tensor& output) {
    torch::Tensor standardized_tensor = output * stds_y + means_y;
    return standardized_tensor;
}


void CovGPPredictor::insert_to_end(torch::Tensor& roll_input, 
                            const torch::Tensor& tar_state, 
                            const torch::Tensor& tar_curv, 
                            const torch::Tensor& ego_state) {

    // Shift elements in roll_input
    roll_input.slice(2, 0, -1) = roll_input.slice(2, 1);

    // Create a new tensor for input_tmp
    torch::Tensor input_tmp = torch::zeros({roll_input.size(0), roll_input.size(1)}).to(device);

    // Fill input_tmp with values
    input_tmp.slice(1, 0, 1) = tar_state.slice(1, 0, 1) - ego_state.slice(1, 0, 1);
    input_tmp.slice(1, 1, 2) = tar_state.slice(1, 1, 2);
    input_tmp.slice(1, 2, 3) = tar_state.slice(1, 2, 3);
    input_tmp.slice(1, 3, 4) = tar_state.slice(1, 3, 4);
    input_tmp.slice(1, 4, 5) = tar_curv.slice(1, 0, 1);
    input_tmp.slice(1, 5, 6) = tar_curv.slice(1, 1, 2);
    input_tmp.slice(1, 6, 7) = ego_state.slice(1, 1, 2);
    input_tmp.slice(1, 7, 8) = ego_state.slice(1, 2, 3);
    input_tmp.slice(1, 8, 9) = ego_state.slice(1, 3, 4);

    // Insert input_tmp into roll_input
    roll_input.slice(2, -1) = input_tmp;

}


