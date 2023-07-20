from barcgp.common.pytypes import VehicleState, VehiclePrediction
from abc import abstractmethod
from matplotlib import pyplot as plt

import numpy as np
import copy
from typing import List
import gc
import torch, gpytorch

from predictor.common.tracks.radius_arclength_track import RadiusArclengthTrack
from predictor.h2h_configs import nl_mpc_params, N
from predictor.controllers.utils.controllerTypes import *
from predictor.prediction.trajectory_predictor import BasePredictor
from predictor.prediction.thetaGP.ThetaGPModel import ThetaGPTrained
from predictor.prediction.cont_encoder.cont_policyEncoder import ContPolicyEncoder
from predictor.prediction.cont_encoder.cont_encoderdataGen import states_to_encoder_input_torch

class ContThetaPolicyPredictor(BasePredictor):
    def __init__(self, N: int, track : RadiusArclengthTrack, policy_name: str, use_GPU: bool, M: int, model=None, cov_factor: float = 1):
        super(ContThetaPolicyPredictor, self).__init__(N, track)
        gc.collect()
        torch.cuda.empty_cache()        
        
        ######### Policy extractor based one LSTMAutoencoder ######### 
        self.encoder_args =  {"batch_size": 128,
                            "device": torch.device("cuda")
                            if torch.cuda.is_available()
                            else torch.device("cpu"),
                            "input_size": 9,
                            "hidden_size": 8,
                            "latent_size": 4,
                            "learning_rate": 0.001,
                            "max_iter": 30000,
                            "sequence_length": 5
                            }
        
        self.policy_encoder = ContPolicyEncoder(model_load = True)
        print("policy extractor loaded")
        ######### Input prediction for Gaussian Processes regression ######### 
        input_predict_model = "thetaGP"
        self.theta_predict_gp = ThetaGPTrained(input_predict_model, use_GPU)
        print("input predict_gp loaded")
    
        self.M = M  # number of samples
        self.cov_factor = cov_factor
        self.ego_state_buffer = []
        self.tar_state_buffer = []
        self.time_length = self.policy_encoder.seq_len
        ## berkely GP can be used if the buffer is empty, currently only 
        # self.gp = GPControllerTrained(policy_name, use_GPU, model)

        self.ego_state_buffer = []
        self.tar_state_buffer = []
        

        self.encoder_input = torch.zeros(self.encoder_args["sequence_length"], self.encoder_args["input_size"])
        self.buffer_update_count  = 0
        
        


    def append_vehicleState(self,ego_state: VehicleState,tar_state: VehicleState):     
            ######################### rolling into buffer ##########################                        
            tmp = self.encoder_input.clone()
            self.encoder_input[0:-1,:] = tmp[1:,:]
            encoder_input_tmp = states_to_encoder_input_torch(tar_state,ego_state)
            self.encoder_input[-1,:] = encoder_input_tmp
            self.buffer_update_count +=1
            if self.buffer_update_count > self.encoder_args["sequence_length"]:
                self.buffer_update_count = self.encoder_args["sequence_length"]+1
                return True
            else:
                return False

    def get_prediction(self, ego_state: VehicleState, target_state: VehicleState,
                       ego_prediction: VehiclePrediction, tar_prediction=None):

        is_encoder_input_ready = self.append_vehicleState(ego_state,target_state)        

        # theta = None
        # if is_encoder_input_ready:
        #     ## Get the latest control policy "theta" given the buffer, 
        #     theta = self.policy_encoder.get_theta_from_buffer(self.encoder_input)
        #     theta[:] = 0
        # #############
        
        #############
        if is_encoder_input_ready: ## encoder_is_ready = True
            pred = self.theta_predict_gp.get_true_prediction_par(self.policy_encoder,self.encoder_input,  ego_state, target_state, ego_prediction, self.track, self.M)            
        else:
            pred = self.get_constant_vel_prediction_par(target_state) # self.gp.get_true_prediction_par(ego_state, target_state, ego_prediction, self.track, self.M)


        # TODO  if input_prediction_is_ready ## 
            ## TODO: get the dynamics residual given current target vehicle states 

        # TODO: target and ego vehicles using dynamics and predicted input

        
        # fill in covariance transformation to x,y
        pred.track_cov_to_local(self.track, self.N, self.cov_factor)
        
        return pred



    
    def get_constant_vel_prediction_par(self, target_state: VehicleState):
        target_state.update_global_velocity_from_body()
        t = target_state.t
        x = target_state.x.x
        y = target_state.x.y
        v_x = target_state.v_x
        v_y = target_state.v_y
        psidot = target_state.w.w_psi
        psi = target_state.e.psi

        t_list = np.zeros((self.N))
        x_list = np.zeros((self.N))
        y_list = np.zeros((self.N))
        psi_list = np.zeros((self.N))

        delta_psi = 0
        for i in range(self.N):
            x_list[i] = x
            y_list[i] = y
            t_list[i] = t
            psi_list[i] = psi
            t += self.dt
            x += self.dt * (v_x * np.cos(delta_psi) - v_y * np.sin(delta_psi))
            y += self.dt * (v_y * np.cos(delta_psi) + v_x * np.sin(delta_psi))


            delta_psi += self.dt * psidot
            psi += self.dt * psidot

        pred = VehiclePrediction(t=target_state.t, x=x_list, y=y_list, psi=psi_list)
        pred.xy_cov = np.repeat(np.diag([self.cov, self.cov])[np.newaxis, :, :], self.N, axis=0)
        return pred