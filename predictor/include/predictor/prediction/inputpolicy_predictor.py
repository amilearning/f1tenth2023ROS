from barcgp.common.pytypes import VehicleState, VehiclePrediction
from abc import abstractmethod
from matplotlib import pyplot as plt

import numpy as np
import copy
from typing import List
import gc
import torch, gpytorch

from barcgp.common.tracks.radius_arclength_track import RadiusArclengthTrack
from barcgp.h2h_configs import nl_mpc_params, N
from barcgp.controllers.utils.controllerTypes import *
from barcgp.prediction.trajectory_predictor import BasePredictor
from barcgp.prediction.dynGP.dynGPmodel import DynGPApproximateTrained
from barcgp.prediction.InputGP.InputGPModel import InputPredictionApproximateTrained
from barcgp.prediction.encoder.policyEncoder import PolicyEncoder 
from barcgp.prediction.Ikd.IkdPredictor import IkdPredictor
from barcgp.prediction.gp_controllers import GPControllerTrained

class InputPolicyPredictor(BasePredictor):
    def __init__(self,vehicle_model, N: int, track : RadiusArclengthTrack, policy_name: str, use_GPU: bool, M: int, model=None, cov_factor: float = 1):
        super(InputPolicyPredictor, self).__init__(N, track)
        gc.collect()
        torch.cuda.empty_cache()
        self.vehicle_dynamics = vehicle_model
        ######### Inverse Kinodynamics prediction based one ConvNeuralNetwork ######### 
        ikd_model = 100
        self.ikd_predictor = IkdPredictor(model_load = True, model_id = ikd_model)
        print("ikd predictor loaded")
        ######### Policy extractor based one LSTMAutoencoder ######### 
        encoder_model = 100
        self.policy_encoder = PolicyEncoder(model_load = True, model_id = encoder_model)
        print("policy extractor loaded")
        ######### Input prediction for Gaussian Processes regression ######### 
        input_predict_model = "inputGP"
        self.input_predict_gp = InputPredictionApproximateTrained(input_predict_model, use_GPU)
        print("input predict_gp loaded")
        ######### Dynamics residual prediction for Gaussian processes regression ######### 
        dynGP_model = "dynGP"
        self.dyngp = DynGPApproximateTrained(dynGP_model, use_GPU)
        print("dynamics residual GP loaded")

        self.M = M  # number of samples
        self.cov_factor = cov_factor
        self.ego_state_buffer = []
        self.tar_state_buffer = []
        self.time_length = self.policy_encoder.seq_len
        self.gp = GPControllerTrained(policy_name, use_GPU, model)

    def get_prediction(self, ego_state: VehicleState, target_state: VehicleState,
                       ego_prediction: VehiclePrediction, tar_prediction=None):

        self.ikd_predictor.append_vehicleState(ego_state,target_state)        
        encoder_swith = False
        if self.ikd_predictor.ego_buffer_for_encoder is not None:
            encoder_swith = True

        theta = None
        if encoder_swith and self.ikd_predictor.ego_buffer_for_encoder.shape[0] >= self.time_length:
            ## Get the latest control policy "theta" given the buffer, 
            theta = self.policy_encoder.get_theta_from_buffer(self.ikd_predictor.ego_buffer_for_encoder.clone(), self.ikd_predictor.tar_buffer_for_encoder.clone())
        
        if theta is not None: ## encoder_is_ready = True
            recent_ego_state = self.ikd_predictor.ego_state_buffer_torch[-1,:]
            recent_tar_state = self.ikd_predictor.tar_state_buffer_torch[-1,:]            
            pred = self.input_predict_gp.get_true_prediction_par(self.vehicle_dynamics, theta, ego_state, target_state, ego_prediction, self.track, self.M)            
        else:
            pred = self.gp.get_true_prediction_par(ego_state, target_state, ego_prediction, self.track, self.M)


        # TODO  if input_prediction_is_ready ## 
            ## TODO: get the dynamics residual given current target vehicle states 

        # TODO: target and ego vehicles using dynamics and predicted input

        
        # fill in covariance transformation to x,y
        pred.track_cov_to_local(self.track, self.N, self.cov_factor)
        
        return pred



    