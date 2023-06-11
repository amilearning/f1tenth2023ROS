import torch 
import array
import copy
import sys
import time
import gpytorch
from typing import Type, List
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
import secrets
from predictor.prediction.abstract_gp_controller import GPController
from predictor.h2h_configs import *
from predictor.common.utils.file_utils import *
from gpytorch.mlls import SumMarginalLogLikelihood
from predictor.prediction.gpytorch_models import ExactGPModel, MultitaskGPModel, MultitaskGPModelApproximate, \
    IndependentMultitaskGPModelApproximate
from predictor.prediction.thetaGP.ThetaGPdataGen import SampleGeneartorThetaGP
import os
from predictor.prediction.dyn_prediction_model import DynamicsModelForPredictor, TorchDynamicsModelForPredictor
from predictor.common.tracks.radius_arclength_track import RadiusArclengthTrack

class ThetaGPApproximate(GPController):
    def __init__(self, sample_generator: SampleGeneartorThetaGP, model_class: Type[gpytorch.models.GP],
                 likelihood: gpytorch.likelihoods.Likelihood, input_size: int, output_size: int, inducing_points: int,
                 enable_GPU=False):
        super().__init__(sample_generator, model_class, likelihood, input_size, output_size, enable_GPU)
        self.model = IndependentMultitaskGPModelApproximate(inducing_points_num=inducing_points,
                                                            input_dim=self.input_size,
                                                            num_tasks=self.output_size)  # Independent
        self.independent = True        
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

    def setup_dataloaders(self,train_dataload,valid_dataload, test_dataloader):
        self.train_loader = train_dataload
        self.valid_loader = valid_dataload
        self.test_loader = test_dataloader


    def pull_samples(self, holdout=150):        
        
        self.train_x = torch.zeros((self.sample_generator.getNumSamples() - holdout, self.input_size))  # [ego_state | tv_state]
        self.test_x = torch.zeros((holdout, self.input_size))  # [ego_state | tv_state]
        self.train_y = torch.zeros([self.sample_generator.getNumSamples() - holdout, self.output_size])  # [tv_actuation]
        self.test_y = torch.zeros([holdout, self.output_size])  # [tv_actuation]

        # Sampling should be done on CPU
        self.train_x = self.train_x.cpu()
        self.test_x = self.test_x.cpu()
        self.train_y = self.train_y.cpu()
        self.test_y = self.test_y.cpu()

        not_done = True
        sample_idx = 0
        while not_done:            
            samp = self.sample_generator.nextSample()
            if samp is not None:                
                samp_input, samp_output = samp
                if sample_idx < holdout:
                    self.test_x[sample_idx] = samp_input
                    self.test_y[sample_idx] = samp_output
                else:
                    self.train_x[sample_idx - holdout] = samp_input
                    self.train_y[sample_idx - holdout] = samp_output
                sample_idx += 1
            else:
                print('Finished')
                not_done = False        
      
        self.means_x = self.train_x.mean(dim=0, keepdim=True)
        self.stds_x = self.train_x.std(dim=0, keepdim=True)
        self.means_y = self.train_y.mean(dim=0, keepdim=True)
        self.stds_y = self.train_y.std(dim=0, keepdim=True)
        
        self.normalize = True
        if self.normalize:
            for i in range(self.stds_x.shape[1]):
                if self.stds_x[0, i] == 0:
                    self.stds_x[0, i] = 1
            self.train_x = (self.train_x - self.means_x) / self.stds_x
            self.test_x = (self.test_x - self.means_x) / self.stds_x

            for i in range(self.stds_y.shape[1]):
                if self.stds_y[0, i] == 0:
                    self.stds_y[0, i] = 1
            self.train_y = (self.train_y - self.means_y) / self.stds_y
            self.test_y = (self.test_y - self.means_y) / self.stds_y
            print(f"train_x shape: {self.train_x.shape}")
            print(f"train_y shape: {self.train_y.shape}")


    def outputToReal(self, output):
        if self.normalize:
            return output

        if self.means_y is not None:
            return output * self.stds_y + self.means_y
        else:
            return output

    def train(self):
        if self.enable_GPU:
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
            self.train_x = self.train_x.cuda()
            self.train_y = self.train_y.cuda()
            self.test_x = self.test_x.cuda()
            self.test_y = self.test_y.cuda()

        # Find optimal model hyper-parameters
        self.model.train()
        self.likelihood.train()

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(self.train_x), torch.tensor(self.train_y)
        )
        valid_dataset = torch.utils.data.TensorDataset(
            torch.tensor(self.test_x), torch.tensor(self.test_y)
        )
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=150 if self.enable_GPU else 100,
                                      shuffle=True,  # shuffle?
                                      num_workers=0 if self.enable_GPU else 8)
        valid_dataloader = DataLoader(valid_dataset,
                                      batch_size=25,
                                      shuffle=False,  # shuffle?
                                      num_workers=0 if self.enable_GPU else 8)

        # Use the Adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=0.005)  # Includes GaussianLikelihood parameters

        # GP marginal log likelihood
        # p(y | x, X, Y) = ∫p(y|x, f)p(f|X, Y)df
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=self.train_y.numel())

        epochs = 100
        last_loss = np.inf
        no_progress_epoch = 0
        not_done = True
        epoch = 0
        best_model = None
        best_likeli = None
        sys.setrecursionlimit(100000)
        while not_done:
        # for _ in range(epochs):
            train_dataloader = tqdm(train_dataloader)
            valid_dataloader = tqdm(valid_dataloader)
            train_loss = 0
            valid_loss = 0
            c_loss = 0
            for step, (train_x, train_y) in enumerate(train_dataloader):
                # Within each iteration, we will go over each minibatch of data
                optimizer.zero_grad()
                output = self.model(train_x)
                loss = -mll(output, train_y)
                train_loss += loss.item()
                train_dataloader.set_postfix(log={'train_loss': f'{(train_loss / (step + 1)):.5f}'})
                loss.backward()
                optimizer.step()
            for step, (train_x, train_y) in enumerate(valid_dataloader):
                optimizer.zero_grad()
                output = self.model(train_x)
                loss = -mll(output, train_y)
                valid_loss += loss.item()
                c_loss = valid_loss / (step + 1)
                valid_dataloader.set_postfix(log={'valid_loss': f'{(c_loss):.5f}'})
            if c_loss > last_loss:
                if no_progress_epoch >= 15:
                    not_done = False
            else:
                best_model = copy.copy(self.model)
                best_likeli = copy.copy(self.likelihood)
                last_loss = c_loss
                no_progress_epoch = 0

            no_progress_epoch += 1
        self.model = best_model
        self.likelihood = best_likeli
    
    def evaluate(self):       
        self.set_evaluation_mode()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # This contains predictions for both outcomes as a list
            predictions = self.likelihood(self.likelihood(self.model(self.test_x[:50])))

        mean = predictions.mean.cpu()
        variance = predictions.variance.cpu()
        self.means_x = self.means_x.cpu()
        self.means_y = self.means_y.cpu()
        self.stds_x = self.stds_x.cpu()
        self.stds_y = self.stds_y.cpu()
        self.test_y = self.test_y.cpu()

        f, ax = plt.subplots(self.output_size, 1, figsize=(15, 10))
        titles = ['vlong', 'vlat', 'wz']
        for i in range(self.output_size):
            unnormalized_mean = self.stds_y[0, i] * mean[:, i] + self.means_y[0, i]
            unnormalized_mean = unnormalized_mean.detach().numpy()
            cov = np.sqrt((variance[:, i] * (self.stds_y[0, i] ** 2)))
            cov = cov.detach().numpy()
            '''lower, upper = prediction.confidence_region()
            lower = lower.detach().numpy()
            upper = upper.detach().numpy()'''
            lower = unnormalized_mean - 2 * cov
            upper = unnormalized_mean + 2 * cov
            tr_y = self.stds_y[0, i] * self.test_y[:50, i] + self.means_y[0, i]
            # Plot training data as black stars
            ax[i].plot(tr_y, 'k*')
            # Predictive mean as blue line
            # ax[i].scatter(np.arange(len(unnormalized_mean)), unnormalized_mean)
            ax[i].errorbar(np.arange(len(unnormalized_mean)), unnormalized_mean, yerr=cov, fmt="o", markersize=4, capsize=8)
            # Shade in confidence
            # ax[i].fill_between(np.arange(len(unnormalized_mean)), lower, upper, alpha=0.5)
            ax[i].legend(['Observed Data', 'Predicted Data'])
            ax[i].set_title(titles[i])
        plt.show()




class ThetaGPTrained(GPController):
    def __init__(self, name, enable_GPU, model=None):
        if model is not None:
            self.load_model_from_object(model)
        else:
            self.load_model(name)
        self.enable_GPU = enable_GPU
        if self.enable_GPU:
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
            self.means_x = self.means_x.cuda()
            self.means_y = self.means_y.cuda()
            self.stds_x = self.stds_x.cuda()
            self.stds_y = self.stds_y.cuda()
        else:
            self.model.cpu()
            self.likelihood.cpu()
            self.means_x = self.means_x.cpu()
            self.means_y = self.means_y.cpu()
            self.stds_x = self.stds_x.cpu()
            self.stds_y = self.stds_y.cpu()
        
        self.dyn_model = DynamicsModelForPredictor() 
############ has to match dim from sampleGeneratorInputGP
        self.input_dim = 12 +4
        self.output_dim = 3   
       
    
       

    def get_true_prediction_par(self,theta_encoder, encoder_input,  ego_state: VehicleState, target_state: VehicleState,
                                ego_prediction: VehiclePrediction, track: RadiusArclengthTrack, M=10):
       
        
        # Set GP model to eval-mode
        self.set_evaluation_mode()
        # draw M samples
        
        preds = self.sample_traj_gp_par(theta_encoder, encoder_input, ego_state, target_state, ego_prediction, track, M)
        
        # numeric mean and covariance calculation.
        # cov_start_time = time.time()
        pred = self.mean_and_cov_from_list(preds, M) 
        # cov_end_time = time.time()
        # cov_elapsed_time = cov_end_time - cov_start_time
        # print(f"COV Elapsed time: {cov_elapsed_time} seconds")    
        pred.t = ego_state.t

        

        return pred


    def sample_traj_gp_par(self,theta_encoder, encoder_input,  ego_state: VehicleState, target_state: VehicleState,
                           ego_prediction: VehiclePrediction, track: RadiusArclengthTrack, M):

        
        prediction_samples = []
        for j in range(M):
            tmp_prediction = VehiclePrediction() 
            tmp_prediction.s = []
            tmp_prediction.x_tran = []
            tmp_prediction.e_psi = []
            tmp_prediction.v_long = []              
            tmp_prediction.v_tran = []
            tmp_prediction.psidot = []      
            prediction_samples.append(tmp_prediction)

            
            
                
        # roll state is ego and tar vehicle dynamics stacked into a big matrix 
        init_tar_state = torch.tensor([target_state.p.s, target_state.p.x_tran, target_state.p.e_psi, target_state.v.v_long, target_state.v.v_tran, target_state.w.w_psi])
        init_tar_curv = torch.tensor([target_state.lookahead.curvature[0], target_state.lookahead.curvature[1] , target_state.lookahead.curvature[2]])
        init_ego_state = torch.tensor([ego_state.p.s, ego_state.p.x_tran, ego_state.p.e_psi, ego_state.v.v_long, ego_state.v.v_tran, ego_state.w.w_psi])
        init_ego_curv = torch.tensor([ego_state.lookahead.curvature[0] ,ego_state.lookahead.curvature[1] ,ego_state.lookahead.curvature[2]])
        
        ## tar (6) + ego (6) + tar_curv(3) + ego_curv(3)
        init_state = torch.hstack([init_tar_state,init_ego_state, init_tar_curv, init_ego_curv])

        roll_state = init_state.repeat(M,1).clone().to(device="cuda")        
        
        for j in range(M):                          # tar 0 1 2 3 4 5       #ego 6 7 8 9 10 11
            prediction_samples[j].s.append(roll_state[j,0].cpu().numpy())
            prediction_samples[j].x_tran.append(roll_state[j,1].cpu().numpy())                    
            prediction_samples[j].e_psi.append(roll_state[j,2].cpu().numpy())
            prediction_samples[j].v_long.append(roll_state[j,3].cpu().numpy())
            prediction_samples[j].v_tran.append(roll_state[j,4].cpu().numpy())
            prediction_samples[j].psidot.append(roll_state[j,5].cpu().numpy())                 

        roll_encoder_input = encoder_input.clone()
        

        horizon = len(ego_prediction.x)       
        stacked_roll_state = roll_state.repeat(horizon,1).to(device="cuda")  
        for i in range(horizon-1):  
            
        ##################################################################################
        ################################## Theta prediction ##############################                  
            # encoder_start_time = time.time()
            if i ==0:
                theta = theta_encoder.get_theta_from_buffer(roll_encoder_input)
                batched_theta = theta.repeat(M,1).to(device="cuda") 
            else:
                if len(roll_encoder_input.shape) < 3:                    
                    roll_encoder_input = roll_encoder_input.repeat(M,1,1)
                roll_encoder_input = self.roll_encoder_input_given_rollstate(roll_encoder_input.clone(),roll_state)
                batched_theta = theta_encoder.get_theta_from_buffer(roll_encoder_input).to(device="cuda")     
            # encoder_end_time = time.time()
            # encoder_elapsed_time = encoder_end_time - encoder_start_time
            # print(f"Encoder Elapsed time: {encoder_elapsed_time} seconds")    
        ####################
            # batched_theta[:] =0                  
        ################################## Theta prediction END ###########################     

        ################################## Target input prediction ##############################          
            # gp_start_time = time.time()
            tmp_state_for_input_prediction = torch.zeros(roll_state.shape[0],self.input_dim).to(device="cuda")
            tmp_state_for_input_prediction[:,0] = roll_state[:,0]-roll_state[:,6]
            tmp_state_for_input_prediction[:,1:6] = roll_state[:,1:6] # target dynamics (ey, epsi, vx, vy, wz)
            tmp_state_for_input_prediction[:,6:9] = roll_state[:,7:10] # ego dynamics (ey, epsi, vx)            
            tmp_state_for_input_prediction[:,9:12] = roll_state[:,12:15] # curvatures (tar_curvs(3))
            tmp_state_for_input_prediction[:,12:] = batched_theta
            
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                               
                prediction = self.model(self.standardize(tmp_state_for_input_prediction))
              
                mx = prediction.mean
                std = prediction.stddev
                
                noise = torch.randn_like(std).to(device="cuda")
                tmp_target_pred = mx + noise * std  
                 
                predicted_target_vel_delta = self.outputToReal(tmp_target_pred)
              
                predicted_target_vel = roll_state[:,3:6] + predicted_target_vel_delta
                
            
            if ego_prediction.u_a is None:
                tmp_ego_input = torch.tensor([0.0, 0.0]).to(device="cuda")    
            else:
                tmp_ego_input = torch.tensor([ego_prediction.u_a[i], ego_prediction.u_steer[i]]).to(device="cuda")
            # gp_end_time = time.time()
            # gp_elapsed_time = gp_end_time - gp_start_time
            # print(f"GP Elapsed time: {gp_elapsed_time} seconds")  
        ################################## Target input prediction END ###########################        
        
        ################################## Vehicle Dynamics Update #################################
            
            GP_Mean_dynamics_update= False
            if GP_Mean_dynamics_update:     
                return   
        ############### Using a Ground truth vehicle update ( only mean input prediction is used to update the model) ###############
                # tmp_tar_state = target_state.copy()           
                # tmp_ego_state = ego_state.copy()                           
                # self.rollstate_to_vehicleState(tmp_tar_state,tmp_ego_state, predicted_target_input,tmp_ego_input, roll_state)                            
                # track.update_curvature(tmp_tar_state)
                # track.update_curvature(tmp_ego_state)   
                # vehicle_dynamics.step(tmp_tar_state)
                # vehicle_dynamics.step(tmp_ego_state)
                # track.update_curvature(tmp_tar_state)
                # track.update_curvature(tmp_ego_state)  
                # self.vehicleState_to_rollstate(tmp_tar_state,tmp_ego_state, predicted_target_input,tmp_ego_input, roll_state)                    
        ############### the distributed inputs are used to update the vehicle dynamics ###############
            else: ## the distributed inputs are used to update the vehicle dynamics 
            ### Using a frenet-based paejeka tire dynamics 
               
                vehicle_simulator = TorchDynamicsModelForPredictor(track)   
                
                stacked_roll_state_for_dynamics_ego = self.roll_state_to_stack_tensor(roll_state)            
                stacked_ego_tar_roll_state = torch.vstack([roll_state[:,0:3], roll_state[:,6:9]])
                stacked_ego_tar_vel = torch.vstack([predicted_target_vel, roll_state[:,9:12]])
                next_x_tar_ego, next_cur_tar_ego=  vehicle_simulator.kinematic_update(stacked_ego_tar_roll_state,stacked_ego_tar_vel)
               
                next_x_tar = next_x_tar_ego[:roll_state.shape[0],:]
                next_x_ego = next_x_tar_ego[roll_state.shape[0]:,:]                
                next_cur_tar = next_cur_tar_ego[0,:roll_state.shape[0]]
                next_cur_ego = next_cur_tar_ego[0,roll_state.shape[0]:]
                # next_x_tar, next_cur_tar=  vehicle_simulator.kinematic_update(roll_state[:,0:3],predicted_target_vel)               
                # next_x_ego, next_cur_ego = vehicle_simulator.dynamics_update(stacked_roll_state_for_dynamics_ego,tmp_ego_input.repeat(M,1).to(device="cuda"))
              
                roll_state = self.stack_tensor_to_roll_state(next_x_tar,next_x_ego,next_cur_tar,next_cur_ego)
             
        ################################## Vehicle Dynamics Update END #################################
            for j in range(M):                                
                prediction_samples[j].s.append(roll_state[j,0].cpu().numpy())
                prediction_samples[j].x_tran.append(roll_state[j,1].cpu().numpy())                    
                prediction_samples[j].e_psi.append(roll_state[j,2].cpu().numpy())
                prediction_samples[j].v_long.append(roll_state[j,3].cpu().numpy())
                prediction_samples[j].v_tran.append(roll_state[j,4].cpu().numpy())
                prediction_samples[j].psidot.append(roll_state[j,5].cpu().numpy())
        ################################## Vehicle Dynamics Update END##############################
            # current_states: ego_s(0), ego_ey(1), ego_epsi(2), ego_vx(3), ego_vy(4), ego_wz(5), 
            #           tar_s(6), tar_ey(7), tar_epsi(8), tar_vx(9), tar_vy(10), tar_wz(11)
            # u(0) = ax_ego, u(1) = delta_ego   
            
            
        for i in range(M):
            prediction_samples[i].s = array.array('d', prediction_samples[i].s)
            prediction_samples[i].x_tran = array.array('d', prediction_samples[i].x_tran)
            prediction_samples[i].e_psi = array.array('d', prediction_samples[i].e_psi)
            prediction_samples[i].v_long = array.array('d', prediction_samples[i].v_long)
            prediction_samples[i].v_tran = array.array('d', prediction_samples[i].v_tran)
            prediction_samples[i].psidot = array.array('d', prediction_samples[i].psidot)            

          

        return prediction_samples



    def roll_state_to_stack_tensor(self,roll_state):
        stacked_roll_state_for_dynamics = torch.zeros(roll_state.shape[0],6).to(device="cuda")
        # target state
        # stacked_roll_state_for_dynamics[:roll_state.shape[0],:] = roll_state[:,0:6]        
        # ego state
        stacked_roll_state_for_dynamics = roll_state[:,6:12]        
        return stacked_roll_state_for_dynamics


    def stack_tensor_to_roll_state(self,next_x_tar,next_x_ego,next_cur_tar,next_cur_ego):
        roll_state = torch.zeros(next_x_tar.shape[0],18).to(device="cuda")
        # target state
        roll_state[:,0:6] = next_x_tar
        # ego state
        roll_state[:,6:12] = next_x_ego
        # tar_curvs
        roll_state[:,12] = next_cur_tar#.squeeze()
        roll_state[:,13] = next_cur_tar#.squeeze()
        roll_state[:,14] = next_cur_tar#.squeeze()
        roll_state[:,15] = next_cur_ego#.squeeze()
        roll_state[:,16] = next_cur_ego#.squeeze()
        roll_state[:,17] = next_cur_ego#.squeeze()
        
        # roll_state[:,12] = next_cur_tar[0].squeeze()
        # roll_state[:,13] = next_cur_tar[1].squeeze()
        # roll_state[:,14] = next_cur_tar[2].squeeze()
        # roll_state[:,15] = next_cur_ego[0].squeeze()
        # roll_state[:,16] = next_cur_ego[1].squeeze()
        # roll_state[:,17] = next_cur_ego[2].squeeze()
        return roll_state

    def rollstate_to_vehicleState(self,tar_state,ego_state,tar_input, ego_input,roll_state):
        tar_state.p.s = np.mean(roll_state[:,0].cpu().numpy())
        tar_state.p.x_tran = np.mean(roll_state[:,1].cpu().numpy())
        tar_state.p.e_psi = np.mean(roll_state[:,2].cpu().numpy())
        tar_state.v.v_long = np.mean(roll_state[:,3].cpu().numpy())
        tar_state.v.v_tran = np.mean(roll_state[:,4].cpu().numpy())
        tar_state.w.w_psi = np.mean(roll_state[:,5].cpu().numpy())       
        # tar_state.lookahead.curvature[0] =  np.mean(roll_state[:,5].cpu().numpy())     
 
        ego_state.p.s = np.mean(roll_state[:,6].cpu().numpy())
        ego_state.p.x_tran = np.mean(roll_state[:,7].cpu().numpy())
        ego_state.p.e_psi = np.mean(roll_state[:,8].cpu().numpy())
        ego_state.v.v_long = np.mean(roll_state[:,9].cpu().numpy())
        ego_state.v.v_tran = np.mean(roll_state[:,10].cpu().numpy())
        ego_state.w.w_psi = np.mean(roll_state[:,11].cpu().numpy())            
   

    def vehicleState_to_rollstate(self,tar_state,ego_state,tar_input, ego_input,roll_state):
        roll_state[:,0] = tar_state.p.s
        roll_state[:,1] = tar_state.p.x_tran
        roll_state[:,2] = tar_state.p.e_psi
        roll_state[:,3] = tar_state.v.v_long
        roll_state[:,4] = tar_state.v.v_tran
        roll_state[:,5] = tar_state.w.w_psi
        roll_state[:,6] = ego_state.p.s
        roll_state[:,7] = ego_state.p.x_tran
        roll_state[:,8] = ego_state.p.e_psi
        roll_state[:,9] = ego_state.v.v_long
        roll_state[:,10] = ego_state.v.v_tran
        roll_state[:,11] = ego_state.w.w_psi
        roll_state[:,12] = tar_state.lookahead.curvature[0]
        roll_state[:,13] = tar_state.lookahead.curvature[1]
        roll_state[:,14] = tar_state.lookahead.curvature[2]
        roll_state[:,15] = ego_state.lookahead.curvature[0]
        roll_state[:,16] = ego_state.lookahead.curvature[1]
        roll_state[:,17] = ego_state.lookahead.curvature[2]
    
    def roll_encoder_input_given_rollstate(self,encoder_input,rollstate):
        # [batch, sequence, #features]
        encoder_input_tmp = encoder_input.clone()
        encoder_input_tmp[:,0:-1,:] = encoder_input[:,1:,:]
        encoder_input_tmp[:,-1,0] = rollstate[:,0] - rollstate[:,6] # tar_s.p.s - ego_s.p.s 
        encoder_input_tmp[:,-1,1:4] = rollstate[:,1:4]  # tar_st.p.x_tran, tar_st.p.e_psi, tar_st.v.v_long
        encoder_input_tmp[:,-1,4] = rollstate[:,12]  # tar_st.lookahead.curvature[0],
        encoder_input_tmp[:,-1,5:8] = rollstate[:,7:10] # ego_st.p.x_tran, ego_st.p.e_psi,ego_st.v.v_long,                       
        encoder_input_tmp[:,-1,8] = rollstate[:,15] # ego_st.lookahead.curvature[0]
        return encoder_input_tmp
    
    ## TODO: This should be matched with the below funciton 
    # states_to_encoder_input_torch
 

        
