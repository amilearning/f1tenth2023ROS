from this import s
import numpy as np
from regex import P
import sys
import os
import rospkg
import math

import time 
import torch
# from ov_ctrl.mppi import MPPI
from ov_ctrl.vehicle_mppi import VehicleMPPI
from hmcgp.common.pytypes import VehicleState
# from matplotlib import pyplot as plt
# from numpy import linalg as la
from torch.distributions.multivariate_normal import MultivariateNormal
from ov_ctrl.utils import torch_get_cuvature, b_to_g_rot, wrap_to_pi, wrap_to_pi_torch, torch_unwrap_s, torch_wrap_s,torch_fren_to_global
from prediction.RacingGP import RacingGP
rospack = rospkg.RosPack()
pkg_dir = rospack.get_path('ov_ctrl')
import yaml



with torch.no_grad() and torch.cuda.amp.autocast():
    
    class VehicleModel:
        def __init__(self, device_ = "cuda", dt = 0.1, N_node = 10, mppi_n_sample = 10, centerline_fren = None,centerline_global = None):           
            with open(pkg_dir+"/include/configs/simulation_config.yaml", "r") as f:
                params = yaml.safe_load(f)
            self.opponent_dist_max = params["opponent_dist_max"]
            self.obstacle_safe_distance = params["obstacle_safe_distance"]
            self.m = 25
            self.width = 0.25
            self.L = 0.9
            self.Lr = 0.45
            self.Lf = 0.45
            self.Caf = self.m * self.Lf/self.L * 0.5 * 0.35 * 180/torch.pi
            self.Car = self.m * self.Lr/self.L * 0.5 * 0.35 * 180/torch.pi
            self.Izz = self.Lf*self.Lr*self.m
            self.g= 9.814195
            self.h = 0.15
            self.N = N_node
            self.dt = dt 
            self.horizon = self.dt* N_node
            self.mppi_n_sample = mppi_n_sample
            self.torch_device = device_
            self.dyn_params = torch.zeros(9).to(device = self.torch_device)
            self.tar_cur_fren = None # torch.zeros(4).to(device = self.torch_device)
            self.centerline_global = centerline_global
            self.centerline_frenet = centerline_fren
            if not torch.is_tensor(self.centerline_frenet):
                self.centerline_frenet = torch.tensor(self.centerline_frenet).to(device=self.torch_device)
            self.curs = torch.zeros(self.mppi_n_sample*2).to(device=self.torch_device) # ego + target 
    ###############################################################################
            self.Bp = 1.0
            self.Cp = 1.25   
            self.wheel_friction = 0.8            
            self.Df = self.wheel_friction*self.m*self.g * self.Lr / (self.Lr + self.Lf)        
            self.Dr = self.wheel_friction*self.m*self.g * self.Lf / (self.Lr + self.Lf)        
    ###############################################################################

            # Q = cost weight [ey, epsi]       
            # state x : s(0), ey(1), epsi(2), vx(3), vy(4), wz(5)
            self.state_dim = 6
            # input u: accel_x(0), delta(1) 
            self.input_dim = 2
            self.u_min = torch.tensor([-3.5,-25*torch.pi/180.0]).to(device= self.torch_device)
            self.u_max = self.u_min*-1
            steering_sigma = 15*3.14195/180.0
            self.noise_sigma = torch.diag(torch.tensor([2.0,steering_sigma]),0)
            self.vehiclemppi = VehicleMPPI(self.dynamics_update, self.running_cost, self.state_dim, self.noise_sigma, num_samples=self.mppi_n_sample, horizon=self.N,
                            lambda_=1, u_min = self.u_min, u_max = self.u_max, u_per_command = 10)
        
        def update_param(self,dyn):
            self.dyn_params = dyn

        def command(self,x,tar_x):
            self.vehiclemppi.dyn_params = self.dyn_params        
            self.tar_cur_fren = tar_x
            return self.vehiclemppi.command(x)
            
        def compute_slip_p(self,x,u):
            clip_vx = torch.max(torch.hstack((x[:,3].view(-1,1),torch.ones(len(x[:,3])).to(device=self.torch_device).view(-1,1))),dim =1).values   
            alpha_f = u[:,1] - torch.arctan2( x[:,4]+self.Lf*x[:,5], clip_vx )
            alpha_r = - torch.arctan2(x[:,4]-self.Lr*x[:,5] , clip_vx)
            return alpha_f,alpha_r
        
        def dynamics_update(self,x,u,t):     
            # x: s(0), ey(1), epsi(2), vx(3), vy(4), wz(5) , u(0) = ax, u(1) = delta        
            if not torch.is_tensor(x):
                x = torch.tensor(x).to(device=self.torch_device)    
            if not torch.is_tensor(u):
                u = torch.tensor(u).to(device=self.torch_device)    
            x[:,2] = wrap_to_pi_torch(x[:,2])
            nx = torch.clone(x).to(device=self.torch_device)  
                        
            roll = torch.zeros(len(x[:,2])).to(device=self.torch_device)           # roll = 0
            pitch = torch.zeros(len(x[:,2])).to(device=self.torch_device)          # pitch = 0
                
            axb = u[:,0]
            delta = u[:,1]   
            ################  Pejekap 
            alpha_f_p, alpha_r_p = self.compute_slip_p(x,u)
            Fyf = self.Df*torch.sin(self.Cp*torch.arctan(self.Bp*alpha_f_p))
            Fyr = self.Dr*torch.sin(self.Cp*torch.arctan(self.Bp*alpha_r_p))
            # x: s(0), ey(1), epsi(2), vx(3), vy(4), wz(5)        
            curs = torch_get_cuvature(x[:,0],self.centerline_frenet)
            if len(curs) == self.mppi_n_sample*2:
                self.curs = curs
            s = x[:,0]
            ey = x[:,1]        
            epsi = x[:,2]         
            vx = x[:,3]
            vy = x[:,4]
            wz = x[:,5]        
            
            nx[:,0] = s + self.dt * ( (vx * torch.cos(epsi) - vy * torch.sin(epsi)) / (1 - curs * ey) )
            nx[:,1] = ey + self.dt * (vx * torch.sin(epsi) + vy * torch.cos(epsi))
            nx[:,2] = epsi + self.dt * ( wz - (vx * torch.cos(epsi) - vy * torch.sin(epsi)) / (1 - curs * ey) * curs )
            nx[:,3] = vx + self.dt * (axb - 1 / self.m * Fyf * torch.sin(delta) + wz*vy)
            nx[:,3] = torch.max(torch.hstack((nx[:,3].view(-1,1),torch.zeros(len(x[:,3])).to(device=self.torch_device).view(-1,1))),dim =1).values           
            nx[:,4] = vy + self.dt * (1 / self.m * (Fyf * torch.cos(delta) + Fyr) - wz * vx)
            nx[:,5] = wz + self.dt * (1 / self.Izz *(self.Lf * Fyf * torch.cos(delta) - self.Lf * Fyr) )

            return nx

        def running_cost(self, state, action, t, prev_state,prev_action):
            # state vector (t) --> s(0), ey(1), epsi(2), vx(3), vy(4), wz(5)  
            # self.target_path --> [x, y, psi, horizon(t)]
            cost = torch.zeros(state.shape[0]).to(device=self.torch_device)        
            if t!= 0:
                delta_steering = torch.abs(action-prev_action)[:,1]            
                cost = torch.where(delta_steering >= 150.0*torch.pi/180.0, cost+delta_steering*1.0,cost)
                
            # s, e_lat, e_psi
            s = state[:,0]
            ey = state[:,1]
            epsi = state[:,2] 
            epsi = wrap_to_pi_torch(epsi)
            vx = state[:,3] 
            vy = state[:,4]
            wz = state[:,5]
            
            if self.tar_cur_fren is not None:
                self.tar_cur_fren[0] = torch_wrap_s(torch.tensor(self.tar_cur_fren[0]).view(-1,1).to(device=self.torch_device),self.centerline_frenet[-1,0])
            if torch.abs(torch.min(s)- torch.max(s)) > 30.0 or torch.min(s) >= self.centerline_frenet[-1,0]-10: # cross the finish line
                s_ref = (torch.max(s)+torch.min(s))/2.0
                s = torch_unwrap_s(s,s_ref,self.centerline_frenet[-1,0])
                ############################## Obstacle constarint ########################
                if self.tar_cur_fren is not None and self.tar_cur_fren[0] < s_ref:                
                    self.tar_cur_fren[0]+=self.centerline_frenet[-1,0]
                    ############################## Obstacle constarint ########################
                    # ego_x_global = fren_to_global(ego_x_frenet, self.centerline_global,self.centerline_fren)            
            
            if self.tar_cur_fren is not None:
                ########## Global opponent filter ########################
                # ego_x,ego_y,ego_psi = torch_fren_to_global(state, self.centerline_global,self.centerline_frenet)
                # tar_x,tar_y,tar_psi = torch_fren_to_global(torch.tensor(self.tar_cur_fren).repeat(2,1).to(device=self.torch_device), self.centerline_global,self.centerline_frenet)
                # dist_to_target = torch.sqrt((ego_x-tar_x[0])**2+(ego_y-tar_y[0])**2)
                
                # cost[dist_to_target_idx] += (1.0-dist_to_target[dist_to_target_idx])*self.dyn_params[9]
                ####### frenet opponent filter #################
                dist_s_to_opp = self.tar_cur_fren[0]-s
                dist_s_to_opp_ = torch_wrap_s(dist_s_to_opp,self.centerline_frenet[-1,0])
                dist_ey_to_opp = torch.abs(self.tar_cur_fren[1]-ey)            
                
                cost = torch.where((dist_ey_to_opp < self.obstacle_safe_distance)&(dist_s_to_opp_ < self.opponent_dist_max), cost+(self.obstacle_safe_distance-dist_ey_to_opp)*self.dyn_params[9],cost)
                
                # cost_dist_ey_to_opp = torch.clip(dist_ey_to_opp[opponent_activate_idx],0,self.obstacle_safe_distance)            
                # if len(opponent_activate_idx)> 0:
                #     cost[opponent_activate_idx] += (self.obstacle_safe_distance-dist_ey_to_opp[opponent_activate_idx])*self.dyn_params[9]
                ####### frenet opponent filter End #################

        
      
            
            # minus velocity constraint
               
            # rollover constraint
            accy = (state[:,4]-prev_state[:,4])/self.dt
            rollover_idx = accy / (self.width * self.g) * (2*self.h)
            rollover_idx_cost = torch.where(rollover_idx < 0.3, 0.0,rollover_idx)                
            # cost for states
            s_cost = s*self.dyn_params[0]
            ey_cost = torch.abs(ey)*self.dyn_params[1]
            epsi_cost = torch.abs(epsi)*self.dyn_params[2]
            vx_cost = vx*self.dyn_params[3]
            vy_cost = torch.abs(vy)*self.dyn_params[4]
            wz_cost = torch.abs(wz)*self.dyn_params[5]                   
            

            # speed limit 
            cost = torch.where(vx >= 7.5, cost+1e1,cost)            
            cost = torch.where(vx <= 0.0, cost+1e3,cost)            
            # track boundary constraint
            cost = torch.where(torch.abs(ey) >= 2.0, cost+1e9,cost)
            # state cost 
            cost +=  s_cost + ey_cost + epsi_cost + vx_cost +vy_cost +wz_cost +rollover_idx_cost*self.dyn_params[6]
            return cost



    class RaceModel:
        def __init__(self, predictor =RacingGP, device_ = "cuda", dt = 0.1, N_node = 10, mppi_n_sample = 10, centerline_fren = None,centerline_global = None):      
            
            with open(pkg_dir+"/include/configs/simulation_config.yaml", "r") as f:
                params = yaml.safe_load(f)
            self.torch_device = device_ 
            self.centerline_global = centerline_global
            self.centerline_frenet = centerline_fren
            self.opponent_dist_max = params["opponent_dist_max"]
            self.obstacle_safe_distance = params["obstacle_safe_distance"]
            if not torch.is_tensor(self.centerline_frenet):
                self.centerline_frenet = torch.tensor(self.centerline_frenet).to(device=self.torch_device)
            
            self.N = N_node
            self.dt = dt 
            self.mppi_n_sample = mppi_n_sample
            
            self.dyn_params = None
            self.state_dim = 12
            self.input_dim = 2    
            self.horizon = self.dt* N_node
                    
            self.theta_mean = None
            self.theta_var = None

            self.u_min = torch.tensor([-3.5,-25*torch.pi/180.0]).to(device= self.torch_device)
            self.u_max = self.u_min*-1
            steering_sigma = 15*3.14195/180.0
            self.noise_sigma = torch.diag(torch.tensor([2.0,steering_sigma]),0).to(device= self.torch_device)

            self.gt_input_prediction = params["gt_input_prediction"]
            self.pred_tar_u_gt = None
            self.fake_noise_sigma = torch.diag(torch.tensor([3,0.4]),0).to(device= self.torch_device)
            self.fake_noise_mu = torch.zeros(2).to(device= self.torch_device)
            self.face_noise_dist = MultivariateNormal(self.fake_noise_mu, covariance_matrix=self.fake_noise_sigma)
            self.fake_input_noise = params["fake_input_noise"]

            ##############################################
            self.predictor = predictor
            self.vehicle_model = VehicleModel(device_ = self.torch_device, dt = self.dt, N_node = self.N, mppi_n_sample = self.mppi_n_sample, centerline_fren = self.centerline_frenet,centerline_global = self.centerline_global)
            ##############################################
            # state vector X(t) --> ego_s(0), ego_ey(1), ego_epsi(2), ego_vlon(3), ego_vlat(4), ego_wz(5), tar_x(6), tar_y(7), tar_psi(8), tar_vlong(9), tar_vlat(10), tar_wz(11)
            self.race_mppi = VehicleMPPI(self.race_dynamics_update, self.race_running_cost, self.state_dim, self.noise_sigma, num_samples=self.mppi_n_sample, horizon=self.N,
                            lambda_=1, u_min = self.u_min, u_max = self.u_max, u_per_command = 10)
            
        def update_param(self,dyn_params):
            self.dyn_params = dyn_params
            self.vehicle_model.update_param(dyn_params)

        def command(self,x,pred_tar_u_gt):        
            self.race_mppi.dyn_params = self.dyn_params  
            if self.gt_input_prediction:
                self.pred_tar_u_gt = pred_tar_u_gt
            else:
                self.pred_tar_u_gt = None
            return self.race_mppi.command(x)

        def predictor_theta_update(self,race_history):
            theta_mean_, theta_logvar_  = self.predictor.vaeforward(race_history)            
            self.theta_mean = theta_mean_
            self.theta_var = torch.exp(theta_logvar_)
            
            
        
        def frens2gpinput(self,x):
            ego_cur = self.vehicle_model.curs[:self.mppi_n_sample]          
            tar_cur = self.vehicle_model.curs[self.mppi_n_sample:]
            theta= torch.ones(len(x),2).to(device=self.torch_device)*self.theta_mean            
            gp_input = torch.vstack([(x[:,6]-x[:,0]),x[:,1],x[:,2],ego_cur, x[:,7], x[:,8], tar_cur, theta[:,0],theta[:,1]]).to(device=self.torch_device)            
            return torch.transpose(gp_input,0,1)

         
        
        def race_running_cost(self,state, action, t, prev_state,prev_action):    
            # state x : ego_s(0), ego_ey(1), ego_epsi(2), ego_vx(3), ego_vy(4), ego_wz(5), 
            #           tar_s(6), tar_ey(7), tar_epsi(8), tar_vx(9), tar_vy(10), tar_wz(11)
            cost = torch.zeros(state.shape[0]).to(device=self.torch_device)   
            if t!= 0:
                delta_steering = torch.abs(action-prev_action)[:,1]            
                cost = torch.where(delta_steering >= 150.0*torch.pi/180.0, cost+delta_steering*1.0, cost)
            # ego states 
            s = state[:,0]
            ey = state[:,1]
            epsi = state[:,2] 
            epsi = wrap_to_pi_torch(epsi)
            vx = state[:,3] 
            vy = state[:,4]
            wz = state[:,5]

            # target states 
            tar_s = state[:,6]
            tar_ey = state[:,7]
            tar_epsi = state[:,8] 
            tar_epsi = wrap_to_pi_torch(tar_epsi)        
            
            s = torch_wrap_s(s,self.centerline_frenet[-1,0])
            tar_s = torch_wrap_s(tar_s,self.centerline_frenet[-1,0])

            if torch.abs(torch.min(s)- torch.max(s)) > 30.0 or torch.min(s) >= self.centerline_frenet[-1,0]-10: # cross the finish line             
                s_ref = (torch.max(s)+torch.min(s))/2.0            
                s = torch_unwrap_s(s,s_ref,torch.max(s))           
                tar_s = torch_unwrap_s(tar_s,s_ref,torch.max(s))           
                
            ############################## Obstacle constarint ########################                        
            dist_s_to_opp = tar_s-s        
            dist_s_to_opp_ = torch_wrap_s(dist_s_to_opp,self.centerline_frenet[-1,0])
            dist_ey_to_opp = torch.abs(tar_ey-ey)                                
            cost = torch.where((dist_ey_to_opp < self.obstacle_safe_distance)&(dist_s_to_opp_ < self.opponent_dist_max), cost+(self.obstacle_safe_distance-dist_ey_to_opp)*self.dyn_params[9],cost)
            ########################### Obstacle constarint END #######################                        
            
            # track boundary constraint
                
            # rollover constraint
            accy = (state[:,4]-prev_state[:,4])/self.dt
            rollover_idx = accy / (self.vehicle_model.width * self.vehicle_model.g) * (2*self.vehicle_model.h)
            rollover_cost = torch.where(rollover_idx < 0.3, 0.0,rollover_idx)*self.dyn_params[6]                
            # cost for states
            s_cost = dist_s_to_opp*self.dyn_params[0]
            ey_cost = torch.abs(ey)*self.dyn_params[1]
            epsi_cost = torch.abs(epsi)*self.dyn_params[2]
            vx_cost = vx*self.dyn_params[3]
            vy_cost = torch.abs(vy)*self.dyn_params[4]
            wz_cost = torch.abs(wz)*self.dyn_params[5]
        
           
            
            cost = torch.where(vx >= 7.5, cost+1e1,cost)            
            cost = torch.where(vx <= 0.0, cost+1e3,cost)  

            cost = torch.where(torch.abs(ey) >= 2.0, cost+1e9, cost)  # torch.exp(ey[track_cross_ey_idx]).to(device=self.torch_device)
            cost +=s_cost + ey_cost + epsi_cost + vx_cost + vy_cost + wz_cost + rollover_cost

            return cost
            
        
        def race_dynamics_update(self, x, u, t):     
            # state x : ego_s(0), ego_ey(1), ego_epsi(2), ego_vx(3), ego_vy(4), ego_wz(5), 
            #           tar_s(6), tar_ey(7), tar_epsi(8), tar_vx(9), tar_vy(10), tar_wz(11)
            # u(0) = ax_ego, u(1) = delta_ego
            if not torch.is_tensor(x):
                x = torch.tensor(x).to(device=self.torch_device)    
            if not torch.is_tensor(u):
                u = torch.tensor(u).to(device=self.torch_device)    

            ego_x = x[:,0:6]
            tar_x = x[:,6:]   
            ego_u = u[:,0:2]  
            if self.gt_input_prediction:
                if not torch.is_tensor(self.pred_tar_u_gt):
                    self.pred_tar_u_gt = torch.tensor(self.pred_tar_u_gt).to(device=self.torch_device)    
                tar_u = self.pred_tar_u_gt[t,:].repeat(ego_u.shape[0],1)
                if self.fake_input_noise:
                    tar_u_noise = self.noise_dist.sample([tar_u.shape[0]]).to(device=self.torch_device)            
                    tar_u += tar_u_noise
            else:
                # target vehicle's input is unkwon 
                gp_input = self.frens2gpinput(x).to(dtype=torch.float)
                # start = time.perf_counter()
                tar_u = self.predictor.gpforward(gp_input)
                # elapsed_time = (time.perf_counter() - start) * 1000
                # print("gp loop elapsed time = " + str(elapsed_time))
                # tar_u = ego_u
                
            
                
            ego_tar_x = torch.vstack([ego_x,tar_x])
            ego_tar_u = torch.vstack([ego_u,tar_u])

            ego_tar_nx = self.vehicle_model.dynamics_update(ego_tar_x, ego_tar_u,t)
            
            nx = torch.clone(x).to(device=self.torch_device)  
            nx[:,0:6] = ego_tar_nx[:ego_x.shape[0],:]
            nx[:,6:] = ego_tar_nx[ego_x.shape[0]:,:]
            
            return nx




