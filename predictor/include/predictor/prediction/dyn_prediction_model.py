from predictor.common.utils.scenario_utils import *
#!/usr/bin python3
import warnings
from typing import List


import numpy as np

import casadi as ca
import array
import sys, os, pathlib

sys.path.append(os.path.join(os.path.expanduser('~'), 'forces_pro_client'))
import forcespro
import forcespro.nlp
import time
from predictor.common.utils.file_utils import *
import torch 
from predictor.prediction.torch_utils import get_curvature_from_keypts_torch, wrap_to_pi, wrap_to_pi_torch 
import yaml


with torch.no_grad() and torch.cuda.amp.autocast():
    class TorchDynamicsModelForPredictor:
        def __init__(self,track, device_ = "cuda", dt = 0.1):                       
            self.m = 2.366
            self.width = 0.2            
            self.Lr = 0.13
            self.Lf = 0.13
            self.L = self.Lr+self.Lf
            self.Caf = self.m * self.Lf/self.L * 0.5 * 0.35 * 180/torch.pi
            self.Car = self.m * self.Lr/self.L * 0.5 * 0.35 * 180/torch.pi
            self.Izz = 0.018 # self.Lf*self.Lr*self.m
            self.g= 9.814195
            self.h = 0.15            
            self.dt = dt             
            self.torch_device = device_                        
            self.track = track            
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

        def compute_slip_p(self,x,u):
            clip_vx = torch.max(torch.hstack((x[:,3].view(-1,1),torch.ones(len(x[:,3])).to(device=self.torch_device).view(-1,1))),dim =1).values   
            alpha_f = u[:,1] - torch.arctan2( x[:,4]+self.Lf*x[:,5], clip_vx )
            alpha_r = - torch.arctan2(x[:,4]-self.Lr*x[:,5] , clip_vx)
            return alpha_f,alpha_r
        
        def dynamics_update(self,x,u):     
            # x: s(0), ey(1), epsi(2), vx(3), vy(4), wz(5) , u(0) = ax, u(1) = delta        
            if not torch.is_tensor(x):
                x = torch.tensor(x).to(device=self.torch_device)    
            if not torch.is_tensor(u):
                u = torch.tensor(u).to(device=self.torch_device)    
            x[:,2] = wrap_to_pi_torch(x[:,2])
            nx = torch.clone(x).to(device=self.torch_device)  
                        
            # roll = torch.zeros(len(x[:,2])).to(device=self.torch_device)           # roll = 0
            # pitch = torch.zeros(len(x[:,2])).to(device=self.torch_device)          # pitch = 0
                
            axb = u[:,0]
            delta = u[:,1]
            ################  Pejekap 
            alpha_f_p, alpha_r_p = self.compute_slip_p(x,u)
            Fyf = self.Df*torch.sin(self.Cp*torch.arctan(self.Bp*alpha_f_p))
            Fyr = self.Dr*torch.sin(self.Cp*torch.arctan(self.Bp*alpha_r_p))
            # x: s(0), ey(1), epsi(2), vx(3), vy(4), wz(5)        
            curs = get_curvature_from_keypts_torch(x[:,0].clone().detach(),self.track)
            # torch_get_cuvature(x[:,0],self.centerline_frenet)
            
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
             
            
            next_curs = get_curvature_from_keypts_torch(nx[:,0].clone().detach(),self.track)
            # next_curs = []  
            # for i in range(int(3)):
            #     tmp = get_curvature_from_keypts_torch(nx[:,0].clone().detach()+0.5*i,self.track)
            #     next_curs.append(tmp)
                


            return nx, next_curs

        def kinematic_update(self,x,v):     
          
            # x: s(0), ey(1), epsi(2), 
            # v: vx(0), vy(1), wz(2)
            if not torch.is_tensor(x):
                x = torch.tensor(x).to(device=self.torch_device)    
            if not torch.is_tensor(v):
                v = torch.tensor(v).to(device=self.torch_device)    
            x[:,2] = wrap_to_pi_torch(x[:,2])
            nx = torch.clone(x).to(device=self.torch_device)  
            
            # roll = torch.zeros(len(x[:,2])).to(device=self.torch_device)           # roll = 0
            # pitch = torch.zeros(len(x[:,2])).to(device=self.torch_device)          # pitch = 0
            # x: s(0), ey(1), epsi(2), vx(3), vy(4), wz(5)        
            
            curs = get_curvature_from_keypts_torch(x[:,0].clone().detach(),self.track)
            
            # torch_get_cuvature(x[:,0],self.centerline_frenet)
            
            s = x[:,0]
            ey = x[:,1]        
            epsi = x[:,2]         
            vx = v[:,0]
            vy = v[:,1]
            wz = v[:,2]      
            
            
            
            nx[:,0] = s + self.dt * ( (vx * torch.cos(epsi) - vy * torch.sin(epsi)) / (1 - curs * ey) )
            nx[:,1] = ey + self.dt * (vx * torch.sin(epsi) + vy * torch.cos(epsi))
            nx[:,2] = epsi + self.dt * ( wz - (vx * torch.cos(epsi) - vy * torch.sin(epsi)) / (1 - curs * ey) * curs )
            
            next_curs = get_curvature_from_keypts_torch(nx[:,0].clone().detach(),self.track)
            
            # next_curs = []  
            # for i in range(int(3)):
            #     tmp = get_curvature_from_keypts_torch(nx[:,0].clone().detach()+0.5*i,self.track)
            #     next_curs.append(tmp)
            
          
            
            vehicle_full_dyn = torch.hstack([nx,v]).to(device=self.torch_device)    
          
            
           
            return vehicle_full_dyn, next_curs


        

class DynamicsModelForPredictor():
    def __init__(self):
        self.dt = 0.1                    
        self.Lr = 0.13
        self.Lf = 0.13
        self.m = 2.366
        self.L = self.Lr+self.Lf
        self.Caf = self.m * self.Lf/self.L * 0.5 * 0.35 * 180/np.pi
        self.Car = self.m * self.Lr/self.L * 0.5 * 0.35 * 180/np.pi
        self.Izz = 0.018  # self.Lf*self.Lr*self.m
        self.g= 9.814195
        self.h = 0.15

        self.Bp = 1.0 # 1.0
        self.Cp = 1.25 # 1.25   
        self.wheel_friction = 0.8 # 0.8            
        self.Df = self.wheel_friction*self.m*self.g * self.Lr / (self.Lr + self.Lf)        
        self.Dr = self.wheel_friction*self.m*self.g * self.Lf / (self.Lr + self.Lf)      


    # one step prediction only .. as the curvature is copied from the current state 
    def DynamicsUpdate(self,tar_state,action):

        next_state = tar_state.copy()     
        x = np.array([tar_state.p.s,tar_state.p.x_tran,tar_state.p.e_psi,tar_state.v.v_long,tar_state.v.v_tran,tar_state.w.w_psi])
        curs = np.array(tar_state.lookahead.curvature[0])
        u = np.array([action[0],action[1]])
        # x: s(0), ey(1), epsi(2), vx(3), vy(4), wz(5) , u(0) = ax, u(1) = delta                
        
        # x[:,2] = wrap_to_pi(x[:,2])
        while x[2] > np.pi-0.01:
            x[2] -= 2.0 * np.pi
        while x[2] < -np.pi+0.01:
            x[2] += 2.0 * np.pi

        nx = x.copy()
 

        
        ################  Pejekap 
        clip_vx = np.max([x[3],1.0])
        alpha_f_p = u[1] - np.arctan2(x[4]+self.Lf*x[5], clip_vx)
        
        alpha_r_p = - np.arctan2(x[4]-self.Lr*x[5] , clip_vx)
        
        Fyf = self.Df*np.sin(self.Cp*np.arctan(self.Bp*alpha_f_p))
        Fyr = self.Dr*np.sin(self.Cp*np.arctan(self.Bp*alpha_r_p))

        axb = u[0]
        delta = u[1]

        s = x[0]
        ey = x[1]        
        epsi = x[2]         
        vx = x[3]
        vy = x[4]
        wz = x[5]        
        
        nx[0] = s + self.dt * ( (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - curs * ey) )
        nx[1] = ey + self.dt * (vx * np.sin(epsi) + vy * np.cos(epsi))
        nx[2] = epsi + self.dt * ( wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - curs * ey) * curs )
        nx[3] = vx + self.dt * (axb - 1 / self.m * Fyf * np.sin(delta) + wz*vy)
        nx[3] = np.max([nx[3],0.0])
        nx[4] = vy + self.dt * (1 / self.m * (Fyf * np.cos(delta) + Fyr) - wz * vx)
        nx[5] = wz + self.dt * (1 / self.Izz *(self.Lf * Fyf * np.cos(delta) - self.Lf * Fyr) )

        next_state.p.s = nx[0]
        next_state.p.x_tran = nx[1]
        next_state.p.e_psi = nx[2]
        next_state.v.v_long = nx[3]
        next_state.v.v_tran = nx[4]
        next_state.w.w_psi = nx[5]
        return next_state
        

        