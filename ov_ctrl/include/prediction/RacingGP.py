import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import GaussianSymmetrizedKLKernel, ScaleKernel, InducingPointKernel
from gpytorch.means import ConstantMean

from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
import numpy as np
import torch
from prediction.GPModels import AppxGPModel, ExactGPModel
from prediction.VAEModel import LSTMVAE
import os
import yaml
import rospkg
rospack = rospkg.RosPack()
pkg_dir = rospack.get_path('ov_ctrl')
with torch.no_grad() and torch.cuda.amp.autocast():

    class IndependentMultitaskGPModelApproximate(gpytorch.models.ApproximateGP):
        def __init__(self, inducing_points_num, input_dim, num_tasks):
            # Let's use a different set of inducing points for each task
            inducing_points = torch.rand(num_tasks, inducing_points_num, input_dim)

            # We have to mark the CholeskyVariationalDistribution as batch
            # so that we learn a variational distribution for each task
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                inducing_points.size(-2), batch_shape=torch.Size([num_tasks]) )

            variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
                gpytorch.variational.VariationalStrategy( self, inducing_points, variational_distribution, learn_inducing_locations=True),
                num_tasks=num_tasks,)

            super().__init__(variational_strategy)

            # The mean and covariance modules should be marked as batch
            # so we learn a different set of hyperparameters
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_tasks], ard_num_dims=input_dim)),
                batch_shape=torch.Size([num_tasks]) )
            
            # self.covar_module = gpytorch.kernels.ScaleKernel(
                # gpytorch.kernels.MaternKernel(nu=1.5, batch_shape=torch.Size([num_tasks])),
                # batch_shape=torch.Size([num_tasks]) )

        def forward(self, x):
            # The forward function should be written as if we were dealing with each output
            # dimension in batch
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        




    class RacingGP():
        def __init__(self):
            
            with open(pkg_dir+"/include/configs/simulation_config.yaml", "r") as f:
                params = yaml.safe_load(f)
    
            self.Distributional_GP = True
            self.train_dataset = None
            self.test_dataset = None                
            
            self.batchsize = params["batch_size"] # 512
            self.std_for_xdata = 0.05
            if torch.cuda.is_available():
                self.device = params["torch_device"] 
            else:
                self.device = torch.device("cpu")

            self.vae_model_dir = params["saved_vae_model_dir"]
            self.track_width = params["track_width"]

            # FOR VAE 
            self.vae_model_file_name = params["vae_model_file_name"] # 0.05
            self.input_size_vae = params["input_size_for_vae"]
            self.hidden_size_vae_lstm = params["hidden_size_for_vae_lstm"]
            self.latent_size_vae = params["latent_size_for_vae_lstm"]
            self.lr_vae = params["learning_rate_vae"] # 0.05
            self.s_diff_clip_min     = params["s_diff_clip_min"] 
            self.ey_clip_min    = params["ey_clip_min"] 
            self.epsi_clip_min  = -torch.pi/2 # -90 degree 
            self.curvature_clip_min  = -1.0
            # Clip target : [ (s_ego-s_tar), ey_ego, epsi_ego, curvature_ego, ey_tar, epsi_tar, curvature_tar] --> horizon x dim(7)
            self.state_clip_min = torch.tensor([self.s_diff_clip_min, self.ey_clip_min, self.epsi_clip_min, self.curvature_clip_min, self.ey_clip_min, self.epsi_clip_min, self.curvature_clip_min]).to(device=self.device)
            self.state_clip_max = -1*self.state_clip_min
            self.gpmodel = None
            self.init_vae()

            # FOR GP 
            self.lr_gp = params["learning_rate_gp"] # 0.05
            self.input_dim_gp = params["input_dim_gp"]
            self.n_inducing = params["num_inducing"]
            self.gp_model_dir = params["saved_gp_model_dir"] 
            self.gp_model_file_name = params["gp_model_file_name"] 
            self.gp_liklihood_file_name = params["gp_liklihood_file_name"]             
            self.gp_model_load = params["gp_model_load"]
            self.batch_for_gp = params["mppi_n_sample"] # for each forward
            self.init_gp()
            
            
            




        def init_vae(self):
            # 
            vae_args =  {
                            "batch_size": 1,
                            "device": self.device,
                            "input_size": self.input_size_vae,
                            "hidden_size": self.hidden_size_vae_lstm,
                            "latent_size": self.latent_size_vae,
                            "learning_rate": self.lr_vae,
                            "max_iter": 10000,
                        }       
            self.VAEModel = LSTMVAE(vae_args)
            self.VAEModel.to(torch.device("cuda"))        
            self.VAEModel.load_state_dict(torch.load(os.path.join(self.vae_model_dir, self.vae_model_file_name)))
            self.VAEModel.eval()
            print("NN Model succesfully loaded")


        def init_gp(self):
            self.gpmodel = IndependentMultitaskGPModelApproximate(inducing_points_num=self.n_inducing, input_dim=self.input_dim_gp, num_tasks=2).to(device=self.device)       # Independent
            self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2).to(device=self.device)
            self.optimizer = torch.optim.Adam([
                {'params': self.gpmodel.parameters()},
                {'params': self.likelihood.parameters()},
            ], lr=0.005)    # Includes GaussianLikelihood parameters
            # GP marginal log likelihood
            # p(y | x, X, Y) = ∫p(y|x, f)p(f|X, Y)df
            self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.gpmodel, num_data=self.batch_for_gp*2).to(device=self.device)

            if self.gp_model_load:            
                self.model_statedict = torch.load(os.path.join(self.gp_model_dir, self.gp_model_file_name))
                self.gpmodel.load_state_dict(self.model_statedict)
                self.liklihood_state_dict = torch.load(os.path.join(self.gp_model_dir, self.gp_liklihood_file_name))
                self.likelihood.load_state_dict(self.liklihood_state_dict)
            
            self.gpmodel.eval()
            self.likelihood.eval()    


        def state_update(self,state):
            # data preprocess here        
            
            # save data for training 
            return
        
        
        def state_preprocess(self,ego_state_history,tar_state_history):        
            # ego_state_history : horizon x state_dim, # tar_state_history : horizon x state_dim 
            # state_dim : (s, ey, epsi, curvature)
            ego_ = torch.tensor(ego_state_history).to(device=self.device)
            tar_ = torch.tensor(tar_state_history).to(device=self.device)        
            state = torch.vstack([(ego_[:,0] - tar_[:,0]), ego_[:,1:], tar_[:,1:]]).to(device=self.device)
            # clip and normalize state data
            cliped_state  =torch.clip(state, torch.tile(self.state_clip_min,[state.shape[1],1]) , torch.tile(self.state_clip_max,[state.shape[1],1]))
            normalized_state = cliped_state / self.state_clip_max
            # normalized state : [ (s_ego-s_tar), ey_ego, epsi_ego, ey_tar, epsi_tar, curvature_ego, curvature_tar] --> horizon x dim(6~)
            return normalized_state
        
        def vaeforward(self, vae_x):        
            # x : states history of ego and target 
            # [(tar_s-ego_s), ego_ey, ego_epsi, ego_cur, tar_ey, tar_epsi, tar_cur] X [Horizon(5steps)]
            if len(vae_x.shape) == 2:
                vae_x = vae_x.view(1,vae_x.shape[0],vae_x.shape[1])
            
            theta_mean_, theta_logvar_ = self.VAEModel.get_theta_dist(vae_x.to(dtype=torch.float))
            
            
            return theta_mean_, theta_logvar_ 

        def gpforward(self,gp_x):            
            # input for gp [(tar_s-ego_s), ego_ey, ego_epsi, ego_cur, tar_ey, tar_epsi, tar_cur, theta_dim1, theta_dim2] 
            if self.gpmodel is None:
                return 
          
            with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_computations():    
                gp_pred_mean = self.likelihood(self.gpmodel(gp_x.to(dtype=torch.float))).mean
            return gp_pred_mean
