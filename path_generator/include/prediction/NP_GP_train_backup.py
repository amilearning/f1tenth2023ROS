import secrets
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import os 
from hmcgp.h2h_configs import *
from hmcgp.common.utils.file_utils import *
from matplotlib import pyplot as plt
from torch.distributions import Normal, MultivariateNormal
from torch.utils.data import DataLoader
from VAEModel import LSTMVAE
from VAEtrain import VAEtrain
from torch.utils.tensorboard import SummaryWriter


import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import GaussianSymmetrizedKLKernel, ScaleKernel, InducingPointKernel
from gpytorch.means import ConstantMean

from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

Distributional_GP = True
LatentPolicy = True 
std_for_xdata = 0.05
class GPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        if Distributional_GP:
            self.covar_module = gpytorch.kernels.ScaleKernel(GaussianSymmetrizedKLKernel())
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


        

class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(GaussianSymmetrizedKLKernel())


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



# class ExactGPModel(ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = ConstantMean()
#         self.base_covar_module = ScaleKernel(GaussianSymmetrizedKLKernel())
#         self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=train_x[:500, :], likelihood=likelihood)

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)




VAEwriter = SummaryWriter()

class RacingGP():
    def __init__(self, gpmodel = GPModel):
        self.gpmodel = gpmodel 


        data_index = "trainx"
        index_number = 0
        self.state_model = "partial" 

    def dataloading(NN_model_name = None):                        
        # pickle_data = pickle_read(os.path.join(racingdata_dir, data_index+ '_'+str(index_number) + '_'+ state_model + '.pkl'))
        np_data = np.asarray(pickle_read(os.path.join(racingdata_dir, NN_model_name)))
        ###########################################################################################################################################
        # full state [s_ego, epsi_ego, etran_ego, vlon_ego, vlat_ego, s_tar, epsi_tar, etran_tar, vlon_tar, vlat_tar, lookahead[0,1,2], tar_accel, tar_delta, tar_Qxref, tar_Qtheta] #                        
        # partial state [s_ego, etran_ego, s_tar, etran_tar, lookahead[0,1,2], tar_accel, tar_delta, tar_Qxref, tar_Qtheta] #                        
        ###########################################################################################################################################
        normalizing = True 
        if normalizing: 
            if self.state_model == 'fullstate':
                clip_mins = np.array([-10, -np.pi, -1*width,    -2.0,       -1.0,     -np.pi,  -1*width,  -2.0,     -1.0,      -1, -1, -1,     -3 , -0.5 , 0, 0])
                clip_maxs = -1*clip_mins
                cliped_np_data = np.clip(np_data[:,0:14], clip_mins, clip_maxs)
            else: # partial state 
                np_data[:,:,0] = np_data[:,:,2] - np_data[:,:,0] # --> (s_tar - s_ego)
                np_data = np_data[:,:,[0,1,3,4,5,6,7,8,9,10]] # remove s_tar 
                # np_data = np_data[:,:,[0,1,3,4,7,8,9,10]] # remove s_tar k

                # partial state [(s_tar - s_ego), etran_ego, etran_tar, lookahead[0,1,2], tar_accel, tar_delta, tar_Qxref, tar_Qtheta] #                        
                clip_mins = np.array([-10, -1*width, -1*width, -1, -1, -1, -3, -0.5 ])
                # clip_mins = np.array([-10, -1*width, -1*width, -1, -3, -0.5 ])
                clip_maxs = -1*clip_mins
                # cliped_np_data = np.clip(np_data[:,:,0:6], np.tile(clip_mins,[np_data.shape[1],1]), np.tile(clip_maxs,[np_data.shape[1],1]))
                cliped_np_data = np.clip(np_data[:,:,0:8], np.tile(clip_mins,[np_data.shape[1],1]), np.tile(clip_maxs,[np_data.shape[1],1]))
            normalized_data = (cliped_np_data/clip_maxs)
            dataset = torch.Tensor(normalized_data)
        else:
            if state_model == 'fullstate':
                dataset = torch.Tensor(np_data[:,0:14])
            else: # partial state 
                np_data[:,:,0] = np_data[:,:,2] - np_data[:,:,0] # --> (s_tar - s_ego)        
                np_data = np_data[:,:,[0,1,3,4,7,8,9,10]] # remove s_tar k
                np_data[:,:,0:6]
                dataset = torch.Tensor(np_data[:,:,0:6])


dataset_size = len(dataset)
train_size = int(dataset_size * 0.9)
test_size = dataset_size - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

print(f"Training Data Size : {len(train_dataset)}")
print(f"Testing Data Size : {len(test_dataset)}")

args =  {
            "batch_size": 512*5,
            "device": torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu"),
            "input_size": train_dataset[0].shape[1],
            "hidden_size": train_dataset[0].shape[1],
            "latent_size": 2,
            "learning_rate": 0.0001,
            "max_iter": 10000,
        }
    
    
train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False)

if LatentPolicy:
    model_name = 'lstmvaef67deb1a.model'
    model_to_load = LSTMVAE(args)
    model_to_load.to(torch.device("cuda"))
    model_to_load.load_state_dict(torch.load(model_dir+model_name))
else:
    model_to_load = None


def batch_data_from_dataloader(dataloader,model_to_load):    
    train_x_distributionals = None
    for count, test_data_batch in enumerate(dataloader):  
         # get distribution of latent variable theta(Multinomial)
        
        # get x label for GP input distribution 
        state_data = test_data_batch[:,-1,:].to(torch.device("cuda"))    
        state_data_mean = state_data[:,0:6]

        if LatentPolicy:
            model_to_load.eval()
            with torch.no_grad():
                _, recon_data, info, theta_mean_, theta_logvar_ = model_to_load(test_data_batch.to(torch.device("cuda")))
            x_mean = torch.hstack([theta_mean_,state_data_mean])
        else:
            x_mean = state_data_mean
        
        # torch.exp(var*.5) = std 
        if Distributional_GP:
            state_data_std = torch.mul(torch.ones(state_data_mean.shape),2*torch.log(torch.ones(1)*std_for_xdata)).to(torch.device("cuda"))        
            x_var =  torch.hstack([theta_logvar_,state_data_std])            
            train_x_distributional = torch.hstack((x_mean, x_var)).to(torch.device("cuda"))   
        else:
            train_x_distributional = x_mean.to(torch.device("cuda"))
        # get y label for GP output   
        y_data = torch.squeeze(state_data[:,6:]).to(torch.device("cuda"))
        

        if count == 0 :
            train_x_distributionals = torch.unsqueeze(train_x_distributional,dim=0)
            y_datas = torch.unsqueeze(y_data,dim=0)
        elif count < len(dataloader)-1: 
            train_x_distributional = torch.unsqueeze(train_x_distributional,dim=0)
            train_x_distributionals = torch.cat([train_x_distributionals,train_x_distributional],dim=0) 
            y_data = torch.unsqueeze(y_data,dim=0)
            y_datas = torch.cat([y_datas,y_data],dim=0) 
 
    return train_x_distributionals, y_datas


def one_data_from_dataloader(dataloader,model_to_load):    
    
    for count, test_data_batch in enumerate(dataloader):  
         # get distribution of latent variable theta(Multinomial)
        # model_to_load.eval()
        # with torch.no_grad():
        #     _, recon_data, info, theta_mean_, theta_logvar_ = model_to_load(test_data_batch.to(torch.device("cuda")))
        # # get x label for GP input distribution 
        # state_data = test_data_batch[:,-1,:].to(torch.device("cuda"))    
        # state_data_mean = state_data[:,0:6]
        # # torch.exp(var*.5) = std 
        # state_data_std = torch.mul(torch.ones(state_data_mean.shape),2*torch.log(torch.ones(1)*1e-2)).to(torch.device("cuda"))        
        # x_mean = torch.hstack([theta_mean_,state_data_mean])
        # x_var =  torch.hstack([theta_logvar_,state_data_std])
        # train_x_distributional = torch.hstack((x_mean, x_var)).to(torch.device("cuda"))   
        # # get y label for GP output   
        # y_data = torch.squeeze(state_data[:,6:]).to(torch.device("cuda"))

        state_data = test_data_batch[:,-1,:].to(torch.device("cuda"))    
        state_data_mean = state_data[:,0:6]

        if LatentPolicy:
            model_to_load.eval()
            with torch.no_grad():
                _, recon_data, info, theta_mean_, theta_logvar_ = model_to_load(test_data_batch.to(torch.device("cuda")))
            x_mean = torch.hstack([theta_mean_,state_data_mean])
        else:
            x_mean = state_data_mean
        
        # torch.exp(var*.5) = std 
        if Distributional_GP:
            state_data_std = torch.mul(torch.ones(state_data_mean.shape),2*torch.log(torch.ones(1)*std_for_xdata)).to(torch.device("cuda"))        
            x_var =  torch.hstack([theta_logvar_,state_data_std])            
            train_x_distributional = torch.hstack((x_mean, x_var)).to(torch.device("cuda"))   
        else:
            train_x_distributional = x_mean.to(torch.device("cuda"))
        # get y label for GP output   
        y_data = torch.squeeze(state_data[:,4:]).to(torch.device("cuda"))
        
        if count == 0 :
            train_x_distributionals = train_x_distributional
            y_datas = y_data
        else: 
            train_x_distributionals = torch.vstack([train_x_distributionals,train_x_distributional])
            y_datas = torch.vstack([y_datas,y_data])
 
    return train_x_distributionals, y_datas


def data_from_dataloader(dataloader,model_to_load):    
    train_x_distributionals = []
    y_datas = []
    for count, test_data_batch in enumerate(dataloader):  
         # get distribution of latent variable theta(Multinomial)
        # model_to_load.eval()
        # with torch.no_grad():
        #     _, recon_data, info, theta_mean_, theta_logvar_ = model_to_load(test_data_batch.to(torch.device("cuda")))
        # # get x label for GP input distribution 
        # state_data = test_data_batch[:,-1,:].to(torch.device("cuda"))    
        # state_data_mean = state_data[:,0:6]
        # # torch.exp(var*.5) = std 
        # state_data_std = torch.mul(torch.ones(state_data_mean.shape),2*torch.log(torch.ones(1)*1e-2)).to(torch.device("cuda"))        
        # x_mean = torch.hstack([theta_mean_,state_data_mean])
        # x_var =  torch.hstack([theta_logvar_,state_data_std])
        # train_x_distributional = torch.hstack((x_mean, x_var)).to(torch.device("cuda"))   
        # # get y label for GP output   
        # y_data = torch.squeeze(state_data[:,6:]).to(torch.device("cuda"))
        state_data = test_data_batch[:,-1,:].to(torch.device("cuda"))    
        state_data_mean = state_data[:,0:6]

        if LatentPolicy:
            model_to_load.eval()
            with torch.no_grad():
                _, recon_data, info, theta_mean_, theta_logvar_ = model_to_load(test_data_batch.to(torch.device("cuda")))
            x_mean = torch.hstack([theta_mean_,state_data_mean])
        else:
            x_mean = state_data_mean
        
        # torch.exp(var*.5) = std 
        if Distributional_GP:
            state_data_std = torch.mul(torch.ones(state_data_mean.shape),2*torch.log(torch.ones(1)*std_for_xdata)).to(torch.device("cuda"))        
            x_var =  torch.hstack([theta_logvar_,state_data_std])            
            train_x_distributional = torch.hstack((x_mean, x_var)).to(torch.device("cuda"))   
        else:
            train_x_distributional = x_mean.to(torch.device("cuda"))
        # get y label for GP output   
        y_data = torch.squeeze(state_data[:,4:]).to(torch.device("cuda"))
        

        train_x_distributionals.append(train_x_distributional)
        y_datas.append(y_data)        
    return train_x_distributionals, y_datas

train_x_distributionals, y_datas = data_from_dataloader(train_dataloader,model_to_load)
one_train_x_distributionals, one_y_datas = one_data_from_dataloader(train_dataloader,model_to_load)
batch_train_x_distributionals, batch_y_datas = batch_data_from_dataloader(train_dataloader,model_to_load)
test_x_distributionals, test_y_datas = data_from_dataloader(test_dataloader,model_to_load)

#################################################################################
############################### Train GP for y_accel ############################
#################################################################################





# one_train_x_distributionals = one_train_x_distributionals.contiguous()
# one_train_x_distributionals[:,3] = one_train_x_distributionals[:,3] + 1e-5
one_train_x_distributionals = one_train_x_distributionals.to(torch.device("cuda"))   
y_accel = torch.squeeze(one_y_datas[:,0])
# y_accel = y_accel.contiguous()
y_accel  = y_accel.to(torch.device("cuda"))   
inducing_points = one_train_x_distributionals[:500, :]
model_accel = GPModel(inducing_points=inducing_points).to(torch.device("cuda"))   
likelihood_accel = gpytorch.likelihoods.GaussianLikelihood().to(torch.device("cuda"))   

# y_accel = batch_y_datas[:,:,0]
# model_accel = ExactGPModel(batch_train_x_distributionals, y_accel, likelihood_accel).to(torch.device("cuda"))        
# model_accel = ExactGPModel(train_x_distributionals[0], torch.squeeze(y_datas[0][:,0]), likelihood_accel).to(torch.device("cuda"))        
# model_accel = ExactGPModel( likelihood_accel).to(torch.device("cuda"))        
# model_accel = ExactGPModel(train_x_distributionals[0], torch.squeeze(y_datas[0][:,0]), likelihood_accel).to(torch.device("cuda"))        


model_accel.train()
likelihood_accel.train()
optimizer_accel = torch.optim.Adam(model_accel.parameters(), lr=0.05)  # Includes GaussianLikelihood parameters
# mll_accel = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_accel, model_accel)
mll_accel = gpytorch.mlls.VariationalELBO(likelihood_accel, model_accel, num_data=y_accel.size(0))

n_epoch = 1
training_iter = len(train_x_distributionals)*n_epoch
batch_count = 0
# with gpytorch.settings.cholesky_jitter(1e-1):
for i in range(training_iter):
    if batch_count > len(train_x_distributionals)-1:
        batch_count = 0
    # batch_count = 0
    x_batch = train_x_distributionals[batch_count]
    # x_batch[:,3] = x_batch[:,3]+ torch.rand(x_batch[:,3].shape).to(torch.device("cuda"))   
    y_batch = y_datas[batch_count][:,0]
    # Zero gradients from previous iteration
    optimizer_accel.zero_grad()
    # Output from model    
    output = model_accel(x_batch)
    # Calc loss and backprop gradients
    loss = -mll_accel(output, y_batch)
    loss.backward()
    print('Iter %d/%d - Loss: %.5f' % (
        i + 1, training_iter, loss.item()
    ))
    # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.6f' % (
    #     i + 1, training_iter, loss.item(),
    #     model_accel.covar_module.base_kernel.lengthscale.item(),
    #     model_accel.likelihood.noise.item()
    # ))
    optimizer_accel.step()
    batch_count+=1
##########################################################



# model_accel.set_train_data(one_train_x_distributionals, torch.squeeze(one_y_datas[:,0]), strict=False)
# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood

# test_x_distributionals, test_y_datas

model_accel.eval()
likelihood_accel.eval()
# x_test = train_x_distributionals[0] # test_x_distributionals[0]
# y_test = y_datas[0][:,0] # test_y_datas[0][:,0].cpu().numpy()
x_test = test_x_distributionals[0] 
y_test = test_y_datas[0][:,0]
with torch.no_grad(), gpytorch.settings.fast_pred_var():    
    observed_pred = likelihood_accel(model_accel(x_test))
with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(8, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot predictive means as blue line
    x_axis = np.linspace(1,len(observed_pred.mean),len(observed_pred.mean))
    ax.plot(x_axis, observed_pred.mean.cpu().numpy(), 'b')
   
    # Shade between the lower and upper confidence bounds
    
    
    ax.plot(x_axis, y_test.cpu().numpy(), 'g')
    ax.fill_between(x_axis, lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.5)
    
    # ax.set_ylim([-3, 3])
    ax.legend(['GP Mean', 'Ground Truth', 'Cov'])

mseloss_for_testing = torch.nn.MSELoss()
output = mseloss_for_testing(observed_pred.mean, y_test)
#################################################################################
############################### Train GP for y_delta ############################
#################################################################################

likelihood_delta = gpytorch.likelihoods.GaussianLikelihood().to(torch.device("cuda"))   
model_delta = ExactGPModel(train_x_distributional, y_delta, likelihood_delta).to(torch.device("cuda"))        
model_delta.train()
likelihood_delta.train()
optimizer_delta = torch.optim.Adam(model_delta.parameters(), lr=0.05)  # Includes GaussianLikelihood parameters
mll_delta = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_delta, model_delta)
training_iter = 100
for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer_delta.zero_grad()
    # Output from model
    output = model_delta(train_x_distributional)
    # Calc loss and backprop gradients
    loss = -mll_delta(output, y_delta)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model_delta.covar_module.base_kernel.lengthscale.item(),
        model_delta.likelihood.noise.item()
    ))
    optimizer_delta.step()



#################################################################################
# prior_theta_mean_vector = torch.zeros(2)
# prior_theta_cov_mtx = torch.eye(2)*1e6
# theta_dist = MultivariateNormal(prior_theta_mean_vector,prior_theta_cov_mtx)


    
    



