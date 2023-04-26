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


class GPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, Distributional_GP):
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
    def __init__(self):
        self.gpmodel = None
        self.LatentPolicy = True
        self.Distributional_GP = False

        data_index = "trainx"
        index_number = 0
        self.state_model = "fullstate" 
        self.train_dataset = None
        self.test_dataset = None
        
        self.model_to_load = None
        self.n_inducing = 500
        self.batchsize = 512 # 512
        self.std_for_xdata = 0.05
        self.lr = 0.05
        self.converge_count_thres = 50
        self.gpmodel_load = False
        if self.gpmodel_load:
            self.accel_model_statedict = torch.load(os.path.join(model_dir, 'model_accel_1.pth'))
            self.accel_liklihood_state_dict = torch.load(os.path.join(model_dir, 'model_accel_1.pth'))
        
    def clipData(self,np_data):
        normalizing = True 
        # full state [s_ego, epsi_ego, etran_ego, vlon_ego, vlat_ego, s_tar, epsi_tar, etran_tar, vlon_tar, vlat_tar, lookahead[0,1,2], tar_accel, tar_delta, tar_Qxref, tar_Qtheta] #                        
        # Input for VAE -> [(tar_s-ego_s), ego_ey, ego_epsi, ego_cur, tar_ey, tar_epsi, tar_cur] 
        torch_data = torch.tensor(np_data)
        cliped_data = torch.zeros([torch_data.shape[0],torch_data.shape[1],7+2])
        cliped_data[:,:,0] = torch_data[:,:,5] - torch_data[:,:,0] 
        cliped_data[:,:,1] = torch_data[:,:,2]
        cliped_data[:,:,2] = torch_data[:,:,1]
        cliped_data[:,:,3] = torch_data[:,:,10]
        cliped_data[:,:,4] = torch_data[:,:,7]
        cliped_data[:,:,5] = torch_data[:,:,6]  # fake input 
        cliped_data[:,:,6] = torch_data[:,:,10]  # fake input         
        cliped_data[:,:,7] = torch_data[:,:,13]  # accel  
        cliped_data[:,:,8] = torch_data[:,:,14]  # delta
        # y_out = torch.zeros([torch_data.shape[0],2])
        # y_out[:,0] = torch_data[:,-1,13]
        # y_out[:,1] = torch_data[:,-1,14]

        # if normalizing: 
        #     np_data[:,:,0] = np_data[:,:,2] - np_data[:,:,0] # --> (s_tar - s_ego)
        #     np_data = np_data[:,:,[0,1,3,4,5,6,7,8,9,10]] # remove s_tar             
        #     # partial state [(s_tar - s_ego), etran_ego, etran_tar, lookahead[0,1,2], tar_accel, tar_delta, tar_Qxref, tar_Qtheta] #                        
        #     clip_mins = np.array([-10, -1*width, -1*width, -1, -1, -1, -3, -0.5 ])            
        #     clip_maxs = -1*clip_mins            
        #     cliped_np_data = np.clip(np_data[:,:,0:8], np.tile(clip_mins,[np_data.shape[1],1]), np.tile(clip_maxs,[np_data.shape[1],1]))
        #     dataset = torch.Tensor(cliped_np_data/clip_maxs)            
        # else:
        #     np_data[:,:,0] = np_data[:,:,2] - np_data[:,:,0] # --> (s_tar - s_ego)        
        #     np_data = np_data[:,:,[0,1,3,4,5,6,7,8,9,10]] # remove s_tar k            
        #     dataset = torch.Tensor(np_data[:,:,0:8])

        
        return cliped_data 

    def dataloading(self, datafile_name = None):                        
        # pickle_data = pickle_read(os.path.join(racingdata_dir, data_index+ '_'+str(index_number) + '_'+ state_model + '.pkl'))
        np_data = np.asarray(pickle_read(os.path.join(racingdata_dir, datafile_name)))
        dataset = self.clipData(np_data)
        ###########################################################################################################################################
        # full state [s_ego, epsi_ego, etran_ego, vlon_ego, vlat_ego, s_tar, epsi_tar, etran_tar, vlon_tar, vlat_tar, lookahead[0,1,2], tar_accel, tar_delta, tar_Qxref, tar_Qtheta] #                        
        # partial state [s_ego, etran_ego, s_tar, etran_tar, lookahead[0,1,2], tar_accel, tar_delta, tar_Qxref, tar_Qtheta] #                        
        ###########################################################################################################################################        
        dataset_size = len(dataset)
        train_size = int(dataset_size * 0.9)
        test_size = dataset_size - train_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        print(f"Training Data Size : {len(self.train_dataset)}")
        print(f"Testing Data Size : {len( self.test_dataset)}")
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batchsize, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batchsize, shuffle=False)    

    def NNmodelLoad(self,args = None, model_name = 'lstmvaeddb19c35.model'):
        if self.train_dataset is None:
            print("Please load data first--> returning")
            return 
        input_size_for_vae = 7
        hidden_size_for_vae_lstm = 7
        latent_size_for_vae_lstm =  2
        learning_rate_vae = 0.001

        if args is None:
            print("Default args is activated")
            args =  {
                            "batch_size": self.batchsize,
                            "device": torch.device("cuda")
                            if torch.cuda.is_available()
                            else torch.device("cpu"),
                            "input_size": input_size_for_vae,
                            "hidden_size": hidden_size_for_vae_lstm,
                            "latent_size": latent_size_for_vae_lstm,
                            "learning_rate": learning_rate_vae,
                            "max_iter": 10000,
                        }               
        self.model_to_load = LSTMVAE(args)
        self.model_to_load.to(torch.device("cuda"))
        saved_vae_model_dir = '/home/hjpc/research/overtaking_ws/src/ov_ctrl/include/models/vae/'
        self.model_to_load.load_state_dict(torch.load(os.path.join(saved_vae_model_dir, model_name)))
        # self.model_to_load.load_state_dict(torch.load(model_dir+model_name))
        print("NN Model succesfully loaded")

            
    
    def dataloader_preprocess(self,dataloader = None):        
        if dataloader is None:
            dataloader = self.train_dataloader
        batch_x_data = None
        batch_y_data = None                
        for count, data_batch in enumerate(dataloader):              
            # get x label for GP input distribution 
            state_data = data_batch[:,-1,:].to(torch.device("cuda"))    
            batch_data_nn = data_batch[:,:,0:7].to(torch.device("cuda"))    
            state_data_mean = state_data[:,0:7]
        
            if self.model_to_load is None:
                print("load model first!")
                return None, None
            self.model_to_load.eval()
            with torch.no_grad():
                _, recon_data, info, theta_mean_, theta_logvar_ = self.model_to_load(batch_data_nn)
            x_mean = torch.hstack([theta_mean_,state_data_mean])
                    
            # torch.exp(var*.5) = std 
            if self.Distributional_GP:
                state_data_std = torch.mul(torch.ones(state_data_mean.shape),2*torch.log(torch.ones(1)*self.std_for_xdata)).to(torch.device("cuda"))        
                x_var =  torch.hstack([theta_logvar_,state_data_std])            
                train_x_distributional = torch.hstack((x_mean, x_var)).to(torch.device("cuda"))   
            else:
                train_x_distributional = x_mean.to(torch.device("cuda"))
            # get y label for GP output   
            y_data = torch.squeeze(state_data[:,7:]).to(torch.device("cuda"))            

            if count == 0 :
                batch_x_data = torch.unsqueeze(train_x_distributional,dim=0)
                batch_y_data = torch.unsqueeze(y_data,dim=0)
            elif count < len(dataloader)-1: 
                batch_x_data = torch.cat([batch_x_data,torch.unsqueeze(train_x_distributional,dim=0)],dim=0)                 
                batch_y_data = torch.cat([batch_y_data,torch.unsqueeze(y_data,dim=0)],dim=0)     
        
        return batch_x_data, batch_y_data 

        
    def train_test_data_preprocess(self):        
        
        batch_x_data, batch_y_data = self.dataloader_preprocess(self.train_dataloader)

        onerow_x_data = batch_x_data.view([batch_x_data.shape[0]*batch_x_data.shape[1],batch_x_data.shape[2]])
        onerow_y_data = batch_y_data.view([batch_y_data.shape[0]*batch_y_data.shape[1],batch_y_data.shape[2]])
        self.batch_x_data= batch_x_data.to(torch.device("cuda"))   
        self.batch_y_data= batch_y_data.to(torch.device("cuda"))   
        self.onerow_x_data= onerow_x_data.to(torch.device("cuda"))   
        self.onerow_y_data= onerow_y_data.to(torch.device("cuda"))   
        self.y_accel = torch.squeeze(self.onerow_y_data[:,0]).to(torch.device("cuda"))  
        self.y_delta = torch.squeeze(self.onerow_y_data[:,1]).to(torch.device("cuda"))  

        test_batch_x_data, test_batch_y_data = self.dataloader_preprocess(self.test_dataloader)        
        self.test_batch_x_data= test_batch_x_data.to(torch.device("cuda"))   
        self.test_batch_y_data= test_batch_y_data.to(torch.device("cuda"))   
        self.test_onerow_x_data= test_batch_x_data.view([test_batch_x_data.shape[0]*test_batch_x_data.shape[1],test_batch_x_data.shape[2]]).to(torch.device("cuda"))   
        self.test_onerow_y_data= test_batch_y_data.view([test_batch_y_data.shape[0]*test_batch_y_data.shape[1],test_batch_y_data.shape[2]]).to(torch.device("cuda"))   
        self.test_y_accel = torch.squeeze(self.test_onerow_y_data[:,0]).to(torch.device("cuda"))  
        self.test_y_delta = torch.squeeze(self.test_onerow_y_data[:,1]).to(torch.device("cuda"))  


    def setupGPModel(self,model_load = False):
        self.model_accel = IndependentMultitaskGPModelApproximate(inducing_points_num=50, input_dim=9, num_tasks=2).to(torch.device("cuda"))       # Independent
        self.likelihood_accel = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2).to(torch.device("cuda"))     

        self.optimizer_accel = torch.optim.Adam([
            {'params': self.model_accel.parameters()},
            {'params': self.likelihood_accel.parameters()},
        ], lr=0.005)    # Includes GaussianLikelihood parameters

        # GP marginal log likelihood
        # p(y | x, X, Y) = ∫p(y|x, f)p(f|X, Y)df
        self.mll_accel = gpytorch.mlls.VariationalELBO(self.likelihood_accel, self.model_accel, num_data=self.test_batch_y_data[-1,:,:].numel()).to(torch.device("cuda"))     



        # self.model_accel = GPModel(inducing_points=self.onerow_x_data[:self.n_inducing,:], Distributional_GP = self.Distributional_GP).to(torch.device("cuda"))   
        # self.likelihood_accel = gpytorch.likelihoods.GaussianLikelihood().to(torch.device("cuda"))           
        # self.optimizer_accel = torch.optim.Adam([{'params': self.model_accel.parameters()},{'params': self.likelihood_accel.parameters()},], lr=0.01)
        # self.mll_accel = gpytorch.mlls.VariationalELBO(self.likelihood_accel, self.model_accel, num_data=self.y_accel.size(0))

        

        # if model_load:
        #     self.model_accel.load_state_dict(self.accel_model_statedict)        
        #     self.likelihood_accel.load_state_dict(self.accel_liklihood_state_dict)

    
    def GPtrain(self,n_epoch):                        
        # self.model_accel.train() 
        # self.likelihood_accel.train()       
        training_iter = len(self.batch_x_data)*n_epoch
        batch_count = 0
        prev_val_loss = 1e10
        converge_count = 0
        for i in range(training_iter):
            self.model_accel.train()
            self.likelihood_accel.train()  
            if batch_count > len(self.batch_x_data)-1:
                batch_count = 0    
            self.optimizer_accel.zero_grad()    
            # with gpytorch.settings.cholesky_jitter(1e-1):        
            # output = self.model_accel(self.batch_x_data[batch_count])
            # rand_x = self.batch_x_data[batch_count]+torch.randn(self.batch_x_data[batch_count].shape).to(torch.device("cuda"))*1e-1
            output = self.model_accel(self.batch_x_data[batch_count])
            
            # Calc loss and backprop gradients
            y_accel_batch = self.batch_y_data[batch_count]
            loss = -self.mll_accel(output, y_accel_batch).mean()
            loss.backward()
            print('Iter %d/%d - Loss: %.5f' % (i + 1, training_iter, loss.item()))
            self.optimizer_accel.step()
            batch_count+=1
            

            self.model_accel.eval()
            self.likelihood_accel.eval()
            self.optimizer_accel.zero_grad()   
            # test_batch = next(iter(self.test_dataloader))
            # x_test =  test_batch[:,:,0:6]
            # y_test =  test_batch[:,:,6:]
            # plt.clf()
            # mse_val_loss = self.draw_output(x_test,y_test)
            # print(mse_val_loss)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():    
                mse_val_loss = self.validation()
                print('validation loss = ' + str(mse_val_loss))
            ### Save and Load model 
            if prev_val_loss < mse_val_loss:                                 
                converge_count = converge_count+1 
            if converge_count > self.converge_count_thres: 
                model_accel_file_name = os.path.join(model_dir, 'race_gp_model_'+str(i) +'.pth')                
                likelihood_accel_file_name = os.path.join(model_dir, 'race_gp_liklihood_'+str(i) +'.pth')
                torch.save(self.model_accel.state_dict(), model_accel_file_name)
                torch.save(self.likelihood_accel.state_dict(), likelihood_accel_file_name)
                print("Loss converged --> Model save and return from training process")                
                break
            
            prev_val_loss = mse_val_loss
            self.optimizer_accel.zero_grad() 

        
    def validation(self):
        loss_array=np.array([])
        for i in range(self.test_batch_x_data.shape[0]):
            x_test = self.test_batch_x_data[i,:,:]
            y_test = self.test_batch_y_data[i,:,:]
            acc_observed_pred = self.likelihood_accel(self.model_accel(x_test))
            mseloss_for_testing = torch.nn.MSELoss()
            mse_val_loss = mseloss_for_testing(acc_observed_pred.mean, y_test.squeeze())
            loss_array = np.hstack([loss_array,mse_val_loss.cpu().numpy()])
        return np.mean(loss_array)
            


    def directforward(self,x):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():    
            acc_observed_pred = self.likelihood_accel(self.model_accel(x))
        return acc_observed_pred

    def forward(self,x):        
    
        # x = self.clipData(x_)
        state_data_mean = x[:,-1,:].to(torch.device("cuda"))                
                       
        if self.LatentPolicy:            
            if self.model_to_load is None:
                print("load model first!")
                return None, None
            self.model_to_load.eval()
            with torch.no_grad():
                _, recon_data, info, theta_mean_, theta_logvar_ = self.model_to_load(x)
            x_mean = torch.hstack([theta_mean_,state_data_mean])
        else:
            x_mean = x[:,-1,:].to(torch.device("cuda"))                   
        
        if self.Distributional_GP:            
            state_data_std = torch.mul(torch.ones(state_data_mean.shape),2*torch.log(torch.ones(1)*self.std_for_xdata)).to(torch.device("cuda"))        
            if self.LatentPolicy:                
                x_var =  torch.hstack([theta_logvar_,state_data_std])            
                test_x = torch.hstack((x_mean, x_var)).to(torch.device("cuda"))                           
        else:
            test_x = x_mean.to(torch.device("cuda"))
        
        self.model_accel.eval()
        self.likelihood_accel.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():    
            acc_observed_pred = self.likelihood_accel(self.model_accel(test_x))
        return acc_observed_pred
    
    def draw_output(self,x,y_test):
        x = x.to(torch.device("cuda"))
        if len(y_test.shape) > 2:
            y_test_accel = y_test[:,-1,0].to(torch.device("cuda"))
        else:
            y_test_accel = y_test[:,0].to(torch.device("cuda"))
        
        
        with torch.no_grad():            
            # f, ax = plt.subplots(1, 1, figsize=(8, 3))            
            # test for preprocessed input
            # acc_observed_pred = self.directforward(x)
            # test for testbatch data
            acc_observed_pred = self.forward(x)
            lower, upper = acc_observed_pred.confidence_region()
            # Plot predictive means as blue line
            x_axis = np.linspace(1,len(acc_observed_pred.mean),len(acc_observed_pred.mean))
            plt.plot(x_axis, acc_observed_pred.mean.cpu().numpy(), 'b')
            plt.plot(x_axis, y_test_accel.cpu().numpy(), 'g')
            plt.fill_between(x_axis, lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.5)
            plt.legend(['GP Mean', 'Ground Truth', 'Cov'])
            plt.pause(0.05)            

            mseloss_for_testing = torch.nn.MSELoss()
            mse_val_loss = mseloss_for_testing(acc_observed_pred.mean, y_test_accel)
            return mse_val_loss
                          
            
            
            
            
                   


latent_RacingGP = RacingGP()
datafilename = 'trainx_0_fullstate.pkl'
latent_RacingGP.dataloading(datafilename)
latent_RacingGP.NNmodelLoad()
latent_RacingGP.train_test_data_preprocess()
     


latent_RacingGP.setupGPModel()
latent_RacingGP.GPtrain(2)
# test_batch = next(iter(latent_RacingGP.test_dataloader))
# x_test = latent_RacingGP.batch_x_data[0] # test_batch[:,:,0:6]
# y_test = latent_RacingGP.batch_y_data[0] # test_batch[:,:,6:]

# x_test =  test_batch[:,:,0:6]
# y_test =  test_batch[:,:,6:]


# x_test = test_x_distributionals[0] 
# y_test = test_y_datas[0][:,0]

# latent_RacingGP.draw_output(x_test,y_test)



