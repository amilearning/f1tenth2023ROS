import torch
import gpytorch
from predictor.prediction.abstract_gp_controller import GPController
import array
from tqdm import tqdm
import numpy as np
from predictor.h2h_configs import *
from predictor.common.utils.file_utils import *
import torch.nn as nn
from predictor.prediction.dyn_prediction_model import TorchDynamicsModelForPredictor
from predictor.common.tracks.radius_arclength_track import RadiusArclengthTrack
from predictor.prediction.covGP.covGPNN_gp_nn_model import COVGPNNModel, COVGPNNModelWrapper
from torch.utils.data import DataLoader, random_split
from typing import Type, List
from predictor.prediction.covGP.covGPNN_dataGen import SampleGeneartorCOVGP
import sys
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from predictor.prediction.torch_utils import get_curvature_from_keypts_torch
import time
import torch.optim.lr_scheduler as lr_scheduler
from predictor.common.utils.scenario_utils import torch_wrap_del_s

class COVGPNN(GPController):
    def __init__(self, args, sample_generator: SampleGeneartorCOVGP, model_class: Type[gpytorch.models.GP],
                 likelihood: gpytorch.likelihoods.Likelihood,
                 enable_GPU=False):
        if args is None:
            self.args = {                    
            "batch_size": 512,
            "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            "input_dim": 9,
            "n_time_step": 10,
            "latent_dim": 4,
            "gp_output_dim": 4,
            "inducing_points" : 300,
            "train_nn" : False                
            }
        else: 
            self.args = args
        self.train_nn = self.args["train_nn"]
        input_size = self.args["input_dim"]
        output_size = self.args["gp_output_dim"]
        inducing_points = self.args["inducing_points"]
        super().__init__(sample_generator, model_class, likelihood, input_size, output_size, enable_GPU)
        
        self.model = COVGPNNModel(self.args).to(device='cuda')
        self.independent = True        
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        
        

    def setup_dataloaders(self,train_dataload,valid_dataload, test_dataloader):
        self.train_loader = train_dataload
        self.valid_loader = valid_dataload
        self.test_loader = test_dataloader


    def pull_samples(self, holdout=150):        
        return 
       


    def outputToReal(self, output):
        if self.normalize:
            return output

        if self.means_y is not None:
            return output * self.stds_y + self.means_y
        else:
            return output




    def cosine_loss(self, z: torch.tensor, z_hat: torch.tensor) -> torch.tensor:
        """ This function calculates the Cosine Loss for a given set of input tensors z and z_hat. The Cosine Loss is
        defined as the negative mean of the cosine similarity between z and z_hat and aims to
        minimize the cosine distance between the two tensors z and z_hat, rather than maximizing their similarity.

        Args:
            - z:        (batch_size, seq_len, output_dim)
            - z_hat:    (batch_size, seq_len, output_dim)
        Returns:
            - loss: torch.tensor
        """
        cos_fn = nn.CosineSimilarity(dim=1).to(z.device)
        cos_sim = cos_fn(z, z_hat)
        # loss = -torch.mean(cos_sim, dim=0).mean()
        loss = -torch.mean(cos_sim, dim=0)
        return loss



    def comput_hidden_norm_constat(self,sampGen: SampleGeneartorCOVGP, args = None):
        train_dataset, val_dataset, test_dataset  = sampGen.get_datasets()
        batch_size = args["batch_size"]
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for step, (train_x, train_y) in enumerate(train_dataloader):                
            train_x_h , train_x_f = train_x[:,:,:int(train_x.shape[-1]/2)], train_x[:,:,int(train_x.shape[-1]/2):]                     
            self.model.eval()            
            torch.cuda.empty_cache()       
            
            #####################



        ############3                
        train_x_h , train_x_f = train_x[:,:,:int(train_x.shape[-1]/2)], train_x[:,:,int(train_x.shape[-1]/2):]                     
        output, encode_future_embeds, fcst_future_embeds = self.model(train_x_h, train_x_f ,train=True, directGP = directGP)
        variational_loss = -mll(output, train_y)


    def euclidean_loss(self, encode_future_embeds: torch.Tensor, fcst_future_embeds: torch.Tensor):        
        dist = torch.cdist(encode_future_embeds,fcst_future_embeds)
        return torch.trace(dist) / dist.shape[0]
        
    
    def train(self,sampGen: SampleGeneartorCOVGP, args = None):
        
        directGP = args['direct_gp']
        include_simts_loss = args['include_simts_loss']
        gp_name = 'simtsGP'
        if directGP:
            gp_name = 'naiveGP'
        else:   
            if include_simts_loss:
                gp_name = 'simtsGP'                
            else:
                gp_name = 'nosimtsGP'                

        n_epoch = args["n_epoch"]


        self.writer = SummaryWriter()

        train_dataset, val_dataset, test_dataset  = sampGen.get_datasets()
        batch_size = self.args["batch_size"]
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        if self.enable_GPU:
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
          
        
        self.model.train()
        self.likelihood.train()
                                                                                       
        optimizer_gp = torch.optim.Adam([{'params': self.model.gp_layer.hyperparameters()},  
                                    {'params': self.model.gp_layer.variational_parameters()},                                      
                                    {'params': self.likelihood.parameters()},
                                        ], lr=0.005)

        optimizer_nn = torch.optim.Adam([{'params': self.model.encdecnn.parameters(), 'lr': 0.05, 'weight_decay':1e-9}])
                    
        # optimizer_all = torch.optim.Adam([{'params': self.model.encdecnn.parameters(), 'lr': 0.002, 'weight_decay':1e-9},                                            
        optimizer_all = torch.optim.Adam([{'params': self.model.encdecnn.parameters(), 'lr': 0.002, 'weight_decay':1e-9},                                            
                                        {'params': self.model.gp_layer.hyperparameters(), 'lr': 0.005},
                                        {'params': self.model.gp_layer.variational_parameters()},
                                        {'params': self.likelihood.parameters()},
                                        ], lr=0.005)

        scheduler = lr_scheduler.StepLR(optimizer_gp, step_size=3000, gamma=0.9)
        

        # GP marginal log likelihood
        # p(y | x, X, Y) = ∫p(y|x, f)p(f|X, Y)df
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model.gp_layer, num_data=sampGen.getNumSamples()*len(sampGen.output_data[0]))
        mseloss = nn.MSELoss()
        max_epochs = n_epoch* len(train_dataloader)
        last_loss = np.inf
        no_progress_epoch = 0
        done = False
        epoch = 0
        best_model = None
        best_likeli = None

        best_recon_loss = np.inf
        no_progress_best_recon = 0
        recon_loss_cerverged = False
        best_dist_loss = np.inf
        no_progress_best_dist = 0
        dist_loss_cerverged = False

        sys.setrecursionlimit(100000)
        
        while not done:
        # for _ in range(epochs):
            train_dataloader = tqdm(train_dataloader)
            valid_dataloader = tqdm(valid_dataloader)
            train_loss = 0
            valid_loss = 0
            c_loss = 0
            recon_loss_sum = 0
            latent_dist_loss_sum = 0
            variational_loss_sum = 0
            for step, (train_x, train_y) in enumerate(train_dataloader):                
                self.model.train()
                self.likelihood.train()     
                torch.cuda.empty_cache()       
                optimizer_gp.zero_grad()    
                optimizer_nn.zero_grad()
                optimizer_all.zero_grad()
                #####################           
                if int(train_x.shape[-1]/2) > 0:
                    train_x_h , train_x_f = train_x[:,:,:int(train_x.shape[-1]/2)], train_x[:,:,int(train_x.shape[-1]/2):]                 
                else:    
                    train_x_h , train_x_f = train_x, train_x                
                output, recon_data, latent_x = self.model(train_x_h,train=True)

                ############# Compute Variational LOSS ####################
                variational_loss = -mll(output, train_y)                
                variational_loss_sum += variational_loss.item()

                ############# Compute Reconstruction LOSS ####################
                reconloss_weight = 1.0
                reconloss = (mseloss(recon_data,train_x_f)* reconloss_weight)
                recon_loss_sum +=reconloss.item()
                if reconloss > best_recon_loss:
                    best_recon_loss = reconloss.item()
                    no_progress_best_recon = 0
                else:
                    no_progress_best_recon += 1
                    if no_progress_best_recon > 20:
                        recon_loss_cerverged = True
                
                
                if directGP:
                    loss = variational_loss
                else:
                    ############# Compute Latent Distance LOSS ####################
                    latent_dist_loss = 0.0
                    

                    # latent_dist = torch.cov(latent_x).float()
                    # out_cov = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5)).to("cuda").double()                                                
                    # out_cov.base_kernel.lengthscale =  0.67 # (self.model.gp_layer.covar_module.base_kernel.lengthscale[i])                   
                    
                    out_dist= torch.cov(train_y).float()
                        # lengthscale_tmp =  min(max(epoch * 0.005 + 0.68, 0.2), 1e3)
                    
                    yinvKy_loss = 0
                    for i in range(self.model.gp_layer.covar_module.base_kernel.batch_shape[0]):
                        jitter = torch.eye(train_y.shape[0]).cuda()*1e-11
                        cov = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5)).to("cuda").double()                                                
                        cov.base_kernel.lengthscale =  (self.model.gp_layer.covar_module.base_kernel.lengthscale[i])                   
                        latent_dist= cov.base_kernel(latent_x,latent_x).evaluate()

                        outcov = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5)).to("cuda").double()                                                
                        outcov.base_kernel.lengthscale =  (self.model.gp_layer.covar_module.base_kernel.lengthscale[i])*0.9                  
                        out_dist = outcov.base_kernel(train_y[:,i], train_y[:,i]).evaluate()

                        yinvKy = (train_y[:,i].unsqueeze(dim=0) @ torch.cholesky_inverse(latent_dist + jitter) @ train_y[:,i].unsqueeze(dim=1))/latent_dist.shape[0]
                        yinvKy_loss += yinvKy[0][0]/self.model.gp_layer.covar_module.base_kernel.batch_shape[0]  # mseloss(latent_dist,out_dist)                    
                        
                        mse = mseloss(latent_dist,out_dist)
                        cos = self.cosine_loss(latent_dist, out_dist)
                        

                        latent_dist_loss += mse
                        

                    latent_dist_loss_weight = 1.0
                    latent_dist_loss = (latent_dist_loss * latent_dist_loss_weight)
                    latent_dist_loss_sum +=latent_dist_loss.item()                    

                    if latent_dist_loss > best_dist_loss:
                        best_dist_loss = latent_dist_loss.item()
                        no_progress_best_dist = 0
                    else:
                        no_progress_best_dist += 1
                        if no_progress_best_dist > 20:
                            dist_loss_cerverged = True
                            
                    
                    if include_simts_loss:    
                        if epoch > 150:
                            loss =    variational_loss 
                        else:                        
                            loss =   latent_dist_loss + reconloss
                            no_progress_epoch = 0                        
                    else:
                        loss = variational_loss 
                train_loss += loss.item()    
                loss.backward()

                if directGP:
                    optimizer_gp.step()
                else:
                    if include_simts_loss:
                        # if recon_loss_cerverged and dist_loss_cerverged and epoch > 500:
                        if epoch > 150:
                            optimizer_gp.step()
                        else:
                            optimizer_all.step()
                    else:
                        optimizer_all.step()
                                    

            if directGP is False:
                tag = 'Lengthscale/yinvKy_loss' + gp_name
                self.writer.add_scalar(tag, yinvKy_loss, epoch)                    

            for i, param_group in enumerate(optimizer_gp.param_groups):
                lr_tag = gp_name+f'/Lr/learning_rate{i+1}'
                self.writer.add_scalar(lr_tag, param_group['lr'], epoch )                
            varloss_tag = gp_name+'/Loss/variational_loss'
            self.writer.add_scalar(varloss_tag, variational_loss_sum, epoch)
            reconloss_tag = gp_name +'/Loss/reconloss'
            self.writer.add_scalar(reconloss_tag, recon_loss_sum, epoch )
            latent_dist_loss_tag = gp_name+'/Loss/latent_dist_loss'
            self.writer.add_scalar(latent_dist_loss_tag, latent_dist_loss_sum, epoch)
            train_loss_tag = gp_name+'/Loss/total_train_loss' 
            self.writer.add_scalar(train_loss_tag, train_loss, epoch)
            
            scheduler.step()            
            if epoch % 50 ==0:
                snapshot_name = gp_name + str(epoch)+ 'snapshot'
                self.set_evaluation_mode()
                self.save_model(snapshot_name)
                self.model.train()
                self.likelihood.train()
            
            for step, (train_x, train_y) in enumerate(valid_dataloader):
                optimizer_gp.zero_grad()                
                output = self.model(train_x)
                loss = -mll(output, train_y)
                valid_loss += loss.item()
                c_loss = valid_loss / (step + 1)
                valid_dataloader.set_postfix(log={'valid_loss': f'{(c_loss):.5f}'})                
            
            valid_loss_tag = gp_name+'/Loss/valid_loss'
            self.writer.add_scalar(valid_loss_tag, valid_loss, epoch)
            if c_loss > last_loss:
                if no_progress_epoch >= 20:
                    if include_simts_loss:
                        if recon_loss_cerverged and dist_loss_cerverged and epoch > 300: 
                            if no_progress_epoch > 20:
                                done = True   
                    else:
                        done = True 
            else:
                best_model = copy.copy(self.model)
                best_likeli = copy.copy(self.likelihood)
                last_loss = c_loss
                no_progress_epoch = 0            
            no_progress_epoch += 1
            epoch +=1
            if epoch > max_epochs:
                done = True

        self.model = best_model
        self.likelihood = best_likeli
        print("test done")
    
    def evaluate(self):       
        import matplotlib.pyplot as plt
        self.set_evaluation_mode()        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # This contains predictions for both outcomes as a list
            # for step, (train_x, train_y) in enumerate(self.test_loader):                               
            (self.test_x, self.test_y) = next(iter(self.test_loader))
            predictions = self.likelihood(self.likelihood(self.model(self.test_x)))

        mean = predictions.mean.cpu()
        variance = predictions.variance.cpu()
        std = predictions.stddev.cpu()
        # self.means_x = self.means_x.cpu()
        # self.means_y = self.means_y.cpu()
        # self.stds_x = self.stds_x.cpu()
        # self.stds_y = self.stds_y.cpu()
        self.test_y = self.test_y.cpu()

        f, ax = plt.subplots(self.output_size, 1, figsize=(15, 10))
        titles = ['xtran', 'epsi', 'vlong']
        for i in range(self.output_size):
            # unnormalized_mean = self.stds_y[0, i] * mean[:, i] + self.means_y[0, i]
            # unnormalized_mean = unnormalized_mean.detach().numpy()
            unnormalized_mean = mean[:,i].detach().numpy()
            # cov = np.sqrt((variance[:, i] * (self.stds_y[0, i] ** 2)))
            cov = std[:,i]
            cov = cov.detach().numpy()
            '''lower, upper = prediction.confidence_region()
            lower = lower.detach().numpy()
            upper = upper.detach().numpy()'''
            lower = unnormalized_mean - 2 * cov
            upper = unnormalized_mean + 2 * cov
            # tr_y = self.stds_y[0, i] * self.test_y[:50, i] + self.means_y[0, i]
            tr_y = self.test_y[:, i]
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


    
    def pred_eval(self,sampGen: SampleGeneartorCOVGP):
        
        self.writer = SummaryWriter()
        train_dataset, val_dataset, test_dataset  = sampGen.get_datasets()
        batch_size = self.args["batch_size"]
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  

        if self.enable_GPU:
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
        # Find optimal model hyper-parameters
        self.model.eval()
        self.likelihood.eval()
        
        z_tmp_list = []
        input_list = []
        dist_thres = np.inf  ## 
        vx_thres = -1*np.inf
        for step, (data_x, data_y) in enumerate(dataloader):    
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                latent_x = self.model.get_hidden(data_x)
                
                delta_s_avg = torch.mean(data_x[:,0,:],dim=1)
                vx_avg, tmp = torch.max(data_x[:,3,:],dim=1)
                filtered_idx = (vx_avg > vx_thres)*(delta_s_avg<dist_thres)*(delta_s_avg>-1*dist_thres)
                selected_latent_x = latent_x[filtered_idx,:]
                selected_data_x = data_x[filtered_idx,:,:]

                z_tmp_list.append(selected_latent_x.view(selected_latent_x.shape[0],-1))
                input_list.append(selected_data_x)
                stacked_z_tmp = torch.cat(z_tmp_list, dim=0)
                input_list_tmp= torch.cat(input_list, dim=0)

    def tsne_evaluate(self,sampGen: SampleGeneartorCOVGP):
        
        self.writer = SummaryWriter()
        train_dataset, val_dataset, test_dataset  = sampGen.get_datasets()
        batch_size = self.args["batch_size"]
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  

        if self.enable_GPU:
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
        # Find optimal model hyper-parameters
        self.model.eval()
        self.likelihood.eval()
        
        z_tmp_list = []
        input_list = []
        dist_thres = np.inf  ## 
        vx_thres = -1*np.inf
        for step, (data_x, data_y) in enumerate(dataloader):    
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                latent_x = self.model.get_hidden(data_x)
                
                
                delta_s_avg = torch.mean(data_x[:,0,:],dim=1)
                vx_avg, tmp = torch.max(data_x[:,3,:],dim=1)
                filtered_idx = (vx_avg > vx_thres)*(delta_s_avg<dist_thres)*(delta_s_avg>-1*dist_thres)
                selected_latent_x = latent_x[filtered_idx,:]
                selected_data_x = data_x[filtered_idx,:,:]

                z_tmp_list.append(selected_latent_x.view(selected_latent_x.shape[0],-1))
                input_list.append(selected_data_x)
                stacked_z_tmp = torch.cat(z_tmp_list, dim=0)
                input_list_tmp= torch.cat(input_list, dim=0)

        return stacked_z_tmp, input_list_tmp





class COVGPNNTrained(GPController):
    def __init__(self, name, enable_GPU, load_trace = False, model=None):        
        if model is not None:
            self.load_model_from_object(model)
        else:
            self.load_model(name)
        self.enable_GPU = enable_GPU
        
        self.load_normalizing_consant(name= name)
        if self.enable_GPU:
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
            # self.means_x = self.means_x.cuda()
            # self.means_y = self.means_y.cuda()
            # self.stds_x = self.stds_x.cuda()
            # self.stds_y = self.stds_y.cuda()
        else:
            self.model.cpu()
            self.likelihood.cpu()
            # self.means_x = self.means_x.cpu()
            # self.means_y = self.means_y.cpu()
            # self.stds_x = self.stds_x.cpu()
            # self.stds_y = self.stds_y.cpu()
        self.load_trace = load_trace
        self.trace_model = None
        if self.load_trace:
            self.gen_trace_model()
            

    def gen_trace_model(self):
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.trace_mode():
            self.model.eval()
            test_x = torch.randn(25,9,10).cuda()
            pred = self.model(test_x)  # Do precomputation
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.trace_mode():
            self.trace_model = torch.jit.trace(COVGPNNModelWrapper(self.model), test_x)
            
            

    def load_normalizing_consant(self, name ='normalizing'):        
        
        model = pickle_read(os.path.join(model_dir, name + '_normconstant.pkl'))        
        self.means_x = model['mean_sample'].cuda()
        if int(self.means_x.shape[1]/2) > 1:
            self.means_x = self.means_x[:,:int(self.means_x.shape[1]/2)]
        self.means_y = model['mean_output'].cuda()
        self.stds_x = model['std_sample'].cuda()
        if int(self.stds_x.shape[1]/2) > 1:
            self.stds_x = self.stds_x[:,:int(self.stds_x.shape[1]/2)]
        self.stds_y = model['std_output'].cuda()        
        # self.independent = model['independent'] TODO uncomment        
        print('Successfully loaded normalizing constants', name)



    def get_true_prediction_par(self, input,  ego_state: VehicleState, target_state: VehicleState,
                                ego_prediction: VehiclePrediction, track: RadiusArclengthTrack, M=10):
       
        
        # Set GP model to eval-mode
        self.set_evaluation_mode()
        # draw M samples
        
        preds = self.sample_traj_gp_par(input, ego_state, target_state, ego_prediction, track, M)
        
        # numeric mean and covariance calculation.
        # cov_start_time = time.time()
        pred = self.mean_and_cov_from_list(preds, M) 
        # cov_end_time = time.time()
        # cov_elapsed_time = cov_end_time - cov_start_time
        # print(f"COV Elapsed time: {cov_elapsed_time} seconds")    
        pred.t = ego_state.t

        

        return pred



    def insert_to_end(self, roll_input, tar_state, tar_curv, ego_state, track):        
        roll_input[:,:,:-1] = roll_input[:,:,1:]
        input_tmp = torch.zeros(roll_input.shape[0],roll_input.shape[1]).to('cuda')        
        input_tmp[:,0] = tar_state[:,0]-ego_state[:,0]                      
        input_tmp[:,0] = torch_wrap_del_s(tar_state[:,0],ego_state[:,0], track)
        input_tmp[:,1] = tar_state[:,1]
        input_tmp[:,2] = tar_state[:,2]
        input_tmp[:,3] = tar_state[:,3]
        input_tmp[:,4] = tar_curv[:,0]
        input_tmp[:,5] = tar_curv[:,1]
        input_tmp[:,6] = ego_state[:,1]
        input_tmp[:,7] = ego_state[:,2] 
        input_tmp[:,8] = ego_state[:,3]                                           
        roll_input[:,:,-1] = input_tmp
        return roll_input.clone()

    
    def sample_traj_gp_par(self, encoder_input,  ego_state: VehicleState, target_state: VehicleState,
                           ego_prediction: VehiclePrediction, track: RadiusArclengthTrack, M):

        '''
        encoder_input = batch x feature_dim x time_horizon
        '''     
        prediction_samples = []
        for j in range(M):
            tmp_prediction = VehiclePrediction() 
            tmp_prediction.s = [target_state.p.s]
            tmp_prediction.x_tran = [target_state.p.x_tran]
            tmp_prediction.e_psi = [ target_state.p.e_psi]
            tmp_prediction.v_long = [ target_state.v.v_long]                          
            prediction_samples.append(tmp_prediction)
    
        

        roll_input = encoder_input.repeat(M,1,1).to('cuda') 
        roll_tar_state = torch.tensor([target_state.p.s, target_state.p.x_tran, target_state.p.e_psi, target_state.v.v_long]).to('cuda')        
        roll_tar_state = roll_tar_state.repeat(M,1)
        roll_tar_curv = torch.tensor([target_state.lookahead.curvature[0], target_state.lookahead.curvature[2]]).to('cuda')        
        roll_tar_curv = roll_tar_curv.repeat(M,1)
        roll_ego_state = torch.tensor([ego_state.p.s, ego_state.p.x_tran, ego_state.p.e_psi, ego_state.v.v_long]).to('cuda')
        roll_ego_state = roll_ego_state.repeat(M,1)

        horizon = len(ego_prediction.x)    
        # start_time = time.time()
        for i in range(horizon-1):         
            # gp_start_time = time.time()  
            roll_input = self.insert_to_end(roll_input, roll_tar_state, roll_tar_curv, roll_ego_state, track)                      
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                if self.load_trace:
                    mean, stddev = self.trace_model(self.standardize(roll_input))
                else:
                    pred_delta_dist = self.model(self.standardize(roll_input))
                    mean = pred_delta_dist.mean
                    stddev = pred_delta_dist.stddev
                # pred_delta_dist = self.model(roll_input)            
                # print(stddev.cpu().numpy())
                
                tmp_delta = torch.distributions.Normal(mean, stddev).sample()            
                    
                pred_delta = self.outputToReal(tmp_delta)
       
            roll_tar_state[:,0] += pred_delta[:,0] 
            roll_tar_state[:,1] += pred_delta[:,1] 
            roll_tar_state[:,2] += pred_delta[:,2]
            roll_tar_state[:,3] += pred_delta[:,3]  
            roll_tar_curv[:,0] = get_curvature_from_keypts_torch(pred_delta[:,0].clone().detach(),track)
            roll_tar_curv[:,1] = get_curvature_from_keypts_torch(pred_delta[:,0].clone().detach()+target_state.lookahead.dl*2,track)                        
            roll_ego_state[:,0] = ego_prediction.s[i+1]
            roll_ego_state[:,1] = ego_prediction.x_tran[i+1]
            roll_ego_state[:,2] =  ego_prediction.e_psi[i+1]
            roll_ego_state[:,3] =  ego_prediction.v_long[i+1]


            for j in range(M):                          # tar 0 1 2 3 4 5       #ego 6 7 8 9 10 11
                prediction_samples[j].s.append(roll_tar_state[j,0].cpu().numpy())
                prediction_samples[j].x_tran.append(roll_tar_state[j,1].cpu().numpy())                    
                prediction_samples[j].e_psi.append(roll_tar_state[j,2].cpu().numpy())
                prediction_samples[j].v_long.append(roll_tar_state[j,3].cpu().numpy())
        
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Time taken for GP(over horizons) call: {elapsed_time} seconds")

        for i in range(M):
            prediction_samples[i].s = array.array('d', prediction_samples[i].s)
            prediction_samples[i].x_tran = array.array('d', prediction_samples[i].x_tran)
            prediction_samples[i].e_psi = array.array('d', prediction_samples[i].e_psi)
            prediction_samples[i].v_long = array.array('d', prediction_samples[i].v_long)            

        
        return prediction_samples



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
 

        
