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
from predictor.prediction.torch_utils import torch_wrap_s
class COVGPNN(GPController):
    def __init__(self, args, sample_generator: SampleGeneartorCOVGP, model_class: Type[gpytorch.models.GP],
                 likelihood: gpytorch.likelihoods.Likelihood,
                 enable_GPU=False):
        if args is None:
            self.args = {                    
            "batch_size": 512,
            "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            "input_dim": 10,
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
        self.output_size = output_size
        inducing_points = self.args["inducing_points"]
        super().__init__(sample_generator, model_class, likelihood, input_size, output_size, enable_GPU)
        
        self.model = COVGPNNModel(self.args).to(device='cuda')
        self.independent = True        
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        

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


    def euclidean_loss(self, encode_future_embeds: torch.Tensor, fcst_future_embeds: torch.Tensor):        
        dist = torch.cdist(encode_future_embeds,fcst_future_embeds)
        return torch.trace(dist) / dist.shape[0]
        
    
    def train(self,sampGen: SampleGeneartorCOVGP,valGEn : SampleGeneartorCOVGP,  args = None):
        self.writer = SummaryWriter()
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

        train_dataset, _, _  = sampGen.get_datasets()
        batch_size = self.args["batch_size"]
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        

        valid_dataset, _, _  = valGEn.get_datasets()        
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)


        if self.enable_GPU:
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
          
        
        self.model.train()
        #self.model.in_covs.train()
        #self.model.out_covs.train()
        self.likelihood.train()
                                                                                       
        optimizer_gp = torch.optim.Adam([{'params': self.model.gp_layer.hyperparameters()},  
                                    {'params': self.model.gp_layer.variational_parameters()},                                      
                                    {'params': self.likelihood.parameters()},
                                        ], lr=0.005)

        
        
        # optimizer_all = torch.optim.Adam([{'params': self.model.in_covs[0].parameters(), 'lr': 0.01, 'weight_decay':1e-8},
        #                                 {'params': self.model.out_covs[0].parameters(), 'lr': 0.01, 'weight_decay':1e-8},
        #                                 {'params': self.model.in_covs[1].parameters(), 'lr': 0.01, 'weight_decay':1e-8},
        #                                 {'params': self.model.out_covs[1].parameters(), 'lr': 0.01, 'weight_decay':1e-8},
        #                                 {'params': self.model.in_covs[2].parameters(), 'lr': 0.01, 'weight_decay':1e-8},
        #                                 {'params': self.model.out_covs[2].parameters(), 'lr': 0.01, 'weight_decay':1e-8},
        #                                 {'params': self.model.in_covs[3].parameters(), 'lr': 0.01, 'weight_decay':1e-8},
        #                                 {'params': self.model.out_covs[3].parameters(), 'lr': 0.01, 'weight_decay':1e-8}                                        
        #                                 ], lr=0.005)

        # optimizer_all = torch.optim.Adam([{'params': self.model.encdecnn.parameters(), 'lr': 0.01, 'weight_decay':1e-9},                                          
        optimizer_all = torch.optim.Adam([{'params': self.model.encdecnn.parameters(), 'lr': 0.05, 'weight_decay':1e-9},                                          
                                        {'params': self.model.gp_layer.hyperparameters(), 'lr': 0.005},                                        
                                        # {'params': self.model.in_covs.parameters(), 'lr': 0.01, 'weight_decay':1e-8},
                                        # {'params': self.model.out_covs.parameters(), 'lr': 0.01, 'weight_decay':1e-8},                                        
                                        {'params': self.model.in_covs.parameters(), 'lr': 0.01, 'weight_decay':1e-8},
                                        {'params': self.model.out_covs.parameters(), 'lr': 0.01, 'weight_decay':1e-8}, 
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
        nn_only_epoch = 0
        best_model = None
        best_likeli = None

        sys.setrecursionlimit(100000)
        # dummy_input = torch.randn(161,9,10).cuda()        
        # self.writer.add_graph(self.model, dummy_input)
        self.model.double()
        while not done:
        # for _ in range(epochs):
            train_dataloader = tqdm(train_dataloader)
            valid_dataloader = tqdm(valid_dataloader)
            train_loss = 0
            valid_loss = 0
            c_loss = 0
            recon_loss_sum = 0
            latent_dist_loss_sum = 0
            std_loss_sum = 0
            scale_loss_sum = 0
            cosine_loss_sum = 0
            cov_loss = 0 
            variational_loss_sum = 0 
            for step, (train_x, train_y) in enumerate(train_dataloader):                
                # self.model.train()
                # self.likelihood.train()     
                torch.cuda.empty_cache()   
                optimizer_all.zero_grad()
                optimizer_gp.zero_grad() 
                
                #####################           
                if int(len(train_x.shape)) > 2:
                    train_x_h  = train_x[:,:,:int(train_x.shape[-1]/2)].double()
                else:    
                    train_x_h  = train_x.double()          
             
                output = self.model(train_x_h.cuda())                
                

                if include_simts_loss: 
                    latent_x = self.model.get_hidden(train_x_h.cuda())
                    cos_loss = 0 # self.cosine_loss(latent_x, train_y)
                    cov_loss = 0    
                    std_loss = 0      
                    scale_loss = 0                
                    for i in range(self.output_size):
                        latent_dist = self.model.in_covs[i](latent_x, latent_x)                     
                        out_dist = self.model.out_covs[i](train_y[:,i].cuda(), train_y[:,i].cuda())
                        out_dist = out_dist.evaluate()
                        latent_dist = latent_dist.evaluate()
                        # abs(torch.eye(latent_dist.shape[0]).cuda()-out_dist) * (torch.eye(latent_dist.shape[0]).cuda()-latent_dist)
                        # std_loss += #torch.mean(abs(torch.eye(latent_dist.shape[0]).cuda()-out_dist) * (torch.eye(latent_dist.shape[0]).cuda()-latent_dist))
                        std_loss  += torch.log((self.model.out_covs[i].lengthscale)/(self.model.in_covs[i].lengthscale + 1e-12))*5e-2               
                        # std_loss  += 1/(self.model.in_covs[i].lengthscale + 1e-9)*1e-1             +1/(self.model.out_covs[i].lengthscale + 1e-9)*1e-1   
                        
                        scale_loss += torch.log(1/self.model.gp_layer.covar_module.outputscale[i])*1e-2
                          
                        
                        cov_loss += mseloss(out_dist, latent_dist)                     
                        
                    latent_std = torch.std(latent_x)                                                   
                    if latent_std > 2.0:
                        std_loss += latent_std*0.1
                    ############# ############################ ####################        
                    # loss =    cov_mse #+ reconloss
                variational_loss = -mll(output, train_y)                
                
                if include_simts_loss:                     
                    loss = cov_loss + variational_loss + std_loss  + scale_loss
                else:
                    loss = variational_loss

                loss.backward()

                if directGP:
                    optimizer_gp.step()
                else:                          
                    optimizer_all.step()
                     
                train_loss += loss.item()    
                variational_loss_sum += variational_loss.item()
                
                if include_simts_loss: 
                    latent_dist_loss_sum +=cov_loss.item()
                    std_loss_sum += std_loss.item()
                    scale_loss_sum += scale_loss.item()
                    cosine_loss_sum += 0 # cos_loss.item()
                    
                    self.writer.add_scalar(gp_name+'/stat/latent_max', torch.max(latent_x), epoch*len(train_dataloader) + step)
                    self.writer.add_scalar(gp_name+'/stat/latent_min', torch.min(latent_x), epoch*len(train_dataloader) + step)
                    self.writer.add_scalar(gp_name+'/stat/latent_std', torch.std(latent_x), epoch*len(train_dataloader) + step)

            for i in range(self.output_size):                        
                
                outputscale_tar = self.model.gp_layer.covar_module.outputscale[i].item()
                outputscale_tag = gp_name+f'/outputscale_{i}'
                self.writer.add_scalar(outputscale_tag, outputscale_tar, epoch )    

                in_lengthscale = self.model.in_covs[i].lengthscale.item()
                lin_tag = gp_name+f'/lengthscale/in_{i}'
                self.writer.add_scalar(lin_tag, in_lengthscale, epoch )    
                out_lengthscale = self.model.out_covs[i].lengthscale.item()
                lout_tag = gp_name+f'/lengthscale/out_{i}'
                self.writer.add_scalar(lout_tag, out_lengthscale, epoch )                            
                model_lengthscale = self.model.gp_layer.covar_module.base_kernel.lengthscale[i].item()
                lmodel_tag = gp_name+f'/lengthscale/model_{i}'
                self.writer.add_scalar(lmodel_tag, model_lengthscale, epoch )                            
                
            # for i, param_group in enumerate(optimizer_gp.param_groups):
            #     lr_tag = gp_name+f'/Lr/learning_rate{i+1}'
            #     self.writer.add_scalar(lr_tag, param_group['lr'], epoch )                
                
            cosine_tag = gp_name+'/Loss/cosine_loss'
            self.writer.add_scalar(cosine_tag, cosine_loss_sum/ len(train_dataloader), epoch)
            varloss_tag = gp_name+'/Loss/variational_loss'
            self.writer.add_scalar(varloss_tag, variational_loss_sum/ len(train_dataloader), epoch)
            std_loss_tag = gp_name +'/Loss/std_loss'
            self.writer.add_scalar(std_loss_tag, std_loss_sum / len(train_dataloader), epoch )
            scale_loss_sum_tag = gp_name +'/Loss/scale_loss'
            self.writer.add_scalar(scale_loss_sum_tag, scale_loss_sum / len(train_dataloader), epoch )
            
            latent_dist_loss_tag = gp_name+'/Loss/latent_dist_loss'
            self.writer.add_scalar(latent_dist_loss_tag, latent_dist_loss_sum/ len(train_dataloader) , epoch)
            train_loss_tag = gp_name+'/Loss/total_train_loss' 
            self.writer.add_scalar(train_loss_tag, train_loss/ len(train_dataloader), epoch)
            
            scheduler.step()            
            if epoch % 50 ==0:
                snapshot_name = gp_name + str(epoch)+ 'snapshot'
                self.set_evaluation_mode()
                self.save_model(snapshot_name)
                self.model.train()
                self.likelihood.train()

            
            for step, (test_x, test_y) in enumerate(valid_dataloader):
                torch.cuda.empty_cache()   
                optimizer_gp.zero_grad()                
                optimizer_all.zero_grad()
                
                if int(len(test_x.shape)) > 2:
                    test_x  = test_x[:,:,:int(test_x.shape[-1]/2)].double()
                else:    
                    test_x = test_x.double()         
                
                output = self.model(test_x)
                loss = -mll(output, test_y)
                valid_loss += loss.item()
                c_loss = valid_loss / (step + 1)
                valid_dataloader.set_postfix(log={'valid_loss': f'{(c_loss):.5f}'})                
            
            valid_loss_tag = gp_name+'/Loss/valid_loss'
            self.writer.add_scalar(valid_loss_tag, valid_loss, epoch)
            if c_loss > last_loss:
                if no_progress_epoch >= 15:
                    if include_simts_loss:     
                        if epoch > nn_only_epoch:                   
                            if no_progress_epoch > 100:
                                done = True   
                                # done = False ## TODO: Delete
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
    
    # def evaluate(self):       
    #     import matplotlib.pyplot as plt
    #     self.set_evaluation_mode()        
    #     with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #         # This contains predictions for both outcomes as a list
    #         # for step, (train_x, train_y) in enumerate(self.test_loader):                               
    #         (self.test_x, self.test_y) = next(iter(self.test_loader))
    #         predictions = self.likelihood(self.likelihood(self.model(self.test_x)))

    #     mean = predictions.mean.cpu()
    #     variance = predictions.variance.cpu()
    #     std = predictions.stddev.cpu()
    #     # self.means_x = self.means_x.cpu()
    #     # self.means_y = self.means_y.cpu()
    #     # self.stds_x = self.stds_x.cpu()
    #     # self.stds_y = self.stds_y.cpu()
    #     self.test_y = self.test_y.cpu()

    #     f, ax = plt.subplots(self.output_size, 1, figsize=(15, 10))
    #     titles = ['xtran', 'epsi', 'vlong']
    #     for i in range(self.output_size):
    #         # unnormalized_mean = self.stds_y[0, i] * mean[:, i] + self.means_y[0, i]
    #         # unnormalized_mean = unnormalized_mean.detach().numpy()
    #         unnormalized_mean = mean[:,i].detach().numpy()
    #         # cov = np.sqrt((variance[:, i] * (self.stds_y[0, i] ** 2)))
    #         cov = std[:,i]
    #         cov = cov.detach().numpy()
    #         '''lower, upper = prediction.confidence_region()
    #         lower = lower.detach().numpy()
    #         upper = upper.detach().numpy()'''
    #         lower = unnormalized_mean - 2 * cov
    #         upper = unnormalized_mean + 2 * cov
    #         # tr_y = self.stds_y[0, i] * self.test_y[:50, i] + self.means_y[0, i]
    #         tr_y = self.test_y[:, i]
    #         # Plot training data as black stars
    #         ax[i].plot(tr_y, 'k*')
    #         # Predictive mean as blue line
    #         # ax[i].scatter(np.arange(len(unnormalized_mean)), unnormalized_mean)
    #         ax[i].errorbar(np.arange(len(unnormalized_mean)), unnormalized_mean, yerr=cov, fmt="o", markersize=4, capsize=8)
    #         # Shade in confidence
    #         # ax[i].fill_between(np.arange(len(unnormalized_mean)), lower, upper, alpha=0.5)
    #         ax[i].legend(['Observed Data', 'Predicted Data'])
    #         ax[i].set_title(titles[i])
    #     plt.show()


    
    # def pred_eval(self,sampGen: SampleGeneartorCOVGP):
        
        
    #     train_dataset, _, _  = sampGen.get_datasets()
    #     batch_size = sampGen.getNumSamples()        
    #     dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  

    #     if self.enable_GPU:
    #         self.model = self.model.cuda()
    #         self.likelihood = self.likelihood.cuda()
    #     # Find optimal model hyper-parameters
    #     self.model.eval()
    #     self.likelihood.eval()
        
    #     z_tmp_list = []
    #     input_list = []
    #     dist_thres = np.inf  ## 
    #     vx_thres = -1*np.inf
    #     for step, (data_x, data_y) in enumerate(dataloader):    
    #         with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #             latent_x = self.model.get_hidden(data_x)
                
    #             delta_s_avg = torch.mean(data_x[:,0,:],dim=1)
    #             vx_avg, tmp = torch.max(data_x[:,3,:],dim=1)
    #             filtered_idx = (vx_avg > vx_thres)*(delta_s_avg<dist_thres)*(delta_s_avg>-1*dist_thres)
    #             selected_latent_x = latent_x[filtered_idx,:]
    #             selected_data_x = data_x[filtered_idx,:,:]

    #             z_tmp_list.append(selected_latent_x.view(selected_latent_x.shape[0],-1))
    #             input_list.append(selected_data_x)
    #             stacked_z_tmp = torch.cat(z_tmp_list, dim=0)
    #             input_list_tmp= torch.cat(input_list, dim=0)

    def tsne_evaluate(self,sampGen: SampleGeneartorCOVGP):
        
        
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
        output_list = []
        cov_list = []
        dist_thres = np.inf  ## 
        vx_thres = -1*np.inf
        for step, (data_x, data_y) in enumerate(dataloader):    
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                latent_x = self.model.get_hidden(data_x.double())
                out = self.model(data_x.double())
                out_std = out.stddev[:,1]
                # out.stddev
                
                
                delta_s_avg = torch.mean(data_x[:,0,:],dim=1)
                vx_avg, tmp = torch.max(data_x[:,3,:],dim=1)
                filtered_idx = (vx_avg > vx_thres)*(delta_s_avg<dist_thres)*(delta_s_avg>-1*dist_thres)
                selected_latent_x = latent_x[filtered_idx,:]
                selected_data_x = data_x[filtered_idx,:,:]
                selected_data_y = data_y[filtered_idx,:]
                

                z_tmp_list.append(selected_latent_x.view(selected_latent_x.shape[0],-1))
                input_list.append(selected_data_x)
                output_list.append(selected_data_y)
                cov_list.append(out_std[filtered_idx])
                stacked_z_tmp = torch.cat(z_tmp_list, dim=0)                
                input_list_tmp= torch.cat(input_list, dim=0)
                output_list_tmp= torch.cat(output_list, dim=0)
                cov_list_tmp= torch.cat(cov_list, dim=0)

        return stacked_z_tmp, input_list_tmp, output_list_tmp, cov_list_tmp





class COVGPNNTrained(GPController):
    def __init__(self, name, enable_GPU, load_trace = False, model=None, args = None, sample_num = 25):        
        self.M = sample_num
        if args is not None:
            self.input_dim = args['input_dim']
            self.n_time_step = args['n_time_step']
        else:
            self.input_dim = 9
            self.n_time_step = 10
        
        self.model_name = name

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
            if self.model_name == 'naiveGP':
                test_x = torch.randn(self.M,self.input_dim).cuda()
            else:
                test_x = torch.randn(self.M,self.input_dim,self.n_time_step).cuda()
            # CHeck the outputscale  and lengthscale
            # tmp_lengthscale = self.model.gp_layer.covar_module.base_kernel.lengthscale
        
            # tmp_out = self.model.gp_layer.covar_module.outputscale
            # xtran_scale = tmp_out[1]*2.0           
            # self.model.gp_layer.covar_module.raw_outputscale[1].data.fill_(self.model.gp_layer.covar_module.raw_outputscale_constraint.inverse_transform(xtran_scale))
             
            # xtran_scale1 = tmp_out[0]*2.0           
            # self.model.gp_layer.covar_module.raw_outputscale[0].data.fill_(self.model.gp_layer.covar_module.raw_outputscale_constraint.inverse_transform(xtran_scale1))
            # xtran_scale = tmp_out[1]*1.5           
            # self.model.gp_layer.covar_module.raw_outputscale[1].data.fill_(self.model.gp_layer.covar_module.raw_outputscale_constraint.inverse_transform(xtran_scale))
            # xtran_scale2 = tmp_out[0]*1            
            # self.model.gp_layer.covar_module.raw_outputscale[0].data.fill_(self.model.gp_layer.covar_module.raw_outputscale_constraint.inverse_transform(xtran_scale2))
            
            pred = self.model(test_x)  # Do precomputation
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.trace_mode():
            self.trace_model = torch.jit.trace(COVGPNNModelWrapper(self.model), test_x)
            
            

    def load_normalizing_consant(self, name ='normalizing'):        
        
        model = pickle_read(os.path.join(model_dir, name + '_normconstant.pkl'))        
        self.means_x = model['mean_sample'].cuda()
        if len(self.means_x.shape) > 1:
            self.means_x = self.means_x[:,:int(self.means_x.shape[1]/2)]
        self.means_y = model['mean_output'].cuda()
        self.stds_x = model['std_sample'].cuda()
        if len(self.stds_x.shape) > 1:
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
        pred = self.mean_and_cov_from_list(preds, M, track= track) 
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
        input_tmp[:,6] = tar_curv[:,2]
        input_tmp[:,7] = ego_state[:,1]
        input_tmp[:,8] = ego_state[:,2] 
        input_tmp[:,9] = ego_state[:,3]                                           
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
        roll_tar_curv = torch.tensor([target_state.lookahead.curvature[0], target_state.lookahead.curvature[1], target_state.lookahead.curvature[2]]).to('cuda')        
        roll_tar_curv = roll_tar_curv.repeat(M,1)
        roll_ego_state = torch.tensor([ego_state.p.s, ego_state.p.x_tran, ego_state.p.e_psi, ego_state.v.v_long]).to('cuda')
        roll_ego_state = roll_ego_state.repeat(M,1)

        horizon = len(ego_prediction.s)    
        # start_time = time.time()
        for i in range(horizon-1):         
            # gp_start_time = time.time()  
            roll_input = self.insert_to_end(roll_input, roll_tar_state, roll_tar_curv, roll_ego_state, track)                      
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                if self.model_name == 'naiveGP':
                    tmp_input = roll_input[:,:,-1]
                else:
                    tmp_input = roll_input
                if self.load_trace:
                    mean, stddev = self.trace_model(self.standardize(tmp_input))
                else:
                    pred_delta_dist = self.model(self.standardize(tmp_input))
                    mean = pred_delta_dist.mean
                    stddev = pred_delta_dist.stddev
                # pred_delta_dist = self.model(roll_input)            
                # print(stddev.cpu().numpy())
                
                tmp_delta = torch.distributions.Normal(mean, stddev).sample()            
                    
                pred_delta = self.outputToReal(tmp_delta)

            roll_tar_state[:,0] += pred_delta[:,0]
            roll_tar_state[:,0] = torch_wrap_s(roll_tar_state[:,0], track.track_length/2.0)
            roll_tar_state[:,1] += pred_delta[:,1]
            roll_tar_state[:,2] += pred_delta[:,2]
            roll_tar_state[:,3] += pred_delta[:,3]
            roll_tar_curv[:,0] = get_curvature_from_keypts_torch(pred_delta[:,0].clone().detach(),track)
            roll_tar_curv[:,1] = get_curvature_from_keypts_torch(pred_delta[:,0].clone().detach()+target_state.lookahead.dl*1,track)                        
            roll_tar_curv[:,2] = get_curvature_from_keypts_torch(pred_delta[:,0].clone().detach()+target_state.lookahead.dl*2,track)                        
            roll_ego_state[:,0] = ego_prediction.s[i+1]
            roll_ego_state[:,0] = torch_wrap_s(roll_ego_state[:,0], track.track_length/2.0)
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





  
    
 

        
