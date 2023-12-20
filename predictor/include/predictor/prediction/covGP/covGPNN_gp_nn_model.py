"""   
 Software License Agreement (BSD License)
 Copyright (c) 2023 Ulsan National Institute of Science and Technology (UNIST)
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************** 
  @author: Hojin Lee <hojinlee@unist.ac.kr>, Sanghun Lee <sanghun17@unist.ac.kr>
  @date: September 10, 2023
  @copyright 2023 Ulsan National Institute of Science and Technology (UNIST)
  @brief: Torch version of util functions
"""

import torch
import torch.nn as nn
import os 
import math
import numpy as np
import gpytorch


def ff(x,k,s):
    return (x-k)/s+1
def rr(y,k,s):
    return (y-1)*s+k


class CovSparseGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points_num, input_dim, num_tasks):
        # Let's use a different set of inducing points for each task
        inducing_points = torch.rand(num_tasks, inducing_points_num, input_dim)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_tasks])
        )

        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
        )

        super().__init__(variational_strategy)


        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5, batch_shape=torch.Size([num_tasks])), # nu = 1.5
            batch_shape=torch.Size([num_tasks])
            # gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_tasks],
            #                                                   ard_num_dims=input_dim))
        )
              # Apply LogNormal prior to the lengthscale parameter
        
        # self.covar_module.base_kernel.lengthscale.prior = lengthscale_prior
        # self.covar_module.outputscale.prior = outputscale_prior


    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    


class CNNModel(nn.Module):    
    def __init__(self, args):
        super(CNNModel, self).__init__()
        
        self.input_dim = args['input_dim']        
        self.output_dim = args['latent_dim'] 
        self.n_time_step = args['n_time_step']        
        
        self.seq_conv = nn.Sequential(
        nn.Conv1d(in_channels=self.input_dim, out_channels=16, kernel_size=3),        
        nn.LeakyReLU(),        
         nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3),        
         nn.LeakyReLU(),        
         nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3),    
         nn.LeakyReLU(),       
        nn.Conv1d(in_channels=16, out_channels=6, kernel_size=3)     
        ) 
        a = torch.randn(1, self.input_dim,self.n_time_step, requires_grad=False)
        
        self.auc_conv_out_size = self._get_conv_out_size(self.seq_conv,self.input_dim,self.n_time_step)        
        
        self.encoder_fc = nn.Sequential(
                # nn.utils.spectral_norm(nn.Linear(self.auc_conv_out_size, 12)),        
                nn.Linear(self.auc_conv_out_size, 12),        
                nn.LeakyReLU(),                                    
                nn.Linear(12, 8),        
                nn.LeakyReLU(),                                    
                nn.Linear(8, self.output_dim)                               
        )
        self.latent_size = self.output_dim
        self.decoder_fc = nn.Sequential(
                nn.Linear(self.output_dim, 12),        
                nn.LeakyReLU(),                                    
                nn.Linear(12, 8),        
                nn.LeakyReLU(),                                    
                nn.Linear(8, self.auc_conv_out_size)                               
        )

        self.seq_deconv = nn.Sequential(
            nn.ConvTranspose1d(in_channels=6, out_channels=16, kernel_size=3),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=3),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=3),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=self.input_dim, kernel_size=3)
        )   
        
        # dummy_input = torch.randn(1, self.input_dim,self.n_time_step, requires_grad=False)
        # # dummy_ou_dim = self.seq_deconv(self.decoder_fc(self.encoder_fc(self.seq_conv(dummy_input)))).view(-1).size(0)
        # self.post_fc = nn.Sequential(
        #         nn.Linear(dummy_ou_dim, 12),        
        #         nn.LeakyReLU(),                                    
        #         nn.Linear(12, 12),        
        #         nn.LeakyReLU(),                          
        #         nn.Linear(12, self.input_dim *self.n_time_step)                               
        # )   
       

    def _get_conv_out_size(self, model, input_dim, seq_dim):
        # dummy_input = torch.randn(1, input_dim, seq_dim, requires_grad=False).to(self.gpu_id).float()         
        dummy_input = torch.randn(1, input_dim, seq_dim, requires_grad=False)
        conv_output = model(dummy_input)
        return conv_output.view(-1).size(0)
    
    def get_latent(self,x):
        x = self.seq_conv(x)
        self.seq_conv_shape1 = x.shape[1]
        self.seq_conv_shape2 = x.shape[2]
        x = self.encoder_fc(x.view(x.shape[0],-1))
        return x

    def forward(self, x):       
        latent = self.get_latent(x) 
        z = self.decoder_fc(latent)        
        y = self.seq_deconv(z.view(z.shape[0],self.seq_conv_shape1, self.seq_conv_shape2))
        # z = self.post_fc(y.view(y.shape[0],-1))
        # z = z.view(z.shape[0],x.shape[1],x.shape[2])
        return y, latent
        


class COVGPNNModel(gpytorch.Module):        
    def __init__(
        self, args):
       
        super(COVGPNNModel, self).__init__()        
        self.args = args                
        self.nn_input_dim = args['input_dim']        
        self.n_time_step = args['n_time_step']     
        # self.gp_input_dim = args['latent_dim']+3 
        self.gp_output_dim =  args['gp_output_dim']        
        self.seq_len = args['n_time_step']
        inducing_points = args['inducing_points']
        
        self.covnn = CNNModel(args)                
        
        self.gp_input_dim = self.covnn.latent_size + 3
        self.gp_layer = CovSparseGP(inducing_points_num=inducing_points,
                                                        input_dim=self.gp_input_dim,
                                                        num_tasks=self.gp_output_dim)  # Independent        
        

    def outputToReal(self, batch_size, pred_dist):
        with torch.no_grad():            
            standardized_mean = pred_dist.mean.view(batch_size,-1,pred_dist.mean.shape[-1])
            standardized_stddev = pred_dist.stddev.view(batch_size,-1,pred_dist.mean.shape[-1])
            return standardized_mean, standardized_stddev
            
            
    def get_hidden(self,input_data):
        aug_pred = self.covnn.get_latent(input_data)        
        return aug_pred

    
    def compute_coeef(self,input_data):
    # input_data=torch.tensor([ delta_s,                        
    #                     tar_st.p.x_tran,
    #                     tar_st.p.e_psi,
    #                     tar_st.v.v_long,
    #                     tar_st.lookahead.curvature[0],
    #                     tar_st.lookahead.curvature[2],
    #                     ego_st.p.x_tran,
    #                     ego_st.p.e_psi, 
    #                     ego_st.v.v_long                       
    #                     ])
        s_diff = input_data[:,0,:]
        x_tran_diff = input_data[:,1,:]-input_data[:,6,:]
        e_psi_diff = input_data[:,2,:]-input_data[:,7,:]
        v_long_diff = input_data[:,3,:]-input_data[:,8,:]        
        input_corrcoefs = []
        input_corrcoefs.append(torch.corrcoef(s_diff))        
        input_corrcoefs.append(torch.corrcoef(x_tran_diff))
        input_corrcoefs.append(torch.corrcoef(e_psi_diff))
        input_corrcoefs.append(torch.corrcoef(v_long_diff))

        # or torch.cov        
        # input_corrcoefs.append(torch.cov(s_diff))        
        # input_corrcoefs.append(torch.cov(x_tran_diff))
        # input_corrcoefs.append(torch.cov(e_psi_diff))
        # input_corrcoefs.append(torch.cov(v_long_diff))


        input_corrcoefs = torch.stack(input_corrcoefs)
        return input_corrcoefs
        

    def forward(self, input_data, train= False):    
        # current vehicle state, pred_action , RGB-D normalized image (4 channel)        
        if train:
            recons, latent_x = self.covnn(input_data)
        else:
            latent_x = self.covnn.get_latent(input_data)
        
        if latent_x.shape[0] == 1:
            gp_input = torch.hstack([ latent_x.reshape(1,-1) , input_data[:,1:4,-1]])
        else:
            gp_input = torch.hstack([latent_x.view(input_data.shape[0],-1), input_data[:,1:4,-1]])
          
        # exp_dir_pred = dir_pred.reshape(dir_pred.shape[0],-1,5)        
        # # remap to [batch , sqeucen, feature]  -> [batch x sqeucen, feature + 1 (temporal encoding)]                        
        if train:
            pred = self.gp_layer(gp_input)
            input_covs = self.compute_coeef(input_data)
            output_covs = []
        # F.mse_loss(recons, input)  
            for i in range(self.gp_layer.covar_module.base_kernel.batch_shape[0]):
                


                cov = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5)).to("cuda")
                cov.outputscale = self.gp_layer.covar_module.outputscale[i]
                cov.base_kernel.lengthscale =  (self.gp_layer.covar_module.base_kernel.lengthscale[i])
                cout = cov(gp_input,gp_input).evaluate().clone()
                # cout = torch.log(cout)
                output_covs.append(cout)
                input_covs[i,:,:] = abs(self.gp_layer.covar_module.outputscale[i].item() * input_covs[i,:,:])

            output_covs = torch.stack(output_covs)
            
            return pred, recons, input_covs, output_covs

        # q = output.covariance_matrix.detach().cpu().numpy()
        
        else:
            pred = self.gp_layer(gp_input)
            return pred
                



    
class COVGPNNModelWrapper(torch.nn.Module):
    def __init__(self, gp):
        super().__init__()
        self.gp = gp
    
    def forward(self, x):
        output_dist = self.gp(x)
        return output_dist.mean, output_dist.stddev
    

# correlation_coefficients = []
# for i in range(len(a)):
#     correlation_coefficient = np.cov([a[i], b[i]])
#     correlation_coefficients.append(correlation_coefficient)
