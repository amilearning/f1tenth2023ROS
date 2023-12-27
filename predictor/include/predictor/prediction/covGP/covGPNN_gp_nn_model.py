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
from einops import reduce, rearrange
import gpytorch
import torch.nn.functional as F

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
    


class Chomp1d(torch.nn.Module):
    """
    Removes the last elements of a time series.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
    is the number of elements to remove.

    @param chomp_size Number of elements to remove.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size ==0:
          return x
        return x[:, :, :-self.chomp_size]
    


class CausalConvolutionBlock(torch.nn.Module):
    """
    Causal convolution block, composed sequentially of two causal convolutions
    (with leaky ReLU activation functions), and a parallel residual connection.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).

    @param in_channels Number of input channels.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param dilation Dilation parameter of non-residual convolutions.
    @param final Disables, if True, the last activation function.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 final=False):
        super(CausalConvolutionBlock, self).__init__()
        
        # Computes left padding so that the applied convolutions are causal
        self.padding = (kernel_size - 1) * dilation
        padding = self.padding
        # First causal convolution
        self.conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
        # self.conv1 = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        # The truncation makes the convolution causal
        self.chomp1 = Chomp1d(padding)
        self.dropout1 = nn.Dropout(0.1)

        # Second causal convolution
        self.conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
        # self.conv2 = torch.nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.dropout2 = nn.Dropout(0.1)

        # Residual connection
        self.upordownsample = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            in_channels, out_channels, 1
        )) if in_channels != out_channels else None

        # Final activation function
        self.relu = None
        
    def forward(self, x):
       
        out_causal=self.conv1(x)
        out_causal=self.chomp1(out_causal)        
        out_causal=self.dropout1(F.gelu(out_causal))
        out_causal=self.conv2(out_causal)
        out_causal=self.chomp2(out_causal)
        out_causal=self.dropout2(F.gelu(out_causal))
        res = x if self.upordownsample is None else self.upordownsample(x)
        
        
        if self.relu is None:
            x = out_causal + res
        else:
            x= self.relu(out_causal + res)
        
        return x

class LinearPred(torch.nn.Module):
    def __init__(self,input_dims,input_len,output_dims,output_len):
        super(LinearPred, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = 'cpu'
        self.Wk_wl = nn.Linear(input_len, output_len).to(self.device)
        self.Wk_wl2 = nn.Linear(input_dims, output_dims).to(self.device)        
        self.relu = nn.ReLU(inplace=False)
      
    def forward(self,x):
        x_pred = self.relu(self.Wk_wl(x)).transpose(1,2)
        x_pred2 = self.Wk_wl2(x_pred)
       
        return x_pred2

class CausalCNNEncoder(torch.nn.Module): 
    """
    Encoder of a time series using a causal CNN: the computed representation is
    the output of a fully connected layer applied to the output of an adaptive
    max pooling layer applied on top of the causal CNN, which reduces the
    length of the time series to a fixed size.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`).

    @param in_channels Number of input channels.
    @param channels Number of channels manipulated in the causal CNN.
    @param depth Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the
           causal CNN is reduced.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """
    def __init__(self, 
                 in_channels,
                 reduced_size,
                 component_dims, 
                 kernel_list=[1,2, 4],
                 ):
        super(CausalCNNEncoder, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = 'cpu'
        self.input_fc = CausalConvolutionBlock(in_channels, reduced_size, 1, 1)
        
        self.kernel_list = kernel_list
        self.multi_cnn = nn.ModuleList(
            [nn.Conv1d(reduced_size, component_dims, k, padding=k-1) for k in kernel_list]
        )
        self.repr_dropout = torch.nn.Dropout(p=0.1)
        
        

    def print_para(self):
        
        return list(self.multi_cnn.parameters())[0].clone()    
        
    def forward(self, x_h, x_f = None, mask = None, train = True):

        # nan_mask_h = ~x_h.isnan().any(axis=-1)
        # x_h[~nan_mask_h] = 0
        
        # x_h = x_h.transpose(2,1)
        x_h = self.input_fc(x_h)        
        
        if train and x_f is not None:
 
            x_f = self.input_fc(x_f)

            trend_h = []            
            trend_f = []
            # trend_h_weights = []
            # trend_f_weights = []

            for idx, mod in enumerate(self.multi_cnn):
              
                out_h = mod(x_h) # b d t
                out_f = mod(x_f)
                if self.kernel_list[idx] != 1:
                  out_h = out_h[..., :-(self.kernel_list[idx] - 1)]
                  out_f = out_f[..., :-(self.kernel_list[idx] - 1)]

                trend_h.append(out_h.transpose(1,2))  # b 1 t d                
                trend_f.append(out_f.transpose(1,2))  # b 1 t d
                # trend_h_weights.append(out_h.transpose(1, 2)[:,-1,:].unsqueeze(-1)) 
                # trend_f_weights.append(out_f.transpose(1, 2)[:,-1,:].unsqueeze(-1)) 
                

            trend_h = reduce(
              rearrange(trend_h, 'list b t d -> list b t d'),
              'list b t d -> b t d', 'mean'
            )

            trend_f = reduce(
              rearrange(trend_f, 'list b t d -> list b t d'),
              'list b t d -> b t d', 'mean'
            )

            trend_h = self.repr_dropout(trend_h)
            latent_x = trend_h[:,-1,:]
          

            # print(rand_idx)
            # self.predictor =LinearPred(component_dims,1,component_dims, int(reduced_size/2))
            # fcst_future_embeds = self.predictor(latent_x.unsqueeze(-1))
           
            return latent_x, trend_f.detach()

        else:
            trend_h = []
            # trend_h_weights = []

            for idx, mod in enumerate(self.multi_cnn):
              
                out_h = mod(x_h)  # b d t

                if self.kernel_list[idx] != 1:
                  out_h = out_h[..., :-(self.kernel_list[idx] - 1)]
                trend_h.append(out_h.transpose(1,2))  # b 1 t d
                # trend_h_weights.append(out_h.transpose(1, 2)[:,-1,:].unsqueeze(-1)) 

            trend_h = reduce(
              rearrange(trend_h, 'list b t d -> list b t d'),
              'list b t d -> b t d', 'mean'
            )

            trend_h = self.repr_dropout(trend_h)
            latent_x = trend_h[:,-1,:]
            return latent_x, None
        

        
class CNNModel(nn.Module):    
    def __init__(self, args):
        super(CNNModel, self).__init__()
        
        self.input_dim = args['input_dim']        
        self.output_dim = args['latent_dim'] 
        self.n_time_step = args['n_time_step']        
        kernel_list=[1,2, 4,8]        
        self.net = CausalCNNEncoder(in_channels = self.input_dim, 
                                reduced_size=60, 
                                component_dims = self.output_dim, 
                                kernel_list= kernel_list).cuda()
        
        x_h = torch.randn(2, self.input_dim,self.n_time_step, requires_grad=False).cuda()
        x_f = torch.randn(2, self.input_dim,self.n_time_step, requires_grad=False).cuda()
        latent_x, trend_f = self.net(x_h, x_f)
        self.latent_size = latent_x.shape[-1]
    


    def _get_conv_out_size(self, model, input_dim, seq_dim):
        # dummy_input = torch.randn(1, input_dim, seq_dim, requires_grad=False).to(self.gpu_id).float()         
        dummy_input = torch.randn(1, input_dim, seq_dim, requires_grad=False)
        conv_output = model(dummy_input)
        return conv_output.view(-1).size(0)
    
    # def get_latent(self,x):        
    #     latent_x, trend_f = self.net(x, None)               
    #     return latent_x

    def forward(self, x, x_f = None):       
        latent_x, trend_f = self.net(x, x_f)
        if x_f is None:
            return latent_x, None
        else:
            return latent_x, trend_f
            
        


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
        self.directGP = args['direct_gp']
        self.include_simts_loss = args['include_simts_loss']
        
        self.covnn = CNNModel(args)   
                              
        self.predictor =LinearPred(args['latent_dim'] ,1,args['latent_dim'] , int(self.seq_len))
    
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-3., 3.)

        if self.directGP:
            self.gp_input_dim =  self.nn_input_dim
        else: 
            # [ policy, dels, xtran, epsi, vx, cur1, cur2]  
            self.gp_input_dim = self.covnn.latent_size + 6

        self.gp_layer = CovSparseGP(inducing_points_num=inducing_points,
                                                        input_dim=self.gp_input_dim,
                                                        num_tasks=self.gp_output_dim)  # Independent        
        

    def outputToReal(self, batch_size, pred_dist):
        with torch.no_grad():            
            standardized_mean = pred_dist.mean.view(batch_size,-1,pred_dist.mean.shape[-1])
            standardized_stddev = pred_dist.stddev.view(batch_size,-1,pred_dist.mean.shape[-1])
            return standardized_mean, standardized_stddev
            
            
    def get_hidden(self,input_data):
        aug_pred, _ = self.covnn(input_data)        
        return aug_pred

    def forward(self, x_h, x_f = None, train= False, directGP = True):    
        # current vehicle state, pred_action , RGB-D normalized image (4 channel)        
        latent_x, trend_f = self.covnn(x_h, x_f)                

        
        if self.directGP:
            gp_input = x_h[:,0:,-1]
        else:
            latent_x = self.scale_to_bounds(latent_x).clone()        
            if latent_x.shape[0] == 1:
                gp_input = torch.hstack([ latent_x.reshape(1,-1) , x_h[:,0:6,-1]])
            else:
                gp_input = torch.hstack([latent_x.view(x_h.shape[0],-1), x_h[:,0:6,-1]])
        
        # exp_dir_pred = dir_pred.reshape(dir_pred.shape[0],-1,5)        
        # # remap to [batch , sqeucen, feature]  -> [batch x sqeucen, feature + 1 (temporal encoding)]                        
        if train:
            pred = self.gp_layer(gp_input)
            trend_f = trend_f.detach()            
            fcst_future_embeds = self.predictor(latent_x.unsqueeze(-1))            
            
            # output_covs = []
            # for i in range(self.gp_layer.covar_module.base_kernel.batch_shape[0]):
            #     cov = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5)).to("cuda")                                                
            #     cov.base_kernel.lengthscale =  (self.gp_layer.covar_module.base_kernel.lengthscale[i])                   
            #     cout = cov.base_kernel(fcst_future_embeds.view(fcst_future_embeds.shape[0],-1),trend_f.view(trend_f.shape[0],-1)).evaluate().clone()                
            #     # a = cov(fcst_future_embeds.view(fcst_future_embeds.shape[0],-1),fcst_future_embeds.view(trend_f.shape[0],-1)).evaluate().clone() /cov.base_kernel.lengthscale
            #     # cout = torch.log(cout)
            #     output_covs.append(cout.clone())                
            # output_covs = torch.stack(output_covs)
            
            return pred, trend_f.detach(), fcst_future_embeds
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
