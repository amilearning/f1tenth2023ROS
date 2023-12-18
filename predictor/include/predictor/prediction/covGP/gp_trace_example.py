import torch
import torch.nn as nn
import os 
import math
import numpy as np
import gpytorch
import time

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
    def __init__(self):
        super(CNNModel, self).__init__()
        
        self.input_dim = 9    
        self.output_dim = 8
        self.n_time_step = 10
        
        self.seq_conv = nn.Sequential(
        nn.Conv1d(in_channels=self.input_dim, out_channels=8, kernel_size=3),        
        nn.LeakyReLU(),        
        nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3),        
        nn.LeakyReLU(),        
        nn.Conv1d(in_channels=8, out_channels=1, kernel_size=3)
        ) 
        
        self.auc_conv_out_size = self._get_conv_out_size(self.seq_conv,self.input_dim,self.n_time_step)        
        
        self.encoder_fc = nn.Sequential(
                nn.Linear(self.auc_conv_out_size, 12),        
                nn.LeakyReLU(),                                    
                nn.Linear(12, 8),        
                nn.LeakyReLU(),                                    
                nn.Linear(8, self.output_dim)                               
        )
     
       

    def _get_conv_out_size(self, model, input_dim, seq_dim):
        # dummy_input = torch.randn(1, input_dim, seq_dim, requires_grad=False).to(self.gpu_id).float()         
        dummy_input = torch.randn(1, input_dim, seq_dim, requires_grad=False)
        conv_output = model(dummy_input)
        return conv_output.view(-1).size(0)
    
  
    def forward(self, x):       
        x = self.seq_conv(x)
        x = self.encoder_fc(x)
        return x
        


class COVGPNNModel(gpytorch.Module):        
    def __init__(
        self):
       
        super(COVGPNNModel, self).__init__()        
        
        self.nn_input_dim = 9     
        self.n_time_step = 10
        self.gp_input_dim = 8
        self.gp_output_dim =  4
        self.seq_len = 10
        inducing_points = 100
        
        self.covnn = CNNModel()                
        self.gp_layer = CovSparseGP(inducing_points_num=inducing_points,
                                                        input_dim=self.gp_input_dim,
                                                        num_tasks=self.gp_output_dim)  # Independent        
        
    def forward(self, input_data, train= False):    
        # current vehicle state, pred_action , RGB-D normalized image (4 channel)        
        latent_x = self.covnn(input_data)
        pred = self.gp_layer(latent_x.squeeze())
         
            
        return pred

    
class MeanVarModelWrapper(torch.nn.Module):
    def __init__(self, gp):
        super().__init__()
        self.gp = gp
    
    def forward(self, x):
        output_dist = self.gp(x)
        return output_dist.mean, output_dist.variance

# correlation_coefficients = []
# for i in range(len(a)):
#     correlation_coefficient = np.cov([a[i], b[i]])
#     correlation_coefficients.append(correlation_coefficient)
model = COVGPNNModel().cuda()

with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.trace_mode():
    model.eval()
    test_x = torch.randn(50,9,10).cuda()
    pred = model(test_x)  # Do precomputation

with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.trace_mode():
    traced_model = torch.jit.trace(MeanVarModelWrapper(model), test_x)
    # net = jit.load('model.zip')
for j in range(10):
    start_time = time.time()

    for i  in range(10):
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.trace_mode():
            # output = model(test_x)
            traced_mean, traced_var = traced_model(test_x)
    end_time = time.time()
    execution_time = end_time - start_time    
    print(f"Total execution time for {j} iterations: {execution_time} seconds")

    # print(torch.norm(traced_mean - pred.mean))
    # print(torch.norm(traced_var - pred.variance))

traced_model.save('traced_exact_gp.pt')
