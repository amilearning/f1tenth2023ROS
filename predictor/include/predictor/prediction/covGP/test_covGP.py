import math
import tqdm
import torch
import gpytorch
from matplotlib import pyplot as plt
import urllib.request
import os
from scipy.io import loadmat
from math import floor
from torch.utils.tensorboard import SummaryWriter

import pickle
def pickle_write(data, path):
    dbfile = open(path, 'wb')
    pickle.dump(data, dbfile)
    dbfile.close()


def pickle_read(path):
    print("path = "+ str(path))
    dbfile = open(path, 'rb')
    data = pickle.load(dbfile)
    dbfile.close()
    return data


writer = SummaryWriter()
X, y = torch.randn(2000, 3), torch.randn(2000)

a_input_net = torch.nn.Sequential(torch.nn.Linear(3,5),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(5,3))

b_input_net = torch.nn.Sequential(torch.nn.Linear(3,5),
                                    torch.nn.LeakyReLU(),
                                    torch.nn.Linear(5,3))

c_input_net = torch.nn.Sequential(torch.nn.Linear(3,1),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(1,3))
eval_x = torch.randn(500,3)+2.0
eval_x = c_input_net(eval_x).detach().clone()


Xa = a_input_net(X[:1000,:])
Xb = b_input_net(X[1000:,:]) + 0.5
X = torch.vstack([Xa,Xb])
policy_net = torch.nn.Sequential(torch.nn.Linear(3,5),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(5,1))


y = policy_net(X)

X = X.detach().clone()
y = y.detach().clone()
# plt.plot(y.detach().numpy())
# plt.show()


train_n = int(floor(0.8 * len(X)))
train_x = X[:train_n, :]
train_y = y[:train_n]

test_x = X[train_n:, :]
test_y = y[train_n:]

if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()

data_dim = train_x.size(-1)

class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 10))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(10, 30))        
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(30, 3))



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
        )
        
              
    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    


        
class COVGPNNModel(gpytorch.Module):        
    def __init__(
        self):
       
        super(COVGPNNModel, self).__init__()        
        self.feature_extractor = LargeFeatureExtractor().cuda()
        self.gplayer = CovSparseGP(10,3,1).cuda()

    def get_latent(self,x):
        return self.feature_extractor(x)
    
    def forward(self, x):   
        latent= self.feature_extractor(x) 
        pred = self.gplayer(latent)
        return pred


# likelihood = gpytorch.likelihoods.GaussianLikelihood()
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=1) 
model = COVGPNNModel()


model_to_save = dict()
model_to_save['model'] = model
model_to_save['likelihood'] = likelihood
model_dir = '/home/racepc'

# readmodel = pickle_read(os.path.join(model_dir, 'testgp' + '.pkl'))
# model = readmodel['model']
# likelihood = readmodel['likelihood']

if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()

training_iterations = 10000

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
# optimizer = torch.optim.Adam([
#     {'params': model.feature_extractor.parameters()},
#     {'params': model.covar_module.parameters()},
#     {'params': model.mean_module.parameters()},
#     {'params': model.likelihood.parameters()},
# ], lr=0.01)
optimizer_nn = torch.optim.Adam([{'params': model.feature_extractor.parameters(), 'lr': 0.01, 'weight_decay':1e-4}
                                ], lr=0.01)

optimizer = torch.optim.Adam([{'params': model.feature_extractor.parameters(), 'lr': 0.01},                                            
                                {'params': model.gplayer.hyperparameters(), 'lr': 0.01},
                                {'params': model.gplayer.variational_parameters()},
                                {'params': likelihood.parameters()},
                                ], lr=0.01)

optimizer_gp = torch.optim.Adam([{'params': model.gplayer.hyperparameters(), 'lr': 0.01},
                                {'params': model.gplayer.variational_parameters()},
                                {'params': likelihood.parameters()},
                                ], lr=0.01)


# "Loss" for GPs - the marginal log likelihood
mseloss = torch.nn.MSELoss()
mll = gpytorch.mlls.VariationalELBO(likelihood, model.gplayer, num_data=len(y))
# def train():
iterator = tqdm.tqdm(range(training_iterations))
best_valid_loss = 0
best_epoch = 0
no_progress_count = 0
for i in iterator:
    torch.cuda.empty_cache()  
    # Zero backprop gradients
    optimizer.zero_grad()
    optimizer_nn.zero_grad()
    optimizer_gp.zero_grad()
    # Get output from model        
    output = model(train_x)
    
    latent_x = model.get_latent(train_x)
    #################################
    yinvKy_loss = 0
    jitter = torch.eye(train_y.shape[0]).cuda()*1e-11
    cov = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5)).to("cuda").double()                                                
    cov.base_kernel.lengthscale = 10.0 # (model.gplayer.covar_module.base_kernel.lengthscale[0])                   
    latent_dist= cov.base_kernel(latent_x,latent_x).evaluate()
    
    latent_std = torch.std(latent_x)
    latent_max = torch.max(latent_x)
    latent_min = torch.min(latent_x)
    latent_mean = torch.mean(latent_x)
    writer.add_scalar("stat/latent_std", latent_std, i)                    
    writer.add_scalar("stat/latent_max", latent_max, i)                    
    writer.add_scalar("stat/latent_min", latent_min, i)                    
    writer.add_scalar("stat/latent_mean", latent_mean, i)                    
    writer.add_scalar("stat/lengthscale_cov", model.gplayer.covar_module.base_kernel.lengthscale[0], i)     

    outcov = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5)).to("cuda").double()                                                
    outcov.base_kernel.lengthscale =  0.1 #  (model.gplayer.covar_module.base_kernel.lengthscale[0])        
    out_dist = outcov.base_kernel(train_y, train_y).evaluate()
    yinvKy = (train_y.T @ torch.cholesky_inverse(latent_dist + jitter).float() @ train_y)
    yinvKy_loss += yinvKy[0][0]/model.gplayer.covar_module.base_kernel.batch_shape[0]  # mseloss(latent_dist,out_dist)                    
    writer.add_scalar("loss/yinvKy_loss", yinvKy_loss, i)                    
    
    cov_loss = mseloss(latent_dist,out_dist)
    writer.add_scalar("loss/cov_loss", cov_loss.item(), i)                    
    # cos = cosine_loss(latent_dist, out_dist)
     
    variation_loss = -mll(output, train_y)
    writer.add_scalar("loss/variational_loss", variation_loss.item(), i)                    
    # Calc loss and backprop derivatives
    
    model_all = False
    if model_all:

        loss = variation_loss 
        writer.add_scalar("loss/total_loss", loss.item(), i)                    
        loss.backward()
        optimizer.step()
        
    else:
        # iterator.set_postfix(loss=loss.item())
        if i < 2000:
            loss = cov_loss
            writer.add_scalar("loss/total_loss", loss.item(), i)                    
            loss.backward()
            optimizer_nn.step()
            no_progress_count = 0
        else:
            loss = variation_loss 
            writer.add_scalar("loss/total_loss", loss.item(), i)                    
            loss.backward()
            # optimizer.step()
            optimizer_gp.step()
        
    if i % 50 ==0:
        optimizer.zero_grad()
        optimizer_nn.zero_grad()
        optimizer_gp.zero_grad()
        
        test_out = model(test_x)
        loss = -mll(test_out,test_y)
        valid_loss = loss.item()
        iterator.set_postfix(loss=valid_loss)
        writer.add_scalar("loss/valid_loss", valid_loss, i)                    
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            
            no_progress_count = 0
            best_epoch = i
        else:
            no_progress_count +=1
            print("no_progress_count = " + str(no_progress_count))
            if no_progress_count > 20:
                print("best_valid_loss = " + str(best_valid_loss) + ", at epoch = " + str(best_epoch))
                break


model.eval()
likelihood.eval()


with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
    preds = model(test_x)
    eval_pred= model(eval_x.cuda())        
print('test_data_COV: {}'.format(torch.mean(torch.max(preds.variance))))
print('New_data COV: {}'.format(torch.mean(torch.max(eval_pred.variance))))
print('Test MAE: {}'.format(torch.mean(torch.abs(preds.mean - test_y))))



model_to_save = dict()
model_to_save['model'] = model
model_to_save['likelihood'] = likelihood
model_dir = '/home/racepc'

readmodel = pickle_read(os.path.join(model_dir, 'testgp' + '.pkl'))
model = readmodel['model']
likelihood = readmodel['likelihood']

pickle_write(model_to_save, os.path.join(model_dir, 'testgp' + '.pkl'))


# MAE best 0.00025