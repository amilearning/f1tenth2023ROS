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

import random
random.seed(42)

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
X, y = torch.linspace(0,10,100), torch.randn(100)

y1 = torch.sigmoid(X) + torch.randn(X.shape[0])*0.02
y2 = torch.linspace(0,0.1,100)+1 + torch.randn(X.shape[0])*0.05
y3 = -torch.sigmoid(X)+1+.5 + torch.randn(X.shape[0])*0.02

y = torch.hstack([y1,y2,y3]).unsqueeze(1)   

y = y[::3]


train_x, test_x = X[::2], X[1::2]
train_y, test_y = y[::2], y[1::2]
train_x = train_x.unsqueeze(1)
test_x = test_x.unsqueeze(1)
# plt.plot(train_x,train_y,'r')
# plt.plot(test_x,test_y,'b')
# plt.show()
if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()

data_dim = train_x.size(-1)

class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 5))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(5, 5))        
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(5, 1))



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
        self.feature_extractor = LargeFeatureExtractor()
        self.gplayer = CovSparseGP(100,1,1)
        
        # self.incov =gpytorch.kernels.MaternKernel(nu=1.5)#.to("cuda").double()                                                
        # self.outcov =gpytorch.kernels.MaternKernel(nu=1.5)#.to("cuda").double()                                                


        self.incov = gpytorch.kernels.MaternKernel(nu=1.5)

        self.outcov = gpytorch.kernels.MaternKernel(nu=1.5) 


    def get_latent(self,x):
        return self.feature_extractor(x)
    
    def forward(self, x, train = False):   
        latent= self.feature_extractor(x) 
        pred = self.gplayer(latent)
        return pred


# likelihood = gpytorch.likelihoods.GaussianLikelihood()
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=1) 
model = COVGPNNModel().cuda()


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

training_iterations = 2000

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
# optimizer_nn = torch.optim.Adam([{'params': model.feature_extractor.parameters(), 'lr': 0.01, 'weight_decay':1e-4}
#                                 ], lr=0.01)


optimizer = torch.optim.Adam([{'params': model.feature_extractor.parameters(), 'lr': 0.1},                                            
                                {'params': model.gplayer.hyperparameters(), 'lr': 0.01},
                                {'params': model.incov.parameters(), 'lr': 0.02 },
                                {'params': model.outcov.parameters(), 'lr': 0.02},
                                {'params': model.gplayer.variational_parameters()},
                                {'params': likelihood.parameters()},
                                ], lr=0.01)

optimizer_gp = torch.optim.Adam([{'params': model.gplayer.hyperparameters(), 'lr': 0.01},
                                {'params': model.gplayer.variational_parameters()},
                                {'params': likelihood.parameters()},
                                ], lr=0.01)


# "Loss" for GPs - the marginal log likelihood
mseloss = torch.nn.MSELoss()
mll = gpytorch.mlls.VariationalELBO(likelihood, model.gplayer, num_data=len(train_y))
# def train():
iterator = tqdm.tqdm(range(training_iterations))
best_valid_loss = 0
best_epoch = 0
no_progress_count = 0
relu = torch.nn.ReLU()
for i in iterator:
    torch.cuda.empty_cache()  
    # Zero backprop gradients
    optimizer.zero_grad()
    # optimizer_nn.zero_grad()
    optimizer_gp.zero_grad()
    # Get output from model        
    output = model(train_x)
    latetn_x = model.get_latent(train_x)
    latent_dist = model.incov[0](latetn_x,latetn_x)
    out_dist = model.outcov[0](train_y,train_y)
    out_dist = out_dist.evaluate()
    latent_dist = latent_dist.evaluate()
    latent_std = torch.std(latetn_x)    
    stddev_loss = torch.relu(latent_std-2)
    
    sensitivity_loss =  torch.log((model.outcov[0].lengthscale+1e-9)/(model.incov[0].lengthscale + 1e-9))
    
    # torch.mean(abs(torch.eye(latent_dist.shape[0]).cuda()-out_dist) * (torch.eye(latent_dist.shape[0]).cuda()-latent_dist))
    dist_loss = mseloss(out_dist, latent_dist) 
    #################################
    yinvKy_loss = 0
    jitter = torch.eye(train_y.shape[0]).cuda()*1e-11    
 

    writer.add_scalar("stat/latent_std", latent_std, i)                    
    
    writer.add_scalar("stat/lengthscale_incov", model.incov[0].lengthscale.item(), i)     
    writer.add_scalar("stat/lengthscale_outcov", model.outcov[0].lengthscale.item(), i)     
    writer.add_scalar("stat/lengthscale_model", model.gplayer.covar_module.base_kernel.lengthscale[0][0][0], i)     

    yinvKy = (train_y.T @ torch.cholesky_inverse( jitter).float() @ train_y)
    yinvKy_loss += yinvKy[0][0]/model.gplayer.covar_module.base_kernel.batch_shape[0]  # mseloss(latent_dist,out_dist)                    
    writer.add_scalar("loss/yinvKy_loss", yinvKy_loss, i)                    
    
    # cov_loss = mseloss(latent_dist,out_dist)*10.0
    writer.add_scalar("loss/dist_loss", dist_loss.item(), i)                    
    writer.add_scalar("loss/sensitivity_loss", sensitivity_loss.item(), i)         
    writer.add_scalar("loss/std_loss", stddev_loss.item(), i)   
    # cos = cosine_loss(latent_dist, out_dist)
     
    variation_loss = -mll(output, train_y)
    writer.add_scalar("loss/variational_loss", variation_loss.item(), i)                    
    # Calc loss and backprop derivatives
    
    

    loss = variation_loss # + sensitivity_loss*1e-1 + dist_loss + stddev_loss*1e-2
    writer.add_scalar("loss/total_loss", loss.item(), i)                    
    loss.backward()
    optimizer.step()

    if i % 50 ==0:
        optimizer.zero_grad()
        # optimizer_nn.zero_grad()
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
            # if i > 5000:
            if no_progress_count > 20:
                print("best_valid_loss = " + str(best_valid_loss) + ", at epoch = " + str(best_epoch))                    
                break


model.eval()
likelihood.eval()


with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
    preds = model(test_x)
    eval_pred= model(test_x.cuda())            
    latent_x_c = (model.get_latent(test_x.cuda())).detach().cpu().numpy()
    
    
    inducing_points= model.gplayer.variational_strategy.base_variational_strategy.inducing_points.detach().cpu().numpy().squeeze()


print('test_data_COV: {}'.format(torch.mean(torch.max(preds.variance))))
print('New_data COV: {}'.format(torch.mean(torch.max(eval_pred.variance))))
print('Test MAE: {}'.format(torch.mean(torch.abs(preds.mean - test_y))))

test_x = test_x.detach().cpu().numpy()

# plt.plot(test_x,latent_x_c,'r')
# plt.show()


with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = eval_pred.confidence_region()
    # Plot training data as black stars
    # ax.plot(train_x.cpu().numpy(), train_y.cpu().numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(test_x, eval_pred.mean.cpu().numpy(), 'b*')
    ax.plot(test_x, test_y.cpu().numpy(), 'r*')
    ax.plot(test_x, latent_x_c, 'g*')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.flatten(), lower.cpu().numpy().flatten(), upper.cpu().numpy().flatten(), alpha=0.5)    
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    plt.show()



fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(inducing_points[:, 0], inducing_points[:, 1], inducing_points[:, 2], c='black', marker='o', s=10)  # s is the size
print('inducing_points:', inducing_points.min(axis=0), inducing_points.max(axis=0))
plt.show()

model_to_save = dict()
model_to_save['model'] = model
model_to_save['likelihood'] = likelihood
model_dir = '/home/racepc'

readmodel = pickle_read(os.path.join(model_dir, 'testgp' + '.pkl'))
model = readmodel['model']
likelihood = readmodel['likelihood']

pickle_write(model_to_save, os.path.join(model_dir, 'testgp' + '.pkl'))


# MAE best 0.00025
