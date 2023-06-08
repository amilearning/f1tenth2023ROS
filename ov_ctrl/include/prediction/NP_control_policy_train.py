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





VAEwriter = SummaryWriter()
data_index = "trainx"
index_number = 0
state_model = "fullstate"

pickle_data = pickle_read(os.path.join(racingdata_dir, data_index+ '_'+str(index_number) + '_'+ state_model + '.pkl'))

###########################################################################################################################################
# full state [s_ego, epsi_ego, etran_ego, vlon_ego, vlat_ego, s_tar, epsi_tar, etran_tar, vlon_tar, vlat_tar, lookahead[0,1,2], tar_accel, tar_delta, tar_Qxref, tar_Qtheta] #                        
# partial state [s_ego, etran_ego, s_tar, etran_tar, lookahead[0,1,2], tar_accel, tar_delta, tar_Qxref, tar_Qtheta] #                        
###########################################################################################################################################

np_data = np.asarray(pickle_data)

s_diff = np_data[:,:,5]-np_data[:,:,0]
# input state : (s_tar-e_ego), ey_ego, epsi_ego, cur_ego, ey_tar, epsie_tar, cur_tar
clip_mins = np.array([-10,      -2.0,    -np.pi,     -1,   -2.0,    -np.pi,     -1])
clip_maxs = -1*clip_mins
ey_ego = np_data[:,:,2]
epsi_ego = np_data[:,:,1]
cur_ego = np_data[:,:,10]
ey_tar = np_data[:,:,7]
epsi_tar = np_data[:,:,6]
cur_tar = np_data[:,:,10]
input_data = np.stack([s_diff,ey_ego,epsi_ego,cur_ego,ey_tar,epsi_tar,cur_tar],2)
cliped_np_data = np.clip(input_data, clip_mins, clip_maxs)


if state_model == 'fullstate':
    s_diff = np_data[:,:,5]-np_data[:,:,0]
    # input state : (s_tar-e_ego), ey_ego, epsi_ego, cur_ego, ey_tar, epsie_tar, cur_tar
    clip_mins = np.array([-10,      -2.0,    -np.pi,     -1,   -2.0,    -np.pi,     -1])
    clip_maxs = -1*clip_mins
    ey_ego = np_data[:,:,2]
    epsi_ego = np_data[:,:,1]
    cur_ego = np_data[:,:,10]
    ey_tar = np_data[:,:,7]
    epsi_tar = np_data[:,:,6]
    cur_tar = np_data[:,:,10]
    input_data = np.stack([s_diff,ey_ego,epsi_ego,cur_ego,ey_tar,epsi_tar,cur_tar],2)
    cliped_np_data = np.clip(input_data, clip_mins, clip_maxs)

    # clip_mins = np.array([-10, -np.pi, -1*width,    -2.0,       -1.0,     -np.pi,  -1*width,  -2.0,     -1.0,      -1, -1, -1,     -3 , -0.5 , 0, 0])
    # clip_maxs = -1*clip_mins
    # cliped_np_data = np.clip(np_data[:,0:14], clip_mins, clip_maxs)
elif state_model == 'partial': # partial state 
    np_data[:,:,0] = np_data[:,:,2] - np_data[:,:,0] # --> (s_tar - s_ego)
    np_data = np_data[:,:,[0,1,3,4,5,6,7,8,9,10]] # remove s_tar 
    # partial state [(s_tar - s_ego), etran_ego, etran_tar, lookahead[0,1,2], tar_accel, tar_delta, tar_Qxref, tar_Qtheta] #                        
    clip_mins = np.array([-10, -1*width, -1*width, -1, -1, -1, -3, -0.5 ])
    clip_maxs = -1*clip_mins
    cliped_np_data = np.clip(np_data[:,:,0:8], np.tile(clip_mins,[np_data.shape[1],1]), np.tile(clip_maxs,[np_data.shape[1],1]))


normalized_data = (cliped_np_data/clip_maxs)

dataset = torch.Tensor(normalized_data)
dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)
validation_size = int(dataset_size * 0.1)
test_size = dataset_size - train_size - validation_size

train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size, test_size])

print(f"Training Data Size : {len(train_dataset)}")
print(f"Validation Data Size : {len(validation_dataset)}")
print(f"Testing Data Size : {len(test_dataset)}")



args =  {
            "batch_size": 512,
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
validation_dataloader = DataLoader(validation_dataset, batch_size=args['batch_size'], shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False)

model = LSTMVAE(args)
model.to(torch.device("cuda"))

VAEtrain(args,model,train_dataloader,validation_dataloader,VAEwriter)

id_ = secrets.token_hex(nbytes=4)
save_dir = model_dir+f"lstmvae{id_}.model"
print("model has been saved in "+ save_dir)
torch.save(model.state_dict(), save_dir )

model_to_load = LSTMVAE(args)
model_to_load.to(torch.device("cuda"))
model_to_load.load_state_dict(torch.load(model_dir+f"lstmvae{id_}.model"))
model_to_load.eval()

##########################################################################################
##########################################################################################
############################## End of training session ###################################
##########################################################################################
##########################################################################################

# Visualize Test data Inference 
for test_data_batch in enumerate(test_dataloader):
    x_data = test_data_batch[1][:,-1,:].to(torch.device("cuda"))


for i in range(len(test_dataset)):
    
    _data = test_data_batch[1][:,-1,:].to(torch.device("cuda"))
    y_data = _data[:,6:]

    x_data_ = _data[:,0:7]
    x_mean = torch.hstack([theta_mean_,x_data])

    

    test_data = torch.Tensor(test_dataset[i]).to(torch.device("cuda")).view(1,train_dataset[0].shape[0],train_dataset[0].shape[1])
    _, recon_data, info, theta_mean_, theta_logvar_ = model_to_load(test_data)
    x_mean = torch.hstack([r,x_data]) 


    with torch.no_grad():
        for idx in range(len(recon_data)):
            plt.plot(recon_data[idx,:,1].cpu(), recon_data[idx,:,2].cpu(), 'r.')
            plt.plot(test_data[idx,:,1].cpu(), test_data[idx,:,2].cpu(), 'g.')                        
            plt.axis([-width,width, -width, width])
            plt.pause(0.01)
            plt.clf()
plt.pause()

################################ Grouping training data for Gaussian process (x,theta)-> u ########################################
for test_data_batch in enumerate(test_dataloader):
    q,w,e,r,t = model_to_load(test_data_batch[1].to(torch.device("cuda")))
    
    x_data = test_data_batch[1][:,-1,:].to(torch.device("cuda"))
    x_mean = torch.hstack([r,x_data])

prior_theta_mean_vector = torch.zeros(2)
prior_theta_cov_mtx = torch.eye(2)*1e6
theta_dist = MultivariateNormal(prior_theta_mean_vector,prior_theta_cov_mtx)

# std = torch.exp(0.5 * logvar)

## Set prior and 


test_data = torch.Tensor(test_dataset[i]).to(torch.device("cuda")).view(1,train_dataset[0].shape[0],train_dataset[0].shape[1])
_, recon_data, info = model_to_load(test_data)


