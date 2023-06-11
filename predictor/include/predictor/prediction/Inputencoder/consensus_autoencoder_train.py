import secrets
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import os 
from tqdm import tqdm

from barcgp.h2h_configs import *
from barcgp.common.utils.file_utils import *
from matplotlib import pyplot as plt
from torch.distributions import Normal, MultivariateNormal
from torch.utils.data import DataLoader

from barcgp.prediction.encoder.encoderModel import LSTMAutomodel

from AutoEncoderTrain import Autotrain
from torch.utils.tensorboard import SummaryWriter


def Autotrain(args, model, train_loader, test_loader, writer):
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])

    ## interation setup
    epochs = tqdm(range(args['max_iter'] // len(train_loader) + 1))

    ## training
    count = 0
    for epoch in epochs:
        model.train()
        optimizer.zero_grad()
        train_iterator = tqdm(
            enumerate(train_loader), total=len(train_loader), desc="training"
        )

        for i, batch_data in train_iterator:

            if count > args['max_iter']:
                return model
            count += 1

            train_data = batch_data[:,:,:,0:-2].to(args['device'])

            ## reshape
            batch_size = train_data.size(0)
            # example_size = past_data.size(1)
            # image_size = past_data.size(1), past_data.size(2)
            # past_data = (
            #     past_data.view(batch_size, example_size, -1).float().to(args['device'])
            # )
            # future_data = future_data.view(batch_size, example_size, -1).float().to(args.device)
            # m_loss, x_hat, mean_consensus_loss
            mloss, recon_x, mean_consensus_loss = model(train_data)

            # Backward and optimize
            optimizer.zero_grad()
            mloss.mean().backward()
            optimizer.step()

            train_iterator.set_postfix({"train_loss": float(mloss.mean())})
            train_iterator.set_postfix({"consensus_loss": float(mean_consensus_loss.mean())})
        writer.add_scalar("train_loss", float(mloss.mean()), epoch)
        writer.add_scalar("consensus_loss", float(mean_consensus_loss.mean()), epoch)

        model.eval()
        eval_loss = 0
        consensus_loss = 0
        test_iterator = tqdm(
            enumerate(test_loader), total=len(test_loader), desc="testing"
        )

        with torch.no_grad():
            for i, batch_data in test_iterator:
                test_data = batch_data[:,:,:,0:-2].to(args['device'])

                ## reshape
                batch_size = test_data.size(0)
                # example_size = past_data.size(1)
                # past_data = (
                #     past_data.view(batch_size, example_size, -1).float().to(args['device'])
                # )
                # future_data = future_data.view(batch_size, example_size, -1).float().to(args.device)

                mloss, recon_x, mean_consensus_loss  = model(test_data)

                eval_loss += mloss.mean().item()
                consensus_loss +=mean_consensus_loss.mean().item()

                test_iterator.set_postfix({"eval_loss": float(mloss.mean())})
                test_iterator.set_postfix({"consensus_loss": float(mean_consensus_loss.mean())})

                # if i == 0:
                #     for idx in range(len(recon_x)):
                #         plt.plot(recon_x[idx,:,0].cpu(), recon_x[idx,:,0].cpu(), 'r.')
                #         plt.plot(past_data[idx,:,0].cpu(), past_data[idx,:,0].cpu(), 'g.')                        
                #         plt.pause(0.01)
                #     plt.clf()
                    # nhw_orig = past_data[0].view(example_size, image_size[0], -1)
                    # nhw_recon = recon_x[0].view(example_size, image_size[0], -1)                    
                    # writer.add_images(f"original{i}", nchw_orig, epoch)
                    # writer.add_images(f"reconstructed{i}", nchw_recon, epoch)

        eval_loss = eval_loss / len(test_loader)
        writer.add_scalar("eval_loss", float(eval_loss), epoch)
        writer.add_scalar("consensus_loss", float(consensus_loss), epoch)        
        print("Evaluation Score : [{}]".format(eval_loss))
        print("consensus_loss : [{}]".format(consensus_loss))

    return model



VAEwriter = SummaryWriter()

aggresive_pickle_data = pickle_read(os.path.join(racingdata_dir, 'aggresive_vae_0.pkl'))
passive_pickle_data = pickle_read(os.path.join(racingdata_dir, 'passive_vae_0.pkl'))
# test_pickle_data = pickle_read(os.path.join(racingdata_dir, 'test_vae_0.pkl'))
test_pickle_data = pickle_read(os.path.join(racingdata_dir, 'step10_vae_0.pkl'))
# pickle_data = aggresive_pickle_data+passive_pickle_data
pickle_data = test_pickle_data # +aggresive_pickle_data # passive_pickle_data+aggresive_pickle_data

# 0 ego_s,
# 1 ego_ey,
# 2 ego_epsi,
# 3 ego_curvature,
# 4 ego_u_a,
# 5 ego_u_steer,       
# 6 tar_s,
# 7 tar_ey,
# 8 tar_epsi,
# 9 tar_curvature,
# 10 tar_u_a,
# 11 tar_u_steer,                                                                          
# 12 Q_xref, 
# 13 Q_theta


# Input for VAE -> 
# [(tar_s-ego_s),
#  ego_ey, ego_epsi, ego_cur,ego_accel, ego_delta,
#  tar_ey, tar_epsi, tar_cur,tar_accel, tar_delta] 
torch_data = torch.tensor(pickle_data)
cliped_data = torch.zeros([torch_data.shape[0],torch_data.shape[1],torch_data.shape[2]-1,13])
cliped_data[:,:,:,0] = torch_data[:,:,:-1,6] - torch_data[:,:,:-1,0] 
cliped_data[:,:,:,1:6] = torch_data[:,:,:-1,1:6]
cliped_data[:,:,:,6:] = torch_data[:,:,:-1,7:]



normalized_data = cliped_data.clone()
for i in range(cliped_data.shape[-1]-2):
    ax_data = cliped_data[:,:,:,i]
    normalized_data[:,:,:,i] = (ax_data-torch.mean(ax_data))/torch.std(ax_data)

# np_data = np.asarray(pickle_data)
# s_diff = np_data[:,:,5]-np_data[:,:,0]
# # input state : (s_tar-e_ego), ey_ego, epsi_ego, cur_ego, ey_tar, epsie_tar, cur_tar
# clip_mins = np.array([-10,      -2.0,    -np.pi,     -1,   -2.0,    -np.pi,     -1])
# clip_maxs = -1*clip_mins
# ey_ego = np_data[:,:,2]
# epsi_ego = np_data[:,:,1]
# cur_ego = np_data[:,:,10]
# ey_tar = np_data[:,:,7]
# epsi_tar = np_data[:,:,6]
# cur_tar = np_data[:,:,10]
# input_data = np.stack([s_diff,ey_ego,epsi_ego,(a-torch.mean(a))/torch.std(a)

# normalized_data = (cliped_np_data/clip_maxs)

# dataset = torch.Tensor(normalized_data)

dataset =normalized_data 

dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)
validation_size = int(dataset_size * 0.1)
test_size = dataset_size - train_size - validation_size
test_dataset = dataset[:test_size,:,:,:]
train_valid_dataset = dataset[test_size:,:,:,:]
train_dataset, validation_dataset, _ = torch.utils.data.random_split(train_valid_dataset, [train_size, validation_size, 0])

print(f"Training Data Size : {len(train_dataset)}")
print(f"Validation Data Size : {len(validation_dataset)}")
print(f"Testing Data Size : {len(test_dataset)}")

batch_size = 512

args =  {
            "batch_size": batch_size,
            "device": torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu"),
            "input_size": 11,
            "hidden_size": 2,
            "latent_size": 5,
            "learning_rate": 0.0001,
            "max_iter": 60000,
        }
    
"""
1. Comparison methods 
-> Offline GP (train with all policy) -> test with trained policy will give fail, test with untrained policy 
-> LSTM(trained with all policy) -> test with untrained policy will give fail, 
-> NMPC based(with differet weighting), 
2. 
"""

    
train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=args['batch_size'], shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=True)

train = True
if train:
    model = LSTMAutomodel(args)
    model.to(torch.device("cuda"))
    Autotrain(args,model,train_dataloader,validation_dataloader,VAEwriter)
    id_ = secrets.token_hex(nbytes=4)
    save_dir = model_dir+f"lstmvae{id_}.model"
    print("model has been saved in "+ save_dir)
    torch.save(model.state_dict(), save_dir )

if train:    
    model_to_load = model
else:    
    model_to_load = LSTMAutomodel(args)
    model_to_load.to(torch.device("cuda"))
    model_to_load.load_state_dict(torch.load(model_dir+f"lstmvae843be856.model")) ## normalized 5 latent    
    # model_to_load.load_state_dict(torch.load(model_dir+f"lstmvae4fd6231f.model")) ## non normalized 5 latent
    # model_to_load.load_state_dict(torch.load(model_dir+f"lstmvae3a3985e9.model")) ## trained with test pkl
    
    

model_to_load.eval()
test_iterator = tqdm(
            enumerate(test_dataloader), total=len(test_dataloader), desc="testing"
        )
batch_size = args["batch_size"]

filted_data = []
theta_s = []
color_indices_s = []
Q_xrefs = []
Q_thetas  = []
with torch.no_grad():
    for i, batch_data in test_iterator:
        in_range_data = []
        for i in range(batch_data.shape[0]):
            test = batch_data[i,0,:,:][:,0]
            if sum(test[test > 0.0]<0.5) > 3:            
                in_range_data.append(np.array(batch_data[i,0,:,:].cpu()))
        if len(in_range_data) < 1:
            continue
        in_range_data = np.array(in_range_data)
        filted_data.append(in_range_data[:,0,:])
        test_data = torch.from_numpy(in_range_data[:,:,0:-2]).to(args['device']) 
        ## reshape
        batch_size = test_data.shape[0]
        # example_size = past_data.size(1)
        # past_data = (
        #     past_data.view(batch_size, example_size, -1).float().to(args['device'])
        # )
        # future_data = future_data.view(batch_size, example_size, -1).float().to(args.device)
        z_latent = model_to_load.get_latent_z(test_data)        
              
        # Q_xref = batch_data[:,0,-2].cpu()
        # Q_theta = batch_data[:,0,-1].cpu()
        
        Q_xref = in_range_data[:,0,-2]
        Q_theta = in_range_data[:,0,-1]

        
        # Concatenate Y and Z into a single array
        combined = np.column_stack((Q_xref, Q_theta))
        # Get unique combinations and their index numbers
        combined_unique, color_indices = np.unique(combined, return_inverse=True, axis=0)
        
        theta_mean_np = np.array(z_latent.cpu())
        
        theta_s.append(theta_mean_np)
        Q_xrefs.append(Q_xref)
        Q_thetas.append(Q_theta)
        # color_indices_s.append(color_indices)
        color_indices_s.append(Q_xref)
        np.concatenate(theta_s, axis = 0)

Q_xref= np.concatenate(Q_xrefs,axis=0)
Q_theta= np.concatenate(Q_thetas,axis=0)
combined = np.column_stack((Q_xref, Q_theta))
# Get unique combinations and their index numbers
combined_unique, color_indices = np.unique(combined, return_inverse=True, axis=0)

filted_data = np.concatenate(filted_data, axis = 0)
theta = np.concatenate(theta_s, axis = 0)
color = color_indices
# color = np.concatenate(color_indices_s, axis = 0)

def run_ztheta_plot():
########################################
    
    plot_start = 0
    plot_finish = 1000
    test_input = test_dataset[plot_start:plot_finish,0,:,:-2]
    torch.cuda.empty_cache()
    with torch.no_grad():
   
        Qxref = test_dataset[plot_start:plot_finish,0,0,-2]
        Qtheta = test_dataset[plot_start:plot_finish,0,0,-1]
        
        z_test = model_to_load.get_latent_z(torch.tensor(test_dataset[plot_start:plot_finish,0,:,:-2]).squeeze().to(torch.device("cuda")))


        fig, axs = plt.subplots(z_test.shape[1]+1, 1)
        for i in range(z_test.shape[1]):
            axs[i].plot(z_test[:,i].cpu().detach().numpy(),'k*')
        axs[z_test.shape[1]].plot(Qxref,'r')
        axs[z_test.shape[1]].plot(Qxref,'b')
        plt.show()
    ############################################################################################3#
    
    # test_input = []
    # for i in range(len(test_dataset)):
    #     test_input.append(test_dataset[i].cpu().detach().numpy())    
    # test_input = np.array(test_input)
    # plot_start = 1000
    # plot_finish = 1200
    # test_input = test_input[plot_start:plot_finish,0,:,:-2]
    with torch.no_grad():
        test_loss, recon_test = model_to_load.forward(torch.tensor(test_input).squeeze().to(torch.device("cuda")))
    fig, axs = plt.subplots(recon_test.shape[2], 1)
    for j in range(recon_test.shape[2]):
        recon = recon_test[:,:,j].cpu().detach().numpy().reshape(-1)
        gt = test_input[:,:,j].reshape(-1)
        axs[j].plot(recon,'r*')
        axs[j].plot(gt,'k*')  


    





###################################
###################################
TSNE_analysis = True
###################################
###################################
if TSNE_analysis:
    print("t-SNE analysis init")
    from sklearn.manifold import TSNE
    for i in range(1):
        ###################################
        ###################################
        dim = 2        
        perplexity_ = 200
        n_iter_ = 5000
        ###################################
        ###################################
        tsne_model = TSNE(n_components=dim,perplexity=perplexity_, verbose= 2,n_iter=n_iter_)
        
        print("t-SNE optimization begin")
        theta_2d = tsne_model.fit_transform(theta)
        print("t-SNE optimization done")
        
        if dim >2:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(theta_2d[:, 0], theta_2d[:, 1],theta_2d[:, 2] ,c=color, cmap='viridis')
        else:
            fig, ax = plt.subplots()
            ax.scatter(theta_2d[:, 0], theta_2d[:, 1], c=color, cmap='viridis')
            cbar = plt.colorbar()
            cbar.set_label('Color Bar Label')
        
        

        for i in range(10):
            points = plt.ginput(1)
            x_clicked, y_clicked = points[0]
            dists = np.sqrt((theta_2d[:, 0] - x_clicked)**2 + (theta_2d[:, 1] - y_clicked)**2)
            index = np.argmin(dists)
            print("clicked x = ",round(x_clicked,1), ", clicked y = ", round(y_clicked,1))
            # print(np.around(filted_data[index,:],3))
            print("tars-egos = " ,   np.round(filted_data[index,0],3))
            print("ego_ey = " ,      np.round(filted_data[index,1],3))
            print("ego_epsi = " ,    np.round(filted_data[index,2],3))
            print("ego_cur = "  ,     np.round(filted_data[index,3],3))
            print("tar_ey = "   ,      np.round(filted_data[index,4],3))
            print("tar_epsi = " ,    np.round(filted_data[index,5],3))
            print("tar_cur = "  ,     np.round(filted_data[index,6],3))
            print("tar_accel = ",   np.round(filted_data[index,7],3))
            print("tar_delta = ",   np.round(filted_data[index,8],3))
            print("T_QXref = "  ,     np.round(filted_data[index,-2],3))
            print("T_Qtheta = " ,   np.round(filted_data[index,-1],3))
    
    
PCA_analysis = True
if PCA_analysis:
    print("PCA analysis init")
    from sklearn.decomposition import PCA
    pca = PCA(n_components= 2)
    pca.fit(theta)
    theta2d_pca = pca.transform(theta)
    fig, ax = plt.subplots()
    ax.scatter(theta2d_pca[:, 0],theta2d_pca[:, 1],  c=color, cmap='viridis')
    for i in range(10):
        points = plt.ginput(1)
        x_clicked, y_clicked = points[0]
        dists = np.sqrt((theta2d_pca[:, 0] - x_clicked)**2 + (theta2d_pca[:, 1] - y_clicked)**2)
        index = np.argmin(dists)
        print("clicked x = ",round(x_clicked,1), ", clicked y = ", round(y_clicked,1))
        # print(np.around(filted_data[index,:],3))
        print("tars-egos = " ,   np.round(filted_data[index,0],3))
        print("ego_ey = " ,      np.round(filted_data[index,1],3))
        print("ego_epsi = " ,    np.round(filted_data[index,2],3))
        print("ego_cur = "  ,     np.round(filted_data[index,3],3))
        print("tar_ey = "   ,      np.round(filted_data[index,4],3))
        print("tar_epsi = " ,    np.round(filted_data[index,5],3))
        print("tar_cur = "  ,     np.round(filted_data[index,6],3))
        print("tar_accel = ",   np.round(filted_data[index,7],3))
        print("tar_delta = ",   np.round(filted_data[index,8],3))
        print("T_QXref = "  ,     np.round(filted_data[index,9],3))
        print("T_Qtheta = " ,   np.round(filted_data[index,10],3))

##########################################################################################
##########################################################################################
############################## End of training session ###################################
##########################################################################################
##########################################################################################


