#!/usr/bin/env python3

from predictor.common.utils.file_utils import *
import numpy as np
import torch
from predictor.prediction.cont_encoder.cont_encoderdataGen import SampleGeneartorContEncoder
from predictor.prediction.cont_encoder.cont_policyEncoder import ContPolicyEncoder
from torch.utils.data import DataLoader, random_split


# Training
def cont_encoder_train(dirs):

    sampGen = SampleGeneartorContEncoder(dirs, randomize=True)
    
    sampGen.plotStatistics()
    
    if not dir_exists(dirs[0]):
        raise RuntimeError(
            f"Directory: {dirs[0]} does not exist, need to train using `gen_training_data` first")

    train_dataset, val_dataset, test_dataset  = sampGen.get_datasets()
    args_ =  {
                "batch_size": 512,
                "device": torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu"),
                "input_size": 9,
                "hidden_size": 8,
                "latent_size": 4,
                "learning_rate": 0.0005,
                "max_iter": 1800000,
                "seq_len" :5
            }
    batch_size = args_["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



    
    policy_encoder = ContPolicyEncoder(args= args_)    
    policy_encoder.set_train_loader(train_loader)
    policy_encoder.set_test_loader(test_loader)

    policy_encoder.train(args= args_)
    
    create_dir(path=model_dir)
    policy_encoder.model_save()
    


# T-SNE analysis 

def tsne_cont_encoder(dirs):
    args_ =  {
            "batch_size": 512,
            "device": torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu"),
            "input_size": 9,
            "hidden_size": 8,
            "latent_size": 4,
            "learning_rate": 0.0001,
            "max_iter": 240000,
            "seq_len" :5
        }
    batch_size = args_["batch_size"]
    
    z_list = []
    input_list = []
    y_label = []
    for i in range(len(dirs)):
        dir = [dirs[i]]
        a_sampGen = SampleGeneartorContEncoder(dir, randomize=True)
        a_train_dataset, a_val_dataset, a_test_dataset  = a_sampGen.get_datasets(filter= True)
        a_train_loader = DataLoader(a_train_dataset, batch_size=batch_size, shuffle=True)    
        a_test_loader = DataLoader(a_test_dataset, batch_size=batch_size, shuffle=False)
        a_policy_encoder = ContPolicyEncoder(args= args_)    
        a_policy_encoder.set_train_loader(a_train_loader)
        a_policy_encoder.set_test_loader(a_test_loader)
        a_policy_encoder.model_load()
        a_stacked_z, a_input = a_policy_encoder.tsne_evaluate()
        z_list.append(a_stacked_z)
        input_list.append(a_input)
        a_y_label = torch.ones(a_stacked_z.shape[0])*(i//3)
        y_label.append(a_y_label)

    stacked_z = torch.vstack(z_list).cpu()
    stacked_input = torch.vstack(input_list).cpu()
    stacked_label = torch.hstack(y_label).cpu()
    # stacked_z = torch.vstack([a_stacked_z,b_stacked_z]).cpu()    
    # stacked_input = torch.vstack([a_input,b_input]).cpu()        
    # y_label = torch.hstack([a_y_label,b_y_label]).cpu().numpy()
    ###################################
    ###################################
    ########## TSNE_analysis ##########
    ###################################
    ###################################
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    for i in range(1):
        ###################################        
        dim = 2        
        perplexity_ = 300
        n_iter_ = 800        

        ###################################
        tsne_model = TSNE(n_components=dim,perplexity=perplexity_, verbose= 2,n_iter=n_iter_)        
        print("t-SNE optimization begin")
        theta_2d = tsne_model.fit_transform(stacked_z)
        print("t-SNE optimization done")
        
        if dim >2:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(theta_2d[:, 0], theta_2d[:, 1],theta_2d[:, 2] ,c=stacked_label, cmap='viridis')
        else:
            fig, ax = plt.subplots()
            scatter_plot = ax.scatter(theta_2d[:, 0], theta_2d[:, 1], c=stacked_label, cmap='viridis')            
            # labels = ["timid", "mild_100", "mild_200", "mild_300", "mild_500","mild_1000", "mild_5000", "reverse"]
            labels = ["timid", "mild_200", "mild_500", "mild_5000", "reverse"]
            plt.legend(handles=scatter_plot.legend_elements()[0], labels=labels, title='Legend')

            plt.show()
            # cbar = plt.colorbar()
            # cbar.set_label('Color Bar Label')
            for i in range(10):
                points = plt.ginput(1)
                x_clicked, y_clicked = points[0]
                dists = np.sqrt((theta_2d[:, 0] - x_clicked)**2 + (theta_2d[:, 1] - y_clicked)**2)
                index = np.argmin(dists)
                print("clicked x = ",round(x_clicked,1), ", clicked y = ", round(y_clicked,1))
                # print(np.around(filted_data[index,:],3))
                print("tars-egos = " ,   np.round(stacked_input[index,0,0].cpu(),3))
                print("tar_ey = " ,      np.round(stacked_input[index,0,1].cpu(),3))
                print("tar_epsi = " ,    np.round(stacked_input[index,0,2].cpu(),3))
                print("tar_vx = " ,    np.round(stacked_input[index,0,3].cpu(),3))
                print("tar_cur = "  ,     np.round(stacked_input[index,0,4].cpu(),3))
                print("ego_ey = "   ,      np.round(stacked_input[index,0,5].cpu(),3))
                print("ego_epsi = " ,    np.round(stacked_input[index,0,6].cpu(),3))
                print("ego_vx = "  ,     np.round(stacked_input[index,0,7].cpu(),3))                                
                print("ego_cur = "  ,     np.round(stacked_input[index,0,8].cpu(),3))       
                print("stacked_label = " ,   stacked_label[index])
    


# if __name__ == "__main__":
#     main()
