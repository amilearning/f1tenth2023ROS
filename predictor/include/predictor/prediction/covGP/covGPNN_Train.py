#!/usr/bin/env python3

from predictor.common.utils.file_utils import *
import numpy as np
import torch
import gpytorch
from datetime import datetime
from predictor.prediction.gpytorch_models import IndependentMultitaskGPModelApproximate

from predictor.prediction.encoder.encoderModel import LSTMAutomodel
from predictor.prediction.encoder.policyEncoder import PolicyEncoder
from torch.utils.data import DataLoader, random_split


from predictor.prediction.covGP.covGPNN_model import COVGPNN
from predictor.prediction.covGP.covGPNN_dataGen import SampleGeneartorCOVGP


# Training
def covGPNN_train(dirs = None, val_dirs = None, real_data = False, args = None):
   
    if args is None:
        print("ARGS should be given!!")
        return 
    sampGen = SampleGeneartorCOVGP(dirs, args = args, randomize=True, real_data = real_data)
    # sampGen.plotStatistics()
    valid_args = args.copy()
    valid_args['add_noise_data'] = False
    valid_args['add_aug_data'] = False
  
    valGEn = SampleGeneartorCOVGP(val_dirs, load_normalization_constant = True, args = valid_args, randomize=False, real_data = real_data, tsne = False)
 
    
    
    if not dir_exists(dirs[0]):
        raise RuntimeError(
            f"Directory: {dirs[0]} does not exist, need to train using `gen_training_data` first")

  
        
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=args["gp_output_dim"]) 
    covgp_predictor = COVGPNN(args, sampGen, IndependentMultitaskGPModelApproximate, likelihood, enable_GPU=True)
    
    # if args["train_nn"l] is False:
    # snapshot_name = 'traceGP_3000snapshot'
    # covgp_predictor.load_model(snapshot_name)
    # print(" model loaded")
    covgp_predictor.train(sampGen, valGEn, args = args)
    covgp_predictor.set_evaluation_mode()
    trained_model = covgp_predictor.model, covgp_predictor.likelihood

    create_dir(path=model_dir)
    if args['direct_gp'] is True:
        gp_name = 'naiveGP'
    else:   
        if(args['include_simts_loss']):
            gp_name = 'simtsGP'
        else:
            gp_name = 'nosimtsGP'
    covgp_predictor.save_model(gp_name)
    # covgp_predictor.load_model(gp_name)
    # covgp_predictor.evaluate()


def tsne_analysis(args = None , snapshot_name = 'covGP',  eval_policy_names = None, perplexity = 10, load_data = False):
    # def tsne_evaluate(self,sampGen: SampleGeneartorCOVGP):    


    dirs = []
    for i in range(len(eval_policy_names)):
        eval_folder = os.path.join(train_dir, eval_policy_names[i])
        dirs.append(eval_folder)


    if args is None:
        args = {                    
                "batch_size": 512,
                "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                "input_dim": 9,
                "n_time_step": 15,
                "latent_dim": 8,
                "gp_output_dim": 4,
                "inducing_points" : 100,
                "train_nn" : False                
                }
        
    


    tsne_data_dir =os.path.join(eval_dir,'tsne.pkl')
    if load_data:    
        loaded_model = pickle_read(tsne_data_dir)
        stacked_z = loaded_model['stacked_z'] 
        stacked_input = loaded_model['stacked_input'] 
        stacked_label = loaded_model['stacked_label'] 
        print('Successfully loaded data')

    else:
      
        
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=args["gp_output_dim"]) 
        
        z_list = []
        input_list = []
        output_list = []
        y_label = []
        covout_list= []
        for i in range(len(dirs)):
            dir = [dirs[i]]
            
            sampGen = SampleGeneartorCOVGP(dir, load_normalization_constant = True, args = args, randomize=False, real_data = False, tsne = True)
            # sampGen.plotStatistics()
            if sampGen.getNumSamples() < 1:
                continue
            covgp_predictor = COVGPNN(args, sampGen, IndependentMultitaskGPModelApproximate, likelihood, enable_GPU=True)
            
            covgp_predictor.load_model(snapshot_name)
            if not dir_exists(dirs[0]):
                raise RuntimeError(
                    f"Directory: {dirs[0]} does not exist, need to train using `gen_training_data` first")

            a_stacked_z, a_input, a_output, cov_output= covgp_predictor.tsne_evaluate(sampGen)
            covgp_predictor.set_evaluation_mode()        
            z_list.append(a_stacked_z)
            input_list.append(a_input)
            output_list.append(a_output)
            covout_list.append(cov_output)
            a_y_label = torch.ones(a_stacked_z.shape[0])*(i%(len(dirs)))
            y_label.append(a_y_label)


        stacked_z = torch.vstack(z_list).cpu()
        stacked_input = torch.vstack(input_list).cpu()
        stacked_output = torch.vstack(output_list).cpu()
        stacked_label = torch.hstack(y_label).cpu()
        cov_label = torch.hstack(covout_list).cpu()
        # stacked_z = torch.vstack([a_stacked_z,b_stacked_z]).cpu()    
        # stacked_input = torch.vstack([a_input,b_input]).cpu()        
        # y_label = torch.hstack([a_y_label,b_y_label]).cpu().numpy()

        model_to_save = dict()
        model_to_save['stacked_z'] = stacked_z
        model_to_save['stacked_input'] = stacked_input
        model_to_save['stacked_output'] = stacked_output
        model_to_save['stacked_label'] = stacked_label    
        pickle_write(model_to_save,tsne_data_dir)
    
        print('Successfully saved data')

    inputs = stacked_input.detach().cpu().numpy()
    outputs = stacked_output.detach().cpu().numpy()
    cov_label = cov_label.detach().cpu().numpy()
    
    inputs = inputs.reshape([-1,inputs.shape[1],inputs.shape[2]])
    tar_ey = abs(inputs[:,1,9])
    # tar_ey[tar_ey>1.5]=0.0
    # idx= abs(inputs[:,1,9])>1.0
    # tar_ey[idx]=0.0
    del_ey = abs(inputs[:,1,9]-inputs[:,6,9])
    idx = inputs[:,0,9]>1.5
    del_ey[inputs[:,0,9]>1.5] = 0.0
    tar_epsi = inputs[:,2,9]
    agglevel = del_ey    
    agglevel = np.clip(agglevel, -30,30)
    agglevel= abs((agglevel)/(np.max(agglevel) - np.min(agglevel)))
    visualize= False
    if visualize:
        plt.plot(inputs[:,0,0])
        plt.plot(inputs[:,1,0])
        plt.plot(inputs[:,2,0])
        plt.plot(agglevel)    
        legend_txt = ['dels','ey','epsi','agglevel']
        plt.legend(legend_txt)
        plt.show()
    


 
    
    

   

    ###################################
    ###################################
    ########## TSNE_analysis ##########
    ###################################
    ###################################
    import matplotlib.pyplot as plt
    if stacked_z.shape[1]< 3:
        plt.plot(stacked_z.squeeze(),stacked_label,'*')
        plt.show()
        return

    from sklearn.manifold import TSNE
    
    from sklearn.decomposition import PCA
    for i in range(1):
        ###################################        
        dim = 2
        perplexity_ = perplexity
        n_iter_ = 40000        


        print("t-SNE optimization begin")
        tsne_model = TSNE(n_components=dim,perplexity=perplexity_, verbose= 2,n_iter=n_iter_)                
        theta_2d = tsne_model.fit_transform(stacked_z)
        print("t-SNE optimization done")
        
        # data_np = stacked_z.numpy()
        # pca = PCA(n_components=dim)
        # pca.fit(data_np)
        # theta_2d = pca.transform(data_np)

        def draw_tsne(label_idx, label_):
            if dim >2:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                scatter_plot = ax.scatter(theta_2d[:, 0], theta_2d[:, 1],theta_2d[:, 2] ,c=(label_), cmap='viridis')
            else:            
                fig, ax = plt.subplots()
                scatter_plot = ax.scatter(theta_2d[:, 0], theta_2d[:, 1], c=(label_), cmap='viridis')    
            cbar = plt.colorbar(scatter_plot)
            cbar.set_label('Target_ey', rotation=270, labelpad=15)

            if eval_policy_names is None:
                labels = [ "timid","blocking", "reverse"]
            else:
                labels = eval_policy_names


            plt.legend(handles=scatter_plot.legend_elements()[0], labels=labels, title='Legend')

            label_map = ['cov', 'tarey', 'label']
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(fig_dir, f"{label_map[label_idx]}_tsne_{snapshot_name}_{current_time}_{dim}.png")
            plt.savefig(file_path)
            
        # draw_tsne(2, stacked_label, 3)
        draw_tsne(0, cov_label)        
        draw_tsne(1, tar_ey)
        draw_tsne(2, stacked_label)
        
        # draw_tsne(1, tar_ey, 3)
        # draw_tsne(0, cov_label, 3)
        # draw_tsne(2, stacked_label, 3)
            # cbar = plt.colorbar()
            # cbar.set_label('Color Bar Label')
            # for i in range(10):
            #     points = plt.ginput(1)
            #     x_clicked, y_clicked = points[0]
            #     dists = np.sqrt((theta_2d[:, 0] - x_clicked)**2 + (theta_2d[:, 1] - y_clicked)**2)
            #     index = np.argmin(dists)
            #     print("clicked x = ",round(x_clicked,1), ", clicked y = ", round(y_clicked,1))
            #     # print(np.around(filted_data[index,:],3))
            #     print("tars-egos = " ,   np.round(stacked_input[index,0,0].cpu(),3))
            #     print("tar_ey = " ,      np.round(stacked_input[index,0,1].cpu(),3))
            #     print("tar_epsi = " ,    np.round(stacked_input[index,0,2].cpu(),3))
            #     print("tar_vx = " ,    np.round(stacked_input[index,0,3].cpu(),3))
            #     print("tar_cur = "  ,     np.round(stacked_input[index,0,4].cpu(),3))
            #     print("ego_ey = "   ,      np.round(stacked_input[index,0,5].cpu(),3))
            #     print("ego_epsi = " ,    np.round(stacked_input[index,0,6].cpu(),3))
            #     print("ego_vx = "  ,     np.round(stacked_input[index,0,7].cpu(),3))                                
            #     print("ego_cur = "  ,     np.round(stacked_input[index,0,8].cpu(),3))       
            #     print("stacked_label = " ,   stacked_label[index])
    

