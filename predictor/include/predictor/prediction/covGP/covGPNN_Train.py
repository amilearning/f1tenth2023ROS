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
def covGPNN_train(dirs = None, real_data = False, args = None):
   
    if args is None:
        print("ARGS should be given!!")
        return 
    sampGen = SampleGeneartorCOVGP(dirs, args = args, randomize=True, real_data = real_data)
 
    # sampGen.plotStatistics()
    
    if not dir_exists(dirs[0]):
        raise RuntimeError(
            f"Directory: {dirs[0]} does not exist, need to train using `gen_training_data` first")

  
        
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=args["gp_output_dim"]) 
    covgp_predictor = COVGPNN(args, sampGen, IndependentMultitaskGPModelApproximate, likelihood, enable_GPU=True)
    
    # if args["train_nn"l] is False:
    # snapshot_name = 'traceGP_3000snapshot'
    # covgp_predictor.load_model(snapshot_name)
    # print(" model loaded")
    covgp_predictor.train(sampGen, args = args)
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
        eval_folder = os.path.join(real_dir, eval_policy_names[i])
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
        y_label = []
        for i in range(len(dirs)):
            dir = [dirs[i]]
            
            sampGen = SampleGeneartorCOVGP(dir, load_normalization_constant = True, args = args, randomize=True, real_data = True, tsne = True)
            if sampGen.getNumSamples() < 1:
                continue
            covgp_predictor = COVGPNN(args, sampGen, IndependentMultitaskGPModelApproximate, likelihood, enable_GPU=True)
            
            covgp_predictor.load_model(snapshot_name)
            if not dir_exists(dirs[0]):
                raise RuntimeError(
                    f"Directory: {dirs[0]} does not exist, need to train using `gen_training_data` first")

            a_stacked_z, a_input = covgp_predictor.tsne_evaluate(sampGen)
            covgp_predictor.set_evaluation_mode()        
            z_list.append(a_stacked_z)
            input_list.append(a_input)
            a_y_label = torch.ones(a_stacked_z.shape[0])*(i%(len(dirs)))
            y_label.append(a_y_label)


        stacked_z = torch.vstack(z_list).cpu()
        stacked_input = torch.vstack(input_list).cpu()
        stacked_label = torch.hstack(y_label).cpu()
        # stacked_z = torch.vstack([a_stacked_z,b_stacked_z]).cpu()    
        # stacked_input = torch.vstack([a_input,b_input]).cpu()        
        # y_label = torch.hstack([a_y_label,b_y_label]).cpu().numpy()

        model_to_save = dict()
        model_to_save['stacked_z'] = stacked_z
        model_to_save['stacked_input'] = stacked_input
        model_to_save['stacked_label'] = stacked_label    
        pickle_write(model_to_save,tsne_data_dir)
    
        print('Successfully saved data')

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
        perplexity_ = perplexity
        n_iter_ = 20000        

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
            if eval_policy_names is None:
                labels = [ "timid","blocking", "reverse"]
            else:
                labels = eval_policy_names


            plt.legend(handles=scatter_plot.legend_elements()[0], labels=labels, title='Legend')

            
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(fig_dir, f"tsne_{snapshot_name}_{current_time}.png")
            plt.savefig(file_path)
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
    

