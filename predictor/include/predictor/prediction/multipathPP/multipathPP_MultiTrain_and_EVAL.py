from predictor.common.utils.file_utils import *



import os
import torch
from predictor.prediction.covGP.EvalMultData import * 
from predictor.prediction.covGP.EvalMultiPrior import * 
from predictor.prediction.multipathPP.multipathPP_Train import multipathpp_train
args_ = {                    
    "batch_size": 128,
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    "input_dim": 10,
    "output_dim": 4,
    "n_time_step": 10,
    "latent_dim": 8,
    "gp_output_dim": 4,
    "inducing_points" : 100,
    "train_nn" : False,
    "include_simts_loss" : True,
    "direct_gp" : False,
    "n_epoch" : 10000,
    'add_noise_data': False,
    'add_aug_data' : False,
    'model_name' : None,
    'eval' : False,
    'load_eval_data' : False
    }



def main_train(train_policy_names = None, valid_policy_names = None):
    train_dirs = []
    for i in range(len(train_policy_names)):
        test_folder = os.path.join(real_dir, train_policy_names[i])
        train_dirs.append(test_folder)

    val_dirs = []

    for i in range(len(valid_policy_names)):
        test_folder = os.path.join(real_dir, valid_policy_names[i])
        val_dirs.append(test_folder)
        
    print("1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("MultiPlusPP init")
    args_['model_name'] = 'multipathpp'
    multipathpp_train(train_dirs, val_dirs, real_data = True, args= args_)
    print("MultiPlusPP train")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")





def main():  
    ####################################################
    ####################################################
    # train_policy_names = ['centerline_train',
    #                       'blocking_train']  
    # train_policy_names = ['dl_1_blocking_train', 'dl_1_real_center_train']
    # valid_policy_names = ['dl_1_blocking_eval', 'dl_1_real_center_eval'] 
    
    # train_policy_names = [ 'dl_1_blocking_train']
    # valid_policy_names = [ 'dl_1_blocking_eval'] 

    train_policy_names = ['dl_1_blocking_train', 'dl_1_300_center_train','hdkl']
    valid_policy_names = ['dl_1_blocking_eval', 'dl_1_300_center_eval','hdkl_eval'] 


    # train_policy_names = ['dl_1_blocking_train', 'dl_1_300_center_train']
    # valid_policy_names = ['dl_1_blocking_eval', 'dl_1_300_center_train'] 
    # train_policy_names = ['dl_1_blocking_train', 'dl_1_300_center_train','hdkl']
    # valid_policy_names = ['dl_1_blocking_eval', 'dl_1_300_center_eval','hdkl_eval'] 
                 
    main_train(train_policy_names, valid_policy_names)
    ####################################################    
    ############ TSNE ##################################
    ####################################################
    eval_policy_names = ['real_center_train'] 
    ####################################################
    

if __name__ == "__main__":
    main()





