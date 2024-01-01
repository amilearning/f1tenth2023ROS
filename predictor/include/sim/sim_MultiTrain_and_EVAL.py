from predictor.common.utils.file_utils import *
from predictor.prediction.covGP.covGPNN_Train import covGPNN_train
import os
import torch
from predictor.prediction.covGP.EvalMultData import * 
from predictor.prediction.covGP.EvalMultiPrior import * 

def get_sim_dir(policy_name_, train_dir_):
    policy_dir = os.path.join(train_dir_, policy_name_)
    scencurve_dir = os.path.join(policy_dir, 'curve')
    scenstraight_dir = os.path.join(policy_dir, 'straight')
    scenchicane_dir = os.path.join(policy_dir, 'chicane')
    dirs = [scencurve_dir, scenstraight_dir, scenchicane_dir]
    return dirs


args_ = {                    
    "batch_size": 512,
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    "input_dim": 9,
    "n_time_step": 10,
    "latent_dim": 9,
    "gp_output_dim": 4,
    "inducing_points" : 200,
    "train_nn" : False,
    "include_simts_loss" : True,
    "direct_gp" : False,
    "n_epoch" : 10000,
    'add_noise_data': True,
    'model_name' : None
    }

def main_train(train_dirs):
    

    print("1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Direct GP init")
    args_["direct_gp"] = True
    args_["include_simts_loss"] = False
    args_['model_name'] = 'naiveGP'
    covGPNN_train(train_dirs, real_data = False, args= args_)
    print(" train Done")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    print("2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("No simtsGPNN_train  init")
    args_["direct_gp"] = False
    args_["include_simts_loss"] = False
    args_['model_name'] = 'nosimtsGP'
    covGPNN_train(train_dirs, real_data = False, args= args_)
    print(" train Done")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


    print("3~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("simtsGPNN_train  init")
    args_["direct_gp"] = False
    args_["include_simts_loss"] = True    
    args_['model_name'] = 'simtsGP'
    covGPNN_train(train_dirs, real_data = False, args= args_)
    print(" train Done")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    

def main():  
    ####################################################
 
    timid_0 = get_sim_dir('timid_0', train_dir)
    aggressive_blocking_0 = get_sim_dir('aggressive_blocking_0', train_dir)
    mild_5000_0 = get_sim_dir('mild_5000_0', train_dir)
    reverse_0 = get_sim_dir('reverse_0', train_dir)

    dirs = timid_0.copy()
    dirs.extend(aggressive_blocking_0)
    dirs.extend(mild_5000_0)
    dirs.extend(reverse_0)

    ####################################################
    main_train(dirs)
    ####################################################
 


if __name__ == "__main__":
    main()





