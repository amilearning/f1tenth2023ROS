from predictor.common.utils.file_utils import *



from predictor.prediction.covGP.covGPNN_Train import covGPNN_train, tsne_analysis
import os
import torch
from predictor.prediction.covGP.EvalMultData import * 
from predictor.prediction.covGP.EvalMultiPrior import * 

args_ = {                    
    "batch_size": 1024,
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    "input_dim": 10,
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
    print("naiveGP init")
    args_["direct_gp"] = True
    args_["include_simts_loss"] = False
    args_['model_name'] = 'naiveGP'
    covGPNN_train(train_dirs, val_dirs, real_data = True, args= args_)
    print("naiveGP train Done")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    print("2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("nosimtsGPNN_train  init")
    args_["direct_gp"] = False
    args_["include_simts_loss"] = False
    args_['model_name'] = 'nosimtsGP'
    covGPNN_train(train_dirs, val_dirs, real_data = True, args= args_)
    print(" nosimtsGPNN_train Done")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


    print("3~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("simtsGPNN_train  init")
    args_["direct_gp"] = False
    args_["include_simts_loss"] = True    
    args_['model_name'] = 'simtsGP'
    # covGPNN_train(train_dirs,val_dirs, real_data = True, args= args_)
    print("simtsGPNN_train Done")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


def gen_eval_data(eval_policy_names):

    eval_dirs = []
    for i in range(len(eval_policy_names)):
        eval_folder = os.path.join(real_dir, eval_policy_names[i])
        eval_dirs.append(eval_folder)

    ########## Generate Prediction data for each predictor ########   
    rospy.init_node("MultiPredPostEval") 
    args_['eval'] = True
    # args_['load_eval_data'] = False
    MultiPredPostEval(eval_dirs, args_)

def main():  
    ####################################################
    ####################################################
    # train_policy_names = ['centerline_train',
    #                       'blocking_train']  
    train_policy_names = ['dl_1_blocking_train', 'dl_1_real_center_train'] # ,'blocking_train']             
    
    valid_policy_names = ['dl_1_blocking_eval', 'dl_1_real_center_eval'] #,'blocking_eval']             
                 
    main_train(train_policy_names, valid_policy_names)
    ####################################################    
    ############ TSNE ##################################
    args_['add_noise_data'] = False
    # tsne_policy_names = ['blocking_tsne',                         
    #                      'centerline_tsne'
    #                      ] 
    tsne_policy_names = ['centerline_tsne',
                        'blocking_tsne','highcenter_tsne']
                        # ,
                        # 'reverse_tsne',
                        # 'highblocking_tsne']
    # ,
    #                     'reverse_tsne',
    #                     'highblocking_tsne']  
 
    
    # args_['model_name'] ='simtsGP'
    # tsne_analysis( args = args_, snapshot_name = 'simtsGP', eval_policy_names = tsne_policy_names, perplexity = 40, load_data=False)
    # args_['model_name'] ='nosimtsGP'
    # tsne_analysis(args = args_, snapshot_name = 'nosimtsGP', eval_policy_names = tsne_policy_names, perplexity = 40, load_data=False)
    
    ####################################################
    eval_policy_names = ['real_center_train'] 
    #,
    #                   'real_blocking_train']     
    
    gen_eval_data(eval_policy_names)
    ####################################################
    

if __name__ == "__main__":
    main()





