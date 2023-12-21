from predictor.common.utils.file_utils import *
from predictor.prediction.cont_encoder.cont_encoderTrain import cont_encoder_train
from predictor.prediction.thetaGP.ThetaGPTrain import thetagp_train
from predictor.prediction.gp_berkely_train import gp_main
from predictor.prediction.covGP.covGPNN_Train import covGPNN_train, tsne_analysis
import os
import torch
from predictor.prediction.covGP.EvalMultData import * 
from predictor.prediction.covGP.EvalMultiPrior import * 

def main():  
    

    ######################## 

    
    #######################
    train_policy_names = ['test', 'test2']
    train_dirs = []
    for i in range(len(train_policy_names)):
        test_folder = os.path.join(real_dir, train_policy_names[i])
        train_dirs.append(test_folder)

    args_ = {                    
        "batch_size": 512,
        "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        "input_dim": 9,
        "n_time_step": 15,
        "latent_dim": 8,
        "gp_output_dim": 4,
        "inducing_points" : 100,
        "train_nn" : False,
        "include_cov_loss" : True,
        "n_epoch" : 50
        }
    
    

    print("1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("GP Berkely train init")
    gp_main(train_dirs, realdata = True)
    print("GP Berkely train Done")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    


    print("2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("covGPNN_train  init")
    covGPNN_train(train_dirs, real_data = True, args= args_)
    print(" train Done")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    
    
    print("3~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("No covGPNN_train  init")
    args_["include_cov_loss"] = False
    covGPNN_train(train_dirs, real_data = True, args= args_)
    print(" train Done")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


    # folder_name = ['centerline_1220', 'nonblocking_yet_racing_1220', 'blocking_1220', 'hjpolicy_1220', 'reverse_1220'] 
    ############# EVAL INIT ####################
    ############# EVAL INIT ####################
    ############# EVAL INIT ####################    
    eval_policy_names = ['test_eval', 'test_eval2']
    eval_dirs = []
    for i in range(len(eval_policy_names)):
        eval_folder = os.path.join(real_dir, eval_policy_names[i])
        eval_dirs.append(eval_folder)

    



    rospack = rospkg.RosPack()
    pkg_dir = rospack.get_path('predictor')
    rospy.init_node("MultiPredPostEval") 
    ########## Generate Prediction data for each predictor ########   
    MultiPredPostEval(eval_dirs)

    
    ############# EVAL INIT ####################
    # folder_name = ['test']
    # test_folder = os.path.join(real_dir, folder_name[0])
    # dirs = [test_folder]    
    ############# EVAL INIT ####################
    # Evaluate the predicted results for different driving policies
    ############# EVAL INIT ####################

    predtype_lateral_errors_list= []
    predtype_longitudinal_errors_list= []
    predtype_pred_covs_list= []
    for j in range(len(eval_policy_names)):                
        lateral_errors_list= []
        longitudinal_errors_list= []
        pred_covs_list= []        
        for i in range(0,5):    
            # i is in predictor type  0 - cav 1 - nmpc 2 - gp 3 - covgp
            lateral_errors, longitudinal_errors, pred_covs = get_process(eval_policy_names[j], predictor_type = i)
            lateral_errors_list.append(lateral_errors)
            longitudinal_errors_list.append(longitudinal_errors)
            pred_covs_list.append(pred_covs)
        
        predtype_lateral_errors_list.append(lateral_errors_list)
        predtype_longitudinal_errors_list.append(longitudinal_errors_list)
        predtype_pred_covs_list.append(pred_covs_list)
        


    draw_barplot_with_list(predtype_lateral_errors_list, 'later', value_name_ = 'Lateral_error')
    draw_barplot_with_list(predtype_longitudinal_errors_list, 'long', value_name_ = 'Long_error')
    draw_barplot_with_list(predtype_pred_covs_list, 'covs', value_name_ = 'COV')

    ############ TSNE ##################################
    tsne_analysis( train_dirs,args = args_, snapshot_name = 'covGP', folders = train_policy_names, perplexity = 5, load_data=False)
    tsne_analysis(train_dirs,args = args_, snapshot_name = 'nocovGP', folders = train_policy_names, perplexity = 5, load_data=False)


if __name__ == "__main__":
    main()





