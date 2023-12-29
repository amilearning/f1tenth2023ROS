from predictor.common.utils.file_utils import *
from predictor.prediction.cont_encoder.cont_encoderTrain import cont_encoder_train
from predictor.prediction.thetaGP.ThetaGPTrain import thetagp_train
from predictor.prediction.gp_berkely_train import gp_main
from predictor.prediction.covGP.covGPNN_Train import covGPNN_train, tsne_analysis
import os
import torch
from predictor.prediction.covGP.EvalMultData import * 
from predictor.prediction.covGP.EvalMultiPrior import * 

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




def main_train(train_policy_names = None):
    train_dirs = []
    for i in range(len(train_policy_names)):
        test_folder = os.path.join(real_dir, train_policy_names[i])
        train_dirs.append(test_folder)


    # print("1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print("GP Berkely train init")
    # # gp_main(train_dirs, realdata = True)
    # print("GP Berkely train Done")
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


    print("1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Direct GP init")
    args_["direct_gp"] = True
    args_["include_simts_loss"] = False
    args_['model_name'] = 'naiveGP'
    # covGPNN_train(train_dirs, real_data = True, args= args_)
    print(" train Done")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    print("2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("No simtsGPNN_train  init")
    args_["direct_gp"] = False
    args_["include_simts_loss"] = False
    args_['model_name'] = 'nosimtsGP'
    # covGPNN_train(train_dirs, real_data = True, args= args_)
    print(" train Done")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


    print("3~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("simtsGPNN_train  init")
    args_["direct_gp"] = False
    args_["include_simts_loss"] = True    
    args_['model_name'] = 'simtsGP'
    # covGPNN_train(train_dirs, real_data = True, args= args_)
    print(" train Done")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    
    


def gen_eval_data(eval_policy_names):

    eval_dirs = []
    for i in range(len(eval_policy_names)):
        eval_folder = os.path.join(real_dir, eval_policy_names[i])
        eval_dirs.append(eval_folder)

    ########## Generate Prediction data for each predictor ########   
    rospy.init_node("MultiPredPostEval") 
    MultiPredPostEval(eval_dirs, args_)

def run_eval(eval_policy_names):
    predtype_lateral_errors_list= []
    predtype_longitudinal_errors_list= []
    predtype_errors_list= []
    predtype_pred_covs_list= []
    filted_train_policy_names = []
    for j in range(len(eval_policy_names)):                
        lateral_errors_list= []
        longitudinal_errors_list= []
        errors_list =[]
        pred_covs_list= []        

        for i in range(0,5):    
            # i is in predictor type  0 nosimtsgp, 1- cav, 2- nmpc, 3 - naivegp , 4 - simtsgp
            lateral_errors, longitudinal_errors, pred_covs = get_process(eval_policy_names[j], predictor_type = i)            
            rmse_longitudinal = np.sqrt(np.sum(longitudinal_errors**2))
            rmse_lateral = np.sqrt(np.sum(lateral_errors**2))
            rmse_combined = np.sqrt((rmse_longitudinal ** 2 + rmse_lateral ** 2) / 2)
            print(str(i) + "th predictor,  mse combined = " + str(rmse_combined))
            # print(str(i) + "th predictor,  long_eror = " + str(np.mean(longitudinal_errors)))
            # print(str(i) + "th predictor,  lat_eror = " + str(np.mean(lateral_errors)))            
            lateral_errors_list.append((lateral_errors))
            longitudinal_errors_list.append((longitudinal_errors))
            pred_covs_list.append(pred_covs)
            errors_list.append( abs(longitudinal_errors) + abs(lateral_errors))
         
        
        predtype_lateral_errors_list.append(lateral_errors_list)
        predtype_longitudinal_errors_list.append(longitudinal_errors_list)
        predtype_errors_list.append(errors_list)
        predtype_pred_covs_list.append(pred_covs_list)
        
           
        

    
    
    draw_barplot_with_list(predtype_lateral_errors_list, eval_policy_names, value_name_ = 'Lateral_error')
    draw_barplot_with_list(predtype_longitudinal_errors_list, eval_policy_names, value_name_ = 'Long_error')
    draw_barplot_with_list(predtype_errors_list, eval_policy_names, value_name_ = 'Error')
    draw_barplot_with_list(predtype_pred_covs_list, eval_policy_names, value_name_ = 'COV')



def main():  
    ####################################################
    train_policy_names = ['centerline_1220',
                        'blocking_1220',
                        'hjpolicy_1220',
                        'highspeed_centerlin_1221',
                        'highspeed_aggresive_1221',
                        'highspeed_hjpolicy_1221']    
    # train_policy_names = ['centerline_1220',
    #                     'blocking_1220',
    #                     'highspeed_aggresive_1221',
    #                     'highspeed_centerlin_1221',
    #                     'highspeed_centerline2_1221',
    #                     'highspeed_hjpolicy_1221',
    #                     'highspeed_reverse_1221',
    #                     'hjpolicy_1220',
    #                     'nonblocking_yet_racing_1220',
    #                     'reverse_1220',
    #                     'wall']    
                    #  'nonsense_reverse',
                        
    ####################################################
    main_train(train_policy_names)
    ####################################################
    args_['add_noise_data'] = False
    ############ TSNE ##################################
    eval_policy_names = ['eval_centerline_1220',
                         'centerline_1220',
                         'highspeed_aggresive_1221',
                         'eval_highspeed_aggresive_1221'] 
    # eval_policy_names = ['blocking_1220']
    args_['model_name'] ='simtsGP'
    tsne_analysis( args = args_, snapshot_name = 'simtsGP', eval_policy_names = eval_policy_names, perplexity = 50, load_data=False)
    args_['model_name'] ='nosimtsGP'
    tsne_analysis(args = args_, snapshot_name = 'nosimtsGP', eval_policy_names = eval_policy_names, perplexity = 50, load_data=False)
    
    ####################################################
    eval_policy_names = ['eval_centerline_1220',
                        'eval_blocking_1220',
                        'eval_highspeed_aggresive_1221']
    # ,
    #                     'eval_highspeed_aggresive_1221',
    #                     'eval_highspeed_hjpolicy_1221'] 

    ####################################################
    gen_eval_data(eval_policy_names)
    ####################################################
    run_eval(eval_policy_names)        


if __name__ == "__main__":
    main()





