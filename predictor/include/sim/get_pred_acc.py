#!/usr/bin/env python3

from tqdm import tqdm
import seaborn as sns
from typing import List
import cProfile, pstats, io
from pstats import SortKey
import pandas as pd

import matplotlib.pyplot as plt

from predictor.common.utils.file_utils import *
from predictor.common.utils.scenario_utils import MultiPolicyEvalData, EvalData, PostprocessData

from predictor.h2h_configs import *
from predictor.common.utils.scenario_utils import RealData
from predictor.common.utils.scenario_utils import wrap_del_s
import seaborn as sns

time_horizon = 12

def get_lat_lon_err(target_state: VehicleState, pred: VehiclePrediction, idx , track):


    if len(pred.s) <1 and pred.x is not None:
        f = track.global_to_local((pred.x[idx], pred.y[idx], pred.psi[idx]))
        if f is not None:
            s, x_tran, e_psi = f
        else:
            return None, None
        lon_error = wrap_del_s(target_state.p.s, s , track)  
        # lon_error = s - target_state.p.s
        lat_error = x_tran - target_state.p.x_tran
    else:
        lat_error = target_state.p.x_tran - pred.x_tran[idx]    
        lon_error = target_state.p.s - pred.s[idx]    
        lon_error = wrap_del_s(target_state.p.s, pred.s[idx] , track)  

    return lat_error, lon_error

def main(args=None):
    

  
    # policy_names = ['timid', 'mild_200', 'aggressive_blocking',  'mild_5000' ,'reverse']
    # policy_names = ['mild_200', 'aggressive_blocking', 'mild_5000', 'reverse']    
    predictor_names = ['hdkl', 'dkl', 'gp','cav','mpcc']
    
    
    lat_error_list = []
    lon_error_list =[]
    invalid_data_count = 0
    for predictor_name in predictor_names:
        ab_p = os.path.join(pred_dir,predictor_name)
        longitudinal_errors = []
        lateral_erros = []
        for filename in os.listdir(ab_p):
            if filename.endswith(".pkl"):
                dbfile = open(os.path.join(ab_p, filename), 'rb')        
                scenario_data: RealData = pickle.load(dbfile)                                                            
                track_ = scenario_data.track
            # scenario_data: RealData = pickle.load(dbfile)                        
                N = scenario_data.N             
                if N > time_horizon: 
                    for t in range(N-1-time_horizon):                      
                        ego_st = scenario_data.ego_states[t]
                        
                        tar_st = scenario_data.tar_states[t]
                        delta_s = wrap_del_s(tar_st.p.s, ego_st.p.s, track_)                  
                       
                        if delta_s < 2.0  and delta_s > 0 and abs(tar_st.v.v_long) > 0.1:                     
                        # if abs(delta_s) < 1.5  and tar_st.v.v_long > 0.1:
                            tar_pred = scenario_data.tar_pred[t]
                            pred_horizon = len(tar_pred.s)
                            terminal_tar_st = scenario_data.tar_states[t+pred_horizon-1]                            
                            lat_error, lon_error = get_lat_lon_err(terminal_tar_st, tar_pred, pred_horizon-1, track_)                            
                            # if lon_error is not None:
                            #     if abs(lon_error) > 0.6 and (predictor_name != 'cav'):                                    
                            #         print(lon_error)
                            #         continue
                          
                            if lat_error is not None and abs(lon_error) < 1.0 :
                                longitudinal_errors.append((lon_error**2))
                                lateral_erros.append((lat_error**2))

        
                dbfile.close()
        print(str(predictor_name) + ' _lon mean =' + str(np.sqrt(np.mean(longitudinal_errors))) + '  std = ' + str(np.std(np.sqrt(longitudinal_errors))))
        print(str(predictor_name) + ' _lat mean =' + str(np.sqrt(np.mean(lateral_erros)))+ '  std = ' + str(np.std(np.sqrt(lateral_erros))))
        
        lat_error_list.append(lateral_erros) 
        lon_error_list.append(longitudinal_errors)
        
        
 
    # Predictor names
    predictors = ["H-DKL", "DKL", "GP", "CAV", "MPCC"]

    # Convert to DataFrame for easier plotting
    df_lat_errors = pd.DataFrame(lat_error_list).transpose()
    df_lat_errors.columns = predictors

    df_long_errors = pd.DataFrame(lon_error_list).transpose()
    df_long_errors.columns = predictors

    # Melt the DataFrames for seaborn
    df_lat_melted = df_lat_errors.melt(var_name='Predictor', value_name='Lateral Error')
    df_long_melted = df_long_errors.melt(var_name='Predictor', value_name='Longitudinal Error')

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 6), sharey=True)

    # Lateral Error Boxplot without outliers
    sns.boxplot(x='Predictor', y='Lateral Error', data=df_lat_melted, ax=axs[0], showfliers=False)
    axs[0].set_title('Lateral Error Distribution by Predictor')
    axs[0].set_xticklabels(predictors, rotation=0)

    # Longitudinal Error Boxplot without outliers
    sns.boxplot(x='Predictor', y='Longitudinal Error', data=df_long_melted, ax=axs[1], showfliers=False)
    axs[1].set_title('Longitudinal Error Distribution by Predictor')
    axs[1].set_xticklabels(predictors, rotation=0)

    plt.tight_layout()
    plt.show()


    print(1)

    # fig, axs = plt.subplots(1, len(liklihood_traces_list), figsize=(12, 4))
    # for i, predictor_data in enumerate(liklihood_traces_list):
    #     # Extract x, y positions, and likelihood for the current predictor type
    #     x, y, likelihood = zip(*predictor_data)

    #     likelihood = (likelihood - min_val) / (max_val - min_val)


    #     # Create a scatter plot with colormap based on likelihood
    #     scatter = axs[i].scatter(x, y, c=likelihood, cmap='viridis', label='Likelihood')
    #     axs[i].set_title(f'Predictor Type {i}')
    #     axs[i].set_xlabel('X')
    #     axs[i].set_ylabel('Y')
    #     axs[i].legend()

    # # Add a colorbar to the last subplot
    # cbar = fig.colorbar(scatter, ax=axs, orientation='vertical')    
    # cbar.set_label('negativeLogLikelihood')

    # plt.show()



###############################################
    # sns.set(style="whitegrid")
    # bin_size = 0.3  # Adjust this value as needed
    # # Create subplots for each predictor type
    # num_predictor_types = len(pred_likelihoods)
    # fig, ax = plt.subplots(figsize=(12, 6))

    # # Define a color palette for different predictors
    # palette = sns.color_palette("viridis", num_predictor_types)
    # colors = ["red", "blue"]

    # # Overlay histograms for each predictor type with different colors
    # for i, predictor_data in enumerate(pred_likelihoods):
    #     sns.histplot(data=predictor_data, kde=True, ax=ax, color=colors[i], alpha=0.3, label=f'Predictor Type {i}', bins=int((max(predictor_data) - min(predictor_data)) / bin_size))
    
    # ax.set_xlabel('Negative Log Likelihood')
    # ax.set_ylabel('Frequency')
    # ax.legend()
    # plt.show()



    # lon_mse_df_set = []
    # lat_mse_df_set = []
    # for tmp_policy_name in policy_names:
    #     policy_dir = os.path.join(eval_dir, tmp_policy_name)
    #     tmp_post_path = os.path.join(policy_dir, tmp_policy_name + '.pkl')
    #     tmp_processed_data = pickle_read(tmp_post_path)                
    #     first_step_erros_lon_df, last_step_erros_lon_df, first_step_erros_lat_df, last_step_erros_lat_df = get_lon_lat_df(tmp_processed_data, tmp_policy_name)
    #     last_step_lon_mse_np_df, last_step_lat_mse_np_df = get_lat_lon_mse(tmp_processed_data, tmp_policy_name)


 
if __name__ == '__main__':
    main()