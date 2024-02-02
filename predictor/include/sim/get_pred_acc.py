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

def compute_liklihood(target_state: VehicleState, pred: VehiclePrediction, pred_horizon_idx):
    
    pose = np.array([target_state.x.x, target_state.x.y])
    mean = np.array([pred.x[pred_horizon_idx], pred.y[pred_horizon_idx]])    
    cov = np.array([[pred.xy_cov[pred_horizon_idx][0][0], 0], [0, pred.xy_cov[pred_horizon_idx][1][1]]])    
    jitter = np.eye(2) * 1e-12
    cov +=jitter
    # Calculate the determinant of the covariance matrix    
    det_cov = np.linalg.det(cov)
    # Calculate the inverse of the covariance matrix
    cov_inv = np.linalg.inv(cov)
    # Calculate the likelihood
    likelihood = (1 / (np.sqrt((2 * np.pi)**2 * det_cov))) * np.exp(-0.5 * (pose - mean).T @ cov_inv @ (pose - mean))
    negative_log_liklihood = -np.log(likelihood) 

    tar_pose_and_likelihood = np.array([target_state.x.x, target_state.x.y, negative_log_liklihood])
    return tar_pose_and_likelihood

def main(args=None):
    

  
    # policy_names = ['timid', 'mild_200', 'aggressive_blocking',  'mild_5000' ,'reverse']
    # policy_names = ['mild_200', 'aggressive_blocking', 'mild_5000', 'reverse']    
    predictor_names = ['dkl_blocking', 'hdkl_blocking']
    
    liklihood_traces_list = []

    invalid_data_count = 0
    for predictor_name in predictor_names:
        ab_p = os.path.join(lik_dir,predictor_name)
        liklihood_traces = []
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
                        if abs(delta_s) < 100.0:
                            
                            tar_pred = scenario_data.tar_pred[t]
                            pred_horizon = len(tar_pred.s)
                            terminal_tar_st = scenario_data.tar_states[t+pred_horizon-1]
                            terminal_liklihood = compute_liklihood(terminal_tar_st, tar_pred, pred_horizon-1)
                            liklihood_traces.append(terminal_liklihood.copy())
        
                dbfile.close()
        liklihood_traces_list.append(liklihood_traces)
        
        
    
    
    likelihoods = []
    for i, predictor_data in enumerate(liklihood_traces_list):
        # Extract x, y positions, and likelihood for the current predictor type
        x, y, likelihood = zip(*predictor_data)
        
        
        likelihoods.append(likelihood)

    pred_likelihoods = likelihoods
    likelihoods = np.hstack(likelihoods)
    min_val = likelihoods.min()
    max_val = likelihoods.max()


    sns.set(style="white")

    predictor_data = pred_likelihoods

    num_predictor_types = len(predictor_data)
    fig, ax = plt.subplots(figsize=(8, 4))

    colors = ["red", "blue"]

    for i, data in enumerate(predictor_data):
        sns.histplot(data=data, kde=True, ax=ax, color=colors[i], alpha=0.5, label=f'{["DKL", "H-DKL"][i]}')

    ax.set_xlabel('Negative Log Likelihood', fontsize=14)
    ax.legend(fontsize=14)
    ax.set_xlim(-1.4, 8)  # Adjust the limits as needed

    plt.show()


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