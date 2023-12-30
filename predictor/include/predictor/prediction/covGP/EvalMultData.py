
import numpy as np   
import os
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from predictor.common.utils.file_utils import *
from predictor.common.utils.scenario_utils import EvalData, PostprocessData, RealData

from predictor.h2h_configs import *
from predictor.common.utils.scenario_utils import wrap_del_s



def multi_policy_lat_lon_error_covs(real_data : RealData):
    """
    @param sim_data: Input evaluation data where we are comparing against `tar_states` (true trajectory)
    @return:
    lateral_error (list of l_1 errors)
    longitudinal_error (list of l_1 errors)
    """

    # class RealData():
    # track: RadiusArclengthTrack
    # N: int
    # ego_states: List[VehicleState]
    # tar_states: List[VehicleState]
    # tar_pred: List[VehiclePrediction] = field(default=List[VehiclePrediction])
    
    total_cov_list = []
    total_lateral_error = []
    total_longitunidal_error = []    
    track = real_data.track
    samps = 0
    init_eval_time_step = 0
    for timeStep in range(init_eval_time_step, len(real_data.tar_states)-1):
        
        lateral_error = []
        longitudinal_error = []
        pred = real_data.tar_pred[timeStep]  # (VehiclePrediction) at current timestep, what is GP prediction
        ego_states = real_data.ego_states[timeStep]
        tar_states = real_data.tar_states[timeStep] 
        # data_skip = True

        # if ego_states.p.s > 15.0 or ego_states.p.s < 3.0:
        #     data_skip = False
        
        del_s  = wrap_del_s(tar_states.p.s, ego_states.p.s, track)
        if abs(del_s) > 3.0:
            continue
        
        # if (tar_states.p.s - ego_states.p.s) < 3.0 and  (tar_states.p.s - ego_states.p.s) > 0.0:
        # # if abs(tar_states.p.s - ego_states.p.s) < track.track_length/2-0.2:
        #     data_skip = False
        # if data_skip: 
        #     continue

        if pred is not None and (pred.x is not None or pred.s is not None):
            N = len(pred.s) if pred.s else len(pred.x)
            if N + timeStep - 1 < len(real_data.tar_states):
                samps += 1

                covs = np.vstack([real_data.tar_pred[timeStep].xy_cov[:,0,0], real_data.tar_pred[timeStep].xy_cov[:,1,1]])
                total_cov_list.append(covs)
                for i in range(1, N):
                    tar_st = real_data.tar_states[timeStep + i]  # (VehicleState) current target state from true traveled trajectory
                    if not pred.s:
                        dx = tar_st.x.x - pred.x[i]
                        dy = tar_st.x.y - pred.y[i]
                        angle = real_data.track.local_to_global((tar_st.p.s, 0, 0))[2]
                        longitudinal = dx * np.cos(angle) + dy * np.sin(angle)
                        lateral = -dx * np.sin(angle) + dy * np.cos(angle)
                    else:
                        longitudinal = wrap_del_s(pred.s[i], tar_st.p.s, track)
                        # if abs(longitudinal) > track.track_length/4:
                        #     if pred.s[i] > track.track_length/4 and tar_st.p.s < track.track_length/4:
                        #         tmp = pred.s[i] - track.track_length/2
                        #         longitudinal = tmp - tar_st.p.s
                        #     elif pred.s[i] < track.track_length/4 and tar_st.p.s > track.track_length/4:
                        #         tmp = tar_st.p.s - track.track_length/2
                        #         longitudinal = pred.s[i] - tmp
                        #     else:
                        #         print("NA")
                    
                        lateral = pred.x_tran[i] - tar_st.p.x_tran                    
                    longitudinal_error.append(longitudinal)
                    lateral_error.append(lateral)
        
                total_longitunidal_error.append(np.array(longitudinal_error))
            
                total_lateral_error.append(np.array(lateral_error))
                
    return np.array(total_lateral_error), np.array(total_longitunidal_error), np.array(total_cov_list)




def get_process(policy_name, predictor_type = 0):
    
    policy_dir = os.path.join(multiEval_dir, policy_name)
    
    longitudinal_errors = []
    lateral_errors = []
    pred_covs = []

    for filename in os.listdir(policy_dir):
        tmp_str = '_'+str(predictor_type)+ '.pkl'
        if filename.endswith(tmp_str):
            # Construct the full file path
            filepath = os.path.join(policy_dir, filename)
            # Read and process the .pkl file
            with open(filepath, 'rb') as file:
                
                data = pickle.load(file) # RealData()
                if (len(data.tar_states) != len(data.tar_pred)) or len(data.tar_states) < 30:
                    continue 
                else:                        
                    lateral_error, longitudinal_error, pred_cov = multi_policy_lat_lon_error_covs(data)
                
                    # lateral_errors.append(np.sqrt(np.mean(lateral_error[:,-1] ** 2)))
                    lateral_errors.append(lateral_error[:,-1])
                    # longitudinal_errors.append(np.sqrt(np.mean(longitudinal_error[:,-1] ** 2)))
                    longitudinal_errors.append(longitudinal_error[:,-1])
                    pred_covs.append(np.mean(np.mean(pred_cov[:,:,:], axis=1), axis=1))
  
    pred_covs = np.concatenate(pred_covs)
    lateral_errors = np.concatenate(lateral_errors)
    longitudinal_errors = np.concatenate(longitudinal_errors)
    
    return lateral_errors, longitudinal_errors, pred_covs



def draw_bar_with_list(list_, policy_names, plot_name = None, value_name_ = None):

    df_list = []
    for j, policy_data in enumerate(list_):
        data = []
        for i, pred_data in enumerate(policy_data):                
            data.append(pred_data)
        data_np = np.transpose(np.array(data))                    
        df = pd.DataFrame(data_np, columns=['NOSIMGP', 'CAV', 'NLMPC', 'NaiveGP', 'SIMGP'])
        df['Policy'] = str(policy_names[j])
        df_list.append(df)

    cat_df = pd.concat(df_list)
    cat_df_melted = pd.melt(cat_df, id_vars='Policy', var_name='Policy_Name', value_name=value_name_)

    # Plot bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Policy', y=value_name_, hue='Policy_Name', data=cat_df_melted)

    # Draw a horizontal line at y=0
    plt.axhline(0, color='gray', linestyle='--')

    plt.xlabel('Policy')
    plt.ylabel(value_name_)

    # Save the figure
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(fig_dir, f"{value_name_}_bar_{current_time}.png")
    plt.savefig(file_path)        



def draw_histo_with_list(list_, policy_names, plot_name = None, value_name_ = None):
        
    df_list = []
    for j, policy_data in enumerate(list_):
        data = []
        for i, pred_data in enumerate(policy_data):                
            data.append(pred_data)
        data_np = np.transpose(np.array(data))                    
        df = pd.DataFrame(data_np, columns=['NOSIMGP', 'CAV', 'NLMPC', 'NaiveGP', 'SIMGP'])
        df['Policy'] = str(policy_names[j])
        df_list.append(df)

    cat_df = pd.concat(df_list)
    cat_df_melted = pd.melt(cat_df, id_vars='Policy', var_name='Policy_Name', value_name=value_name_)

    # Plot histograms for each policy
    g = sns.FacetGrid(cat_df_melted, col='Policy', hue='Policy_Name', sharey=False)
    g.map(sns.histplot, value_name_)
    plt.xlim((-1,1))
    # Draw a horizontal line at y=0
    plt.axhline(0, color='gray', linestyle='--')
    plt.legend()
    # Save the figure
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(fig_dir, f"{value_name_}_hist_{current_time}.png")
    plt.savefig(file_path)


def draw_barplot_with_list(list_, policy_names, plot_name = None, value_name_ = None):
    
    df_list = []
    for j, policy_data in enumerate(list_):
        data = []
        for i, pred_data in enumerate(policy_data):                
            data.append(pred_data)
        data_np = np.transpose(np.array(data))                    
        df = pd.DataFrame(data_np, columns=['NOSIMGP', 'CAV', 'NLMPC', 'NaiveGP', 'SIMGP'])
        df['Policy'] = str(policy_names[j])
        df_list.append(df)

    cat_df = pd.concat(df_list)
    cat_df_melted = pd.melt(cat_df, id_vars='Policy', var_name='Policy_Name', value_name=value_name_)
    error_plt= sns.catplot(x='Policy', y=value_name_, hue='Policy_Name', kind='box', data=cat_df_melted,showfliers = False)    
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Policy')
    plt.ylabel(value_name_)        
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(fig_dir, f"{value_name_}_{current_time}.png")
    plt.savefig(file_path)

