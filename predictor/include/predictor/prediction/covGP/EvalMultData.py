
import numpy as np   
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from predictor.common.utils.file_utils import *
from predictor.common.utils.scenario_utils import EvalData, PostprocessData, RealData

from predictor.h2h_configs import *


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
    init_eval_time_step = 20
    for timeStep in range(init_eval_time_step, len(real_data.tar_states)-1):
        
        lateral_error = []
        longitudinal_error = []
        pred = real_data.tar_pred[timeStep]  # (VehiclePrediction) at current timestep, what is GP prediction
        ego_states = real_data.ego_states[timeStep]
        tar_states = real_data.tar_states[timeStep] 
        # data_skip = True

        # if ego_states.p.s > 15.0 or ego_states.p.s < 3.0:
        #     data_skip = False
        
            
        
        
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
                        longitudinal = pred.s[i] - tar_st.p.s
                        lateral = pred.x_tran[i] - tar_st.p.x_tran                    
                    longitudinal_error.append(longitudinal)
                    lateral_error.append(lateral)
        
                total_longitunidal_error.append(np.array(longitudinal_error))
                total_lateral_error.append(np.array(lateral_error))
                
    return np.array(total_lateral_error), np.array(total_longitunidal_error), np.array(total_cov_list)


def main(args=None):
    

  
    def get_process(policy_name):
        
        policy_dir = os.path.join(multiEval_dir, policy_name)
        
        longitudinal_errors = []
        lateral_errors = []
        pred_covs = []

        for filename in os.listdir(policy_dir):
            if filename.endswith('.pkl'):
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

    # policy_names = ['eval_reverse_only_elbo', 'eval_reverse_withpcov', 'eval_reverse_naivegp']
    policy_names = ['centerline_1220', 'nonblocking_yet_racing_1220', 'blocking_1220', 'hjpolicy_1220', 'reverse_1220'] 
    # policy_names = [ 'mild_200', 'aggressive_blocking',  'mild_5000' ,'reverse']
    lateral_errors_list= []
    longitudinal_errors_list= []
    pred_covs_list= []
    for j in range(len(policy_names)):        
        lateral_errors, longitudinal_errors, pred_covs = get_process(policy_names[j])
        lateral_errors_list.append(lateral_errors)
        longitudinal_errors_list.append(longitudinal_errors)
        pred_covs_list.append(pred_covs)
    


    def draw_barplot_with_list(list_, plot_name = None):
        plt.figure()
        data = []
        for i, errors in enumerate(list_):
            for error in errors:
                data.append({'Policy': policy_names[i], plot_name: error})
        df = pd.DataFrame(data)


        # Create the bar plot
        sns.barplot(x='Policy', y=plot_name, data=df, capsize=0.2)

        # Adding labels and title
        plt.xlabel('Driving Policy')
        plt.ylabel(plot_name)
        plt.title(plot_name+ 'Statistics by Driving Policy')


    draw_barplot_with_list(lateral_errors_list, 'later')
    # draw_barplot_with_list(longitudinal_errors_list, 'long')
    draw_barplot_with_list(pred_covs_list, 'covs')
    
    plt.show()
if __name__ == '__main__':
    main()