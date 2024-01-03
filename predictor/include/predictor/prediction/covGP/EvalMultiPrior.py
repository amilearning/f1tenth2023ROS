#!/usr/bin/env python3
"""   
 Software License Agreement (BSD License)
 Copyright (c) 2022 Ulsan National Institute of Science and Technology (UNIST)
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************** 
  @author: Hojin Lee <hojinlee@unist.ac.kr>
  @date: September 10, 2022
  @copyright 2022 Ulsan National Institute of Science and Technology (UNIST)
  @brief: ROS node for sampling based nonlinear model predictive controller (Model predictive path integrator) 
  @details: main node for MPPI
"""
import threading
import pickle
import rospy
import os
import numpy as np 
import rospkg
from visualization_msgs.msg import MarkerArray
from predictor.common.utils.scenario_utils import EvalData, PostprocessData, RealData
from predictor.common.pytypes import VehicleState, ParametricPose, BodyLinearVelocity, VehicleActuation
from predictor.path_generator import PathGenerator
from predictor.prediction.covGP.covGPNN_predictor import CovGPPredictor
from predictor.prediction.trajectory_predictor import ConstantAngularVelocityPredictor, NLMPCPredictor, GPPredictor, MPCCPredictor
from predictor.h2h_configs import *
from predictor.common.utils.file_utils import *
from predictor.simulation.dynamics_simulator import DynamicsSimulator
from predictor.controllers.MPCC_H2H_approx import MPCC_H2H_approx
from predictor.utils import prediction_to_marker
from predictor.controllers.utils.controllerTypes import PIDParams
from predictor.controllers.PID import PIDLaneFollower
from predictor.dynamics.models.dynamics_models import CasadiDynamicBicycleFull
from collections import deque
import gpytorch

from torch.utils.data import DataLoader
import torch
from predictor.prediction.covGP.covGPNN_dataGen import SampleGeneartorCOVGP
from predictor.common.utils.scenario_utils import wrap_s_np
from datetime import datetime


# folder_name = ['centerline_1220', 'nonblocking_yet_racing_1220', 'blocking_1220', 'hjpolicy_1220', 'reverse_1220'] 
# centerline_1220 = os.path.join(real_dir, folder_name[0])
# nonblocking_yet_racing_1220 = os.path.join(real_dir, folder_name[1])
# blocking_1220 = os.path.join(real_dir, folder_name[2])
# hjpolicy_1220 = os.path.join(real_dir, folder_name[3])
# reverse_1220 = os.path.join(real_dir, folder_name[4])
# dirs = [centerline_1220, nonblocking_yet_racing_1220, blocking_1220, hjpolicy_1220, reverse_1220]

# folder_name = ['test']
# test_folder = os.path.join(real_dir, folder_name[0])
# dirs = [test_folder]



# rospack = rospkg.RosPack()
# pkg_dir = rospack.get_path('predictor')

class MultiPredPostEval:
    def __init__(self, dirs, args = None):
        self.dirs = dirs
        self.visualize = True
        self.n_nodes = rospy.get_param('~n_nodes', default=10)
        self.t_horizon = rospy.get_param('~t_horizon', default=1.0)                   
        self.torch_device = "cuda:0"   ## Specify the name of GPU 
        # self.torch_dtype  = torch.double
        self.dt = self.t_horizon / self.n_nodes*1.0        
        ## 
        # Generate Racing track info 
        self.track_load = False
        track_file_path = os.path.join(track_dir, 'track.pickle')        
        self.track_info = PathGenerator()
        if self.track_load:
            with open(track_file_path, 'rb') as file:            
                self.track_info = pickle.load(file)
        else: ## save track
            with open(track_file_path, 'wb') as file:
                pickle.dump(self.track_info, file)
        

         
        self.driving_policies = [dir.split('/')[-1] for dir in self.dirs]
        prediction_types  = 5  # List of prediction types
        self.error_data = {policy: {str(ptype): {"lon": [], "lat": []} 
                        for ptype in range(prediction_types)} 
                for policy in self.driving_policies}
        
        
                
        
# Open the file in binary mode for writing

        self.track_info = PathGenerator()        
        while self.track_info.track_ready is False:
             rospy.sleep(0.01)
        ##
        use_GPU = True
        M = 25
        self.cur_ego_state = VehicleState(t=0.0,
                                      p=ParametricPose(s=0.1, x_tran=0.2, e_psi=0.0),
                                      v=BodyLinearVelocity(v_long=0.5),
                                      u=VehicleActuation(t=0.0, u_a=0.0, u_steer = 0.0))
        self.cur_tar_state = VehicleState(t=0.0,
                                      p=ParametricPose(s=1.0, x_tran=0.0, e_psi=0.0),
                                      v=BodyLinearVelocity(v_long=0.5),
                                      u=VehicleActuation(t=0.0, u_a=0.0, u_steer = 0.0))
          
        self.tv_pred_marker_pub = rospy.Publisher('/tv_pred',MarkerArray,queue_size = 2)
        self.ego_pred_marker_pub = rospy.Publisher('/ego_pred',MarkerArray,queue_size = 2)
        # prediction callback   
        self.tv_pred = None
        
        ### setup ego controller to compute the ego prediction 
        self.vehicle_model = CasadiDynamicBicycleFull(0.0, ego_dynamics_config, track=self.track_info.track)
        self.gp_mpcc_ego_controller = MPCC_H2H_approx(self.vehicle_model, self.track_info.track, control_params = gp_mpcc_ego_params, name="gp_mpcc_h2h_ego", track_name="test_track")        
        self.ego_warm_start()


        ############################## MPCC Predictor ##################################
        self.mpcc_predictor = MPCCPredictor(N=self.n_nodes, track=self.track_info.track, vehicle_config= mpcc_timid_params, cov=.01)
        ############################## ContantAngularVelocity Predictor ##################################
        self.cav_predictor = ConstantAngularVelocityPredictor(N=self.n_nodes, cov= .01)            
        ############################## SIMTSGP Predictor ####################################        
        self.predictor = CovGPPredictor(N=self.n_nodes, track=self.track_info.track,  use_GPU=use_GPU, M=M, cov_factor=np.sqrt(2.0), input_predict_model = "simtsGP", args= args.copy())                    
        ############################## NoSIMTSGP Predictor ####################################        
        self.predictor_withoutCOV = CovGPPredictor(N=self.n_nodes, track=self.track_info.track,  use_GPU=use_GPU, M=M, cov_factor=np.sqrt(2.0), input_predict_model = "nosimtsGP", args= args.copy())                    
        ############################## NaiveGP Predictor ####################################        
        self.predictor_naivegp = CovGPPredictor(N=self.n_nodes, track=self.track_info.track,  use_GPU=use_GPU, M=M, cov_factor=np.sqrt(2.0), input_predict_model = "naiveGP", args= args.copy())                            
        
        ### EVAL init  ########
        
        if args['load_eval_data']:
            load_data_name = 'error_dict_20240103_130834.pkl'
            eval_data_load = os.path.join(multiEval_dir, load_data_name)
            if os.path.exists(eval_data_load):
                self.error_data = pickle_read(eval_data_load)
            else:
                print("load eval data not found ")
                return 
        else:            
            self.pred_eval(args = args, predictor_type = 4)                
            self.pred_eval(args = args, predictor_type = 2)             
            self.pred_eval(args = args, predictor_type = 1)        
            self.pred_eval(args = args, predictor_type = 0)
            self.pred_eval(args= args, predictor_type = 3)
            print("gen done")
        
            if self.error_data is not None:
                self.save_errors()            
        
        self.plot_errors()

        
    def save_errors(self):  
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        pickle_write(self.error_data, os.path.join(multiEval_dir, f"error_dict_{current_time}.pkl"))            

    def plot_errors(self):
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns

        predictor_labels = {
            '0': "nosimtsGP",
            '1': "CAV",
            '2': "MPCC",
            '3': "naiveGP",
            '4': "simtimGP"
            # ... Add custom labels for each ptype
            }

        records = []  # List to store dataframe records

        # Iterate through the nested dictionary and create records
        for policy, ptypes in self.error_data.items():
            for ptype, errors in ptypes.items():
                for lon_error, lat_error in zip(errors["lon"], errors["lat"]):
                    # Create a dictionary for each record
                    record = {
                        "DrivingPolicy": policy,
                        "PredictorType": predictor_labels[ptype],
                        "Lon_Error": lon_error[-1],
                        "Lat_Error": lat_error[-1]
                    }
                    # Add the record to the list
                    records.append(record)

        # Convert the records to a pandas DataFrame
        error_df = pd.DataFrame(records)

        predictor_types = error_df['PredictorType'].unique()
        drivign_policies = error_df['DrivingPolicy'].unique()


            # Create a list of labels in the order the histograms were plotted
        # predictor_labels = [predictor_labels[ptype] for ptype in predictor_types]

##########################################################################
##########################################################################
        plt.figure()
        sns.boxplot(x='DrivingPolicy', y='Lon_Error', hue='PredictorType', data=error_df, showfliers=False)
        
        plt.axhline(0, color='gray', linestyle='--')
        # Additional plot formatting
        plt.title('Longitudinal Error by Policy Type and Name')
        plt.xlabel('Policy')
        plt.ylabel('Longitudinal Error')
        

        # Adjust the legend or axes as needed, for example, to prevent overlapping:
        plt.legend(bbox_to_anchor=(0.01, 0.95),title='Predictor Type', loc='upper left', borderaxespad=0)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(fig_dir, f"lon_evalbar_{current_time}.png")
        plt.savefig(file_path)   
##########################################################################
##########################################################################
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='DrivingPolicy', y='Lat_Error', hue='PredictorType', data=error_df, showfliers=False)
        plt.axhline(0, color='gray', linestyle='--')
        # Additional plot formatting
        plt.title('Lateral Error by Policy Type and Name')
        plt.xlabel('Policy')
        plt.ylabel('Lateral Error')        
        plt.legend(bbox_to_anchor=(0.01, 0.95),title='Predictor Type', loc='upper left', borderaxespad=0)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(fig_dir, f"lat_evalbar_{current_time}.png")
        plt.savefig(file_path)   
##########################################################################
##########################################################################
      
##########################################################################
##########################################################################        
            
        n_bins = 50
        fig, axes = plt.subplots(1, len(drivign_policies), figsize=(20, 6), sharey=True)  # Adjust the figure size as needed
        for idx, policy in enumerate(drivign_policies):
            subsets = error_df[error_df['DrivingPolicy'] == policy]            
            for ptype in predictor_types:     
                subset = subsets[subsets['PredictorType'] == ptype]
                # Plot a histogram for the subset
                axes[idx].hist(subset['Lat_Error'], bins=n_bins, alpha=0.4, label=ptype)
            
            axes[idx].axvline(0, color='gray', linestyle='--')
            max_error = np.max(np.abs(subsets['Lat_Error']))
            axes[idx].set_xlim(-max_error, max_error)
            # Additional plot formatting
            axes[idx].set_title(f'Lateral Error by {policy}')
            axes[idx].set_xlabel('Lateral Error')
            axes[idx].set_ylabel('Frequency')

        plt.legend(bbox_to_anchor=(0.01, 0.95),title='Predictor Type', loc='upper left', borderaxespad=0)
        plt.tight_layout()
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(fig_dir, f"lat_hist_{policy}_{current_time}.png")
        plt.savefig(file_path) 
       

##########################################################################
##########################################################################        
        fig, axes = plt.subplots(1, len(drivign_policies), figsize=(20, 6), sharey=True)  # Adjust the figure size as needed
        for idx, policy in enumerate(drivign_policies):
            subsets = error_df[error_df['DrivingPolicy'] == policy]            
            for ptype in predictor_types:     
                subset = subsets[subsets['PredictorType'] == ptype]
                # Plot a histogram for the subset
                axes[idx].hist(subset['Lon_Error'], bins=n_bins, alpha=0.4, label=ptype)
            
            axes[idx].axvline(0, color='gray', linestyle='--')
            max_error = np.max(np.abs(subsets['Lon_Error']))
            axes[idx].set_xlim(-max_error, max_error)
            # Additional plot formatting
            axes[idx].set_title(f'Longitudinal Error by {policy}')
            axes[idx].set_xlabel('Longitudinal Error')
            axes[idx].set_ylabel('Frequency')

        plt.legend(bbox_to_anchor=(0.01, 0.95),title='Predictor Type', loc='upper left', borderaxespad=0)
        plt.tight_layout()
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(fig_dir, f"lon_hist_{policy}_{current_time}.png")
        plt.savefig(file_path) 
       



    def pred_eval_parallel(self):
        threads = []
        for i in range(5):
            # Create a thread for each predictor type
            thread = threading.Thread(target=self.pred_eval, args=(i,))
            threads.append(thread)
            thread.start()


        # Wait for all threads to complete
        for thread in threads:
            thread.join()

    # def add_noise(self, ego_state, tar_state):

    def pred_eval(self, args = None, predictor_type = 0, snapshot_name = None, load_data = False):
        if predictor_type ==0:
            args['model_name'] = 'nosimtsGP'
        elif predictor_type ==1:
            args['model_name'] = None # CAV Predictor
        elif predictor_type ==2:
            args['model_name'] = None # MPCC Predictor
            self.mpcc_predictor.set_warm_start(self.cur_tar_state)
        elif predictor_type ==3: 
            args['model_name'] = 'naiveGP'
        elif predictor_type ==4:
            args['model_name'] = 'simtsGP'
        else:   
            return


        
        
        for j in range(len(self.dirs)):
            tmp_dir = os.path.join(multiEval_dir,self.dirs[j].split('/')[-1])
            driving_policy_name = tmp_dir.split('/')[-1]
            create_dir(path=tmp_dir)   
            dir = [self.dirs[j]]                        
           
            sampGen = SampleGeneartorCOVGP(dir, load_normalization_constant = True, args = args, randomize=False, real_data = True, tsne = True)
            
            input_buffer_list, ego_state_list, tar_state_list, gt_tar_state_list, ego_pred_list = sampGen.get_eval_data(dir,real_data = True, noise = False)            
            pred_tar_state_list = []
            self.tv_pred = None ## Assume each directory contains single time race
            self.ego_warm_start()
            
            
            for i in range(len(input_buffer_list)):                
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    input_buffer = input_buffer_list[i]
                    ego_state = ego_state_list[i]
                    tar_state = tar_state_list[i]
                    
                    _, problem, cur_obstacles = self.gp_mpcc_ego_controller.step(ego_state, tv_state=tar_state, tv_pred=self.tv_pred)                    
                    ego_pred = self.gp_mpcc_ego_controller.get_prediction()
                    ego_pred = ego_pred_list[i]
                    if predictor_type == 0:
                        self.tv_pred = self.predictor_withoutCOV.get_eval_prediction(input_buffer, ego_state, tar_state, ego_pred)                         
                    elif predictor_type == 1:
                        self.tv_pred =  self.cav_predictor.get_prediction(ego_state = ego_state, target_state = tar_state, ego_prediction = ego_pred)                               
                    elif predictor_type == 2:
                        self.tv_pred = self.mpcc_predictor.get_prediction(ego_state = ego_state, target_state = tar_state, ego_prediction = ego_pred)
                    elif predictor_type == 3:
                        # self.tv_pred = self.gp_predictor.get_prediction(ego_state = ego_state, target_state = tar_state, ego_prediction = ego_pred)
                        self.tv_pred = self.predictor_naivegp.get_eval_prediction(input_buffer, ego_state, tar_state, ego_pred)                        
                    elif predictor_type == 4:
                        self.tv_pred = self.predictor.get_eval_prediction(input_buffer, ego_state, tar_state, ego_pred)
                    else: 
                        print("select predictor")
                    
                    if self.tv_pred.s is not None:
                        # self.track_info.track.global_to_local([self.tv_pred.x, self.tv_pred.y, self.tv_pred.psi])                        
                        lon_error = np.array(gt_tar_state_list[i].s) - np.array(self.tv_pred.s) 
                        lat_error = np.array(gt_tar_state_list[i].x_tran) - np.array(self.tv_pred.x_tran)
                    else: 
                        lon_error = [] #np.array(gt_tar_state_list[i].s) - np.array(gt_tar_state_list[i].s)
                        lat_error = [] #np.array(gt_tar_state_list[i].x_tran) - np.array(gt_tar_state_list[i].x_tran)
                        for idx in range(len(self.tv_pred.x)):                                                    
                            tv_fren = self.track_info.track.global_to_local((self.tv_pred.x[idx], self.tv_pred.y[idx], self.tv_pred.psi[idx]))
                            if tv_fren is not None:
                                lon = gt_tar_state_list[i].s[idx] - tv_fren[0] 
                                lat = gt_tar_state_list[i].x_tran[idx] - tv_fren[1] 
                            lon_error.append(lon)
                            lat_error.append(lat)
                        lon_error = np.array(lon_error)
                        lat_error = np.array(lat_error)
                    
                    ## semi-wrapping for track length
                    lon_error[lon_error < -self.track_info.track.track_length/4] += self.track_info.track.track_length/2            
                    lon_error[lon_error >= self.track_info.track.track_length/4] -= self.track_info.track.track_length/2    

                    self.error_data[driving_policy_name][str(predictor_type)]['lon'].append(lon_error)
                    self.error_data[driving_policy_name][str(predictor_type)]['lat'].append(lat_error)
                    # self.multi_pred_lon_err[predictor_type].append(lon_error)
                    # self.multi_pred_lat_err[predictor_type].append(lat_error)

                    

                    if self.visualize:
                        tv_pred_markerArray = prediction_to_marker(self.tv_pred)
                        self.tv_pred_marker_pub.publish(tv_pred_markerArray)
                        ego_pred_markerArray = prediction_to_marker(ego_pred)
                        self.ego_pred_marker_pub.publish(ego_pred_markerArray)

                    pred_tar_state_list.append(self.tv_pred.copy())
                    
                    if i % 60 == 0:                         
                        print("Sample : {} out of {}".format(i, len(input_buffer_list)))    
            tmp_real_data = RealData(self.predictor.track, len(input_buffer_list),ego_state_list, tar_state_list, pred_tar_state_list)
            pickle_write(tmp_real_data, os.path.join(tmp_dir, 'predictor_'  +str(i) + '_'+ str(predictor_type) +'.pkl'))            
            
        
            # if self.predictor and self.cur_ego_state is not None:                
            #     _, problem, cur_obstacles = self.gp_mpcc_ego_controller.step(self.cur_ego_state, tv_state=self.cur_tar_state, tv_pred=self.tv_pred)
            #     ego_pred = self.gp_mpcc_ego_controller.get_prediction()
                
            #     if self.cur_ego_state.t is not None and self.cur_tar_state.t is not None and ego_pred.x is not None:            
                
            


    def ego_warm_start(self):
        cur_state_copy = self.cur_ego_state.copy()
        x_ref = cur_state_copy.p.x_tran
        
        pid_steer_params = PIDParams()
        pid_steer_params.dt = self.dt
        pid_steer_params.default_steer_params()
        pid_steer_params.Kp = 1
        pid_speed_params = PIDParams()
        pid_speed_params.dt = self.dt
        pid_speed_params.default_speed_params()
        pid_controller_1 = PIDLaneFollower(cur_state_copy.v.v_long, x_ref, self.dt, pid_steer_params, pid_speed_params)
        ego_dynamics_simulator = DynamicsSimulator(0.0, ego_dynamics_config, track=self.track_info.track) 
        input_ego = VehicleActuation()
        t = 0.0
        state_history_ego = deque([], self.n_nodes); input_history_ego = deque([], self.n_nodes)
        n_iter = self.n_nodes+1
        approx = True
        while n_iter > 0:
            pid_controller_1.step(cur_state_copy)
            ego_dynamics_simulator.step(cur_state_copy)            
            self.track_info.track.update_curvature(cur_state_copy)
            input_ego.t = t
            cur_state_copy.copy_control(input_ego)
            q, _ = ego_dynamics_simulator.model.state2qu(cur_state_copy)
            u = ego_dynamics_simulator.model.input2u(input_ego)
            if approx:
                q = np.append(q, cur_state_copy.p.s)
                q = np.append(q, cur_state_copy.p.s)
                u = np.append(u, cur_state_copy.v.v_long)
            state_history_ego.append(q)
            input_history_ego.append(u)
            t += self.dt
            n_iter-=1
           
        compose_history = lambda state_history, input_history: (np.array(state_history), np.array(input_history))
        ego_warm_start_history = compose_history(state_history_ego, input_history_ego)
        self.gp_mpcc_ego_controller.initialize()
        self.gp_mpcc_ego_controller.set_warm_start(*ego_warm_start_history)
        

            


# def main():
    
#     MultiPredPostEval()

# if __name__ == "__main__":
#     main()




 
    


