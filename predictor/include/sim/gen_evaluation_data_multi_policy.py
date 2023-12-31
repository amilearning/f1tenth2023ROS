#!/usr/bin/env python3
import os
import pickle
import pathlib

import numpy as np
from collections import deque

from predictor.common.utils.file_utils import *

from predictor.common.pytypes import VehicleActuation, VehicleState, BodyLinearVelocity, ParametricPose, VehiclePrediction
from predictor.dynamics.models.model_types import DynamicBicycleConfig
from predictor.common.utils.scenario_utils import MultiPolicyEvalData, SimData, EvalData, smoothPlotResults, ScenarioGenParams, ScenarioGenerator

from predictor.controllers.MPCC_H2H_approx import MPCC_H2H_approx

from predictor.simulation.dynamics_simulator import DynamicsSimulator

import multiprocessing as mp
from predictor.h2h_configs import *
from predictor.prediction.trajectory_predictor import ConstantVelocityPredictor, ConstantAngularVelocityPredictor, GPPredictor, NoPredictor, MPCCPredictor, MPCPredictor
from predictor.common_control import run_pid_warmstart
from predictor.prediction.covGP.covGPNN_predictor import CovGPPredictor
import torch 

total_runs = 1
M = 50
# target_policy_name = 'timid'
# folder_name = 'timid'
target_policy_names = [ 'timid', 'aggressive_blocking',  'mild_5000' ,'reverse']
folder_names = [ 'timid_0', 'aggressive_blocking',  'mild_5000' ,'reverse']
Q_xref_set = [ 0.0, 500.0, 5000.0, 1000.0]
tp_model_name = 'tphmcl' # this is not used 
gp_model_name = 'gpberkely'
track_types = ['straight','curve', 'chicane']

T = 20


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
    'add_noise_data': False,
    'model_name' : None
    }

naivegp_args = args_.copy()
naivegp_args['model_name'] = 'naiveGP'

nosimtsGP_args = args_.copy()
nosimtsGP_args['model_name'] = 'nosimtsGP'

simtsGP_args = args_.copy()
simtsGP_args['model_name'] = 'simtsGP'


# policy_dir = os.path.join(eval_dir, folder_name)

# predictors = [GPPredictor(N, None, gp_model_name, True, M, cov_factor=np.sqrt(2)),
#                 # GPPredictor(N, None, gp_model_name, True, M, cov_factor=1),
#                     # GPPredictor(N, None, gp_model_name, True, M, cov_factor=np.sqrt(0.5)),                    
#                     CovGPPredictor(N, None, tp_model_name, True, M, cov_factor=np.sqrt(2)), 
#                 ConstantAngularVelocityPredictor(N, cov=.01),
#                 # ConstantAngularVelocityPredictor(N, cov=.005),
#                 # ConstantAngularVelocityPredictor(N, cov=.0),
#                 NLMPCPredictor(N, None, cov=.01, v_ref=mpcc_tv_params.vx_max),
#                 # NLMPCPredictor(N, None, cov=.005, v_ref=mpcc_tv_params.vx_max),
#                 # NLMPCPredictor(N, None, cov=.0, v_ref=mpcc_tv_params.vx_max)
#               ]

predictors = [      ConstantAngularVelocityPredictor(N, cov= .01),                                
                    MPCCPredictor(N, None, vehicle_config= mpcc_timid_params, cov=.01),
                    CovGPPredictor(N, None,  use_GPU=True, M=M, cov_factor=np.sqrt(2.0), input_predict_model = "naiveGP", args= naivegp_args),
                    CovGPPredictor(N, None,  use_GPU=True, M=M, cov_factor=np.sqrt(2.0), input_predict_model = "nosimtsGP", args= nosimtsGP_args),                    
                    CovGPPredictor(N, None,  use_GPU=True, M=M, cov_factor=np.sqrt(2.0), input_predict_model = "simtsGP", args= simtsGP_args)                 
              ]


"""ConstantVelocityPredictor(N), ConstantAngularVelocityPredictor(N),
NoPredictor(N), MPCPredictor(N), NLMPCPredictor(N, None)]"""
# names = ["GP2", "GP1", "GP_5 ","TP_2 ","TP_1 ","TP_5 ", "CAV_01", "CAV_005", "CAV0", "NLMPC_01", "NLMPC_005", "NLMPC0"]
# names = ["GP2", "GP_5 ","TP_2 ","TP_5 ", "CAV_01",  "CAV0", "NLMPC_01",  "NLMPC0"]
# names = ["GP_2", "TP_2 ","CAV_01", "NLMPC_01"]
names = [ "CAV", "MPCC", "naiveGP", "nosimtsGP", "simtsGP"]

def main(args=None):

    for j in range(len(target_policy_names)):
        target_policy_name = target_policy_names[j]
        folder_name = target_policy_name
        policy_dir = os.path.join(eval_dir, folder_name)

        mpcc_tv_params_ = mpcc_tv_params 
        mpcc_tv_params_.Q_xref = Q_xref_set[j]
        
        t = 0  # Initial time increment
        '''
        Collect the actual data
        '''
        # scen_params = ScenarioGenParams(types=['track'], egoMin=egoMin, egoMax=egoMax, tarMin=tarMin,
        scen_params = ScenarioGenParams(types=track_types, egoMin=egoMin, egoMax=egoMax, tarMin=tarMin,
                                        tarMax=tarMax,
                                        width=width)
        scen_gen = ScenarioGenerator(scen_params)
        ## 
        # build forcespro     
        scen = scen_gen.genScenario()    
        predictor_idx = 4
        predictors[predictor_idx].track = scen.track
        runSimulation(dt, t, N, names[j], predictors[predictor_idx], scen, 0, mpcc_tv_params_,target_policy_name, policy_dir, 0)
        # runSimulation(dt, t, N, names[j], predictors[2], scen, 0, mpcc_tv_params_,target_policy_name, policy_dir, 0)
        print("FORCESPRO INIT DONE")
        ## 
        gp_params = []    
        params = []
        d = 0
        for i in range(total_runs):
            scen = scen_gen.genScenario()
            offset =  0 # np.random.uniform(0, 30)
            for k in range(len(names)):
                predictors[k].track = scen.track

                if isinstance(predictors[k], GPPredictor) or  isinstance(predictors[k], CovGPPredictor):
                    gp_params.append((dt, t, N, names[k], predictors[k], scen, i+d,mpcc_tv_params_,target_policy_name, policy_dir, offset))                
                else:
                    params.append((dt, t, N, names[k], predictors[k], scen, i+d, mpcc_tv_params_,target_policy_name, policy_dir, offset))


        # print("Starting non-GP Evaluation!")
        # process_pool = mp.Pool(processes=8)
        # process_pool.starmap(runSimulation, params)
        # process_pool.close()
        # process_pool.join()
        
        print(len(params[0]))
        process_pool = mp.Pool(processes=10)
        process_pool.starmap(runSimulation, params)
        print("Closing!")
        process_pool.close()
        process_pool.join()

        print("Starting GP Evaluation!")
        for p in gp_params:
            runSimulation(*p)

        # for p in params:
        #     runSimulation(*p)


        
    

def runSimulation(dt, t, N, name, predictor, scenario, id,mpcc_tv_params_,target_policy_name, policy_dir, offset=0):
    print(f'{name} sim {id}')
    tv_inf, ego_inf = False, False
    track_obj = scenario.track

    ego_dynamics_simulator = DynamicsSimulator(t, ego_dynamics_config, track=track_obj)
    tar_dynamics_simulator = DynamicsSimulator(t, tar_dynamics_config, track=track_obj)
    tv_history, ego_history, vehiclestate_history, ego_sim_state, tar_sim_state, egost_list, tarst_list = run_pid_warmstart(
        scenario, ego_dynamics_simulator, tar_dynamics_simulator, n_iter=n_iter, t=t, offset=offset)

    if isinstance(predictor, GPPredictor) or isinstance(predictor, CovGPPredictor):
        mpcc_ego_controller = MPCC_H2H_approx(ego_dynamics_simulator.model, track_obj, gp_mpcc_ego_params, name="gp_mpcc_h2h_ego", track_name='track')
    else:
        mpcc_ego_controller = MPCC_H2H_approx(ego_dynamics_simulator.model, track_obj, mpcc_ego_params, name="mpcc_h2h_ego", track_name='track')
    mpcc_ego_controller.initialize()
    mpcc_ego_controller.set_warm_start(*ego_history)

    # mpcc_tv_controller = MPCC_H2H_approx(tar_dynamics_simulator.model, track_obj, mpcc_tv_params_, name="mpcc_h2h_tv", track_name='track')
    mpcc_tv_controller = MPCC_H2H_approx(tar_dynamics_simulator.model, track_obj, mpcc_tv_params_, name="mpcc_tv_params", track_name='track')
    mpcc_tv_controller.initialize()
    mpcc_tv_controller.set_warm_start(*tv_history)

    if isinstance(predictor, MPCCPredictor):        
        predictor.set_warm_start(tar_sim_state)

    gp_tarpred_list = [None] * n_iter
    egopred_list = [None] * n_iter
    tarpred_list = [None] * n_iter

    ego_prediction, tar_prediction, tv_pred = None, None, None
    while True:
        # if tar_sim_state.p.s >= 0.8 * scenario.length - offset or ego_sim_state.p.s >= 0.8 * scenario.length - offset or ego_sim_state.t > 37:
        
        if abs(ego_sim_state.p.x_tran) > (scenario.track.track_width/2.0)*0.99:
            break 
        if ego_sim_state.v.v_long < 0 or tar_sim_state.v.v_long < 0:
            break 
        if tar_sim_state.p.s >= 0.8 * scenario.length or ego_sim_state.p.s >= 0.8 * scenario.length:
            break
        else:
            if predictor:
                ego_pred = mpcc_ego_controller.get_prediction()
                if ego_pred.x is not None:                    
                    tv_pred = predictor.get_prediction(ego_sim_state, tar_sim_state, ego_pred, tar_prediction)
                    if tv_pred is None:
                        gp_tarpred_list.append(None)
                    else:
                        tv_pred.xy_cov =tv_pred.xy_cov * 1.0# np.clip(tv_pred.xy_cov,1.0,np.inf)
                        gp_tarpred_list.append(tv_pred.copy())
                else:
                    gp_tarpred_list.append(None)
            # update control inputs
            info, b, exitflag = mpcc_tv_controller.step(tar_sim_state, tv_state=ego_sim_state, tv_pred=ego_prediction, policy=target_policy_name)
            if not info["success"]:
                if not exitflag == 0:
                    tv_inf=True
                    pass
            info, b, exitflag = mpcc_ego_controller.step(ego_sim_state, tv_state=tar_sim_state, tv_pred=tv_pred)
            if not info["success"]:
                if not exitflag == 0:
                    ego_inf=True
                    pass

            # step forward
            tar_prediction = None if not mpcc_tv_controller.get_prediction() else mpcc_tv_controller.get_prediction().copy()
            tar_prediction.t = tar_sim_state.t
            tar_dynamics_simulator.step(tar_sim_state)
            track_obj.update_curvature(tar_sim_state)

            ego_prediction = None if not mpcc_ego_controller.get_prediction() else mpcc_ego_controller.get_prediction().copy()
            ego_prediction.t = ego_sim_state.t
            # ego_prediction.xy_cov = np.repeat(np.diag([0.001, 0.001])[np.newaxis, :, :], 11, axis=0)
            ego_dynamics_simulator.step(ego_sim_state)

            # log states
            egost_list.append(ego_sim_state.copy())
            tarst_list.append(tar_sim_state.copy())
            egopred_list.append(ego_prediction)
            tarpred_list.append(tar_prediction)
            # print('Current time', ego_sim_state.t)


    scenario_sim_data = EvalData(scenario, len(egost_list), egost_list, tarst_list, egopred_list, tarpred_list, gp_tarpred_list, tv_infeasible=tv_inf, ego_infeasible=ego_inf)
    
    if isinstance(predictor, GPPredictor) or isinstance(predictor, CovGPPredictor):
        ego_config = gp_mpcc_ego_params
    else:
        ego_config = mpcc_ego_params

    multi_policy_sim_data = MultiPolicyEvalData(ego_config = ego_config, tar_config = mpcc_tv_params_, evaldata = scenario_sim_data)
    root_dir = os.path.join(policy_dir, scenario.track_type)
    create_dir(path=root_dir)
    root_dir = os.path.join(root_dir, name)
    create_dir(path=root_dir)
    pickle_write(multi_policy_sim_data, os.path.join(root_dir, str(id) + '.pkl'))

    if total_runs == 1:        
        smoothPlotResults(scenario_sim_data, speedup=1.6, close_loop=False)

if __name__ == '__main__':
    main()
