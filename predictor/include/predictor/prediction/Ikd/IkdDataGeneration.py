#!/usr/bin/env python3

import multiprocessing as mp
from collections import deque
import numpy as np
import random

from barcgp.common.utils.file_utils import *
from barcgp.dynamics.models.model_types import DynamicBicycleConfig
from barcgp.common.utils.scenario_utils import IkdData, ScenarioGenParams, ScenarioGenerator
from barcgp.common_control import run_pid_warmstart
from barcgp.controllers.NL_MPC import NL_MPC

from barcgp.controllers.MPCC_H2H_approx import MPCC_H2H_approx
from barcgp.simulation.dynamics_simulator import DynamicsSimulator
from barcgp.h2h_configs import *


total_runs = 1000

policy_name = 'IDK'
policy_dir = os.path.join(train_dir, policy_name)
track_types = ['straight', 'curve', 'chicane']
T = 10

n_core = mp.cpu_count()
print("The number of cpu core : ",n_core)
if n_core > 10:
    n_core = 10
else:
    n_core = 5



def main(args=None):
    t = 0  # Initial time increment
    '''
    Collect the actual data
    '''

    scen_params = ScenarioGenParams(types=track_types, egoMin=IKD_egoMin, egoMax=IKD_egoMax,tarMin=tarMin,
                                    tarMax=tarMax,width=IKD_width)
    scen_gen = ScenarioGenerator(scen_params)

    ## single process
    # for i in range(total_runs):
    #     runSimulation(dt,t,N,scen_gen.genScenario(), i)

    ## multi-process
    params = []
    process_pool = mp.Pool(processes=n_core)
    for i in range(total_runs):
        params.append((dt, t, N, scen_gen.genScenario(), i))
    process_pool.starmap(runSimulation, params)
    process_pool.close()
    process_pool.join()

def runSimulation(dt, t, N, scenario, id):

    track_obj = scenario.track

    ego_dynamics_simulator = DynamicsSimulator(t, ego_dynamics_config, track=track_obj)
    tar_dynamics_simulator = DynamicsSimulator(t, tar_dynamics_config, track=track_obj)

    
    tv_history, ego_history, _, ego_sim_state, tar_sim_state, egost_list, tarst_list = run_pid_warmstart(
        scenario, ego_dynamics_simulator, tar_dynamics_simulator, n_iter=n_iter, t=t)


    mpcc_ego_controller = MPCC_H2H_approx(ego_dynamics_simulator.model, track_obj, mpcc_ego_params, name="mpcc_h2h_ego")

    mpcc_ego_controller.initialize()
    mpcc_ego_controller.set_warm_start(*ego_history)
    
    egopred_list = [None] * n_iter

    ego_prediction = None
    done = False
    prev_ego_sim_state = ego_sim_state.copy()
    update_count = 0
    # print('Current time', ego_sim_state.t)
    randomness = int(random.uniform(0,5))
    tar_prediction = None
    while ego_sim_state.t < T and not done:
        if ego_sim_state.p.s >= 1.9 * scenario.length:
            break
        elif ego_sim_state.v.v_long >= mpcc_ego_params.vx_max*2.0:
            break
        else:
            # update control inputs
            # step forward
            
            info, b, exitflag = mpcc_ego_controller.step(ego_sim_state, tv_state=tar_sim_state, tv_pred=tar_prediction)                       
            random_input_switch = random.uniform(-1,randomness)
            if random_input_switch > 0:
                mpcc_ego_controller.random_step( ego_sim_state)
            
         
            # print("solve time : ", info['solve_time'])
            


            ego_prediction = mpcc_ego_controller.get_prediction().copy()
            ego_prediction.t = ego_sim_state.t
            ego_prediction.xy_cov = np.repeat(np.diag([0.001, 0.001])[np.newaxis, :, :], 11, axis=0)
            ego_dynamics_simulator.step(ego_sim_state)
            
            egost_list.append(ego_sim_state.copy())
            egopred_list.append(ego_prediction)
            prev_ego_sim_state = ego_sim_state.copy()
        
    scenario_sim_data = IkdData(scenario, len(egost_list), egost_list)
    root_dir = os.path.join(policy_dir, scenario.track_type)
    create_dir(path=root_dir)
    pickle_write(scenario_sim_data, os.path.join(root_dir, str(id) + '.pkl'))
    

    print("finished")

if __name__ == '__main__':
    main()