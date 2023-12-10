#!/usr/bin/env python3
from barcgp.common.utils.scenario_utils import ScenarioGenParams, ScenarioGenerator
from MPCC_H2H_approx_build import MPCC_H2H_approx

from h2h_configs_build import *
from dynamics_models import CasadiDynamicsModel, CasadiDynamicBicycleFull
M = 50  # Number of samples for GP
T = 20  # Max number of seconds to run experiment
t = 0  # Initial time increment
    ##############################################################################################################################################

scen_params = ScenarioGenParams(types=['track'], egoMin=egoMin, egoMax=egoMax, tarMin=egoMin, tarMax=egoMax, width=width)
scen_gen = ScenarioGenerator(scen_params)
scenario = scen_gen.genScenario()    
track_name = scenario.track_type
track_obj = scenario.track
dynamic = CasadiDynamicBicycleFull(0, ego_dynamics_config, track=track_obj)

gp_mpcc_ego_controller = MPCC_H2H_approx(dynamic, track_obj, mpcc_ego_params, name="gp_mpcc_h2h_ego", track_name=track_name)
gp_mpcc_ego_controller.initialize()