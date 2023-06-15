from predictor.common.pytypes import VehicleState, VehiclePrediction, VehicleActuation, ParametricPose, BodyLinearVelocity
from abc import abstractmethod

import numpy as np
# import copy
# from typing import List

# GP imports
import gc
# import torch, gpytorch
import torch

from predictor.prediction.gp_controllers import GPControllerTrained
from predictor.common.tracks.radius_arclength_track import RadiusArclengthTrack

from predictor.h2h_configs import nl_mpc_params, N
from predictor.dynamics.models.model_types import DynamicBicycleConfig
from predictor.simulation.dynamics_simulator import DynamicsSimulator
# from predictor.controllers.NL_MPC import NL_MPC
from predictor.controllers.utils.controllerTypes import *


class BasePredictor():
    '''
    Base class for trajectory predictors

    Implements basic logic in _add_predictor, _restart_predictor, and _destroy_predictor that will suffice for most prediction setups

    General format is to provide
        env_state - a dictionary with keys that correspond to names of vehicles and values that correspond to the current estimated state of the vehicle
        t - the current time.

    The timestamp of the measurements in env_state is ignored by this class
        - when checking if a predictor is outdated it checks how long it has been since data appeared for a named vehicle
        - because of this, repeatedly supplying outdated data can cause unexpected behavior

    Additionally, no state estimation / filtering of provided vehicle states is performed (by default) - this must be implemented in a subclass or elsewhere if desired.
    '''

    def __init__(self, N=10, track: RadiusArclengthTrack = None, interval=0.1, startup_cycles=5, clear_timeout=1, destroy_timeout=5,  cov: float = 0):
        self.N = N
        self.track = track
        self.dt = interval
        self.startup_cycles_needed = startup_cycles
        self.cov = cov
        self.startup_cycles_done = dict()  # startup cycles completed for a particular vehicle name
        self.predictions = dict()  # predictions for each vehicle - should be either None or a VehicleCoords object with array-valued fields
        self.memory_dict = dict()  # optional dict for storing memory related to a vehicle in, e.g. for state estimation
        self.last_update = dict()  # dict for storing last time each channel was updated

        self.clear_timeout = clear_timeout  # allowed time without measurements before a named vehicle predictor is reset
        self.destroy_timeout = destroy_timeout  # allowed time without measurements before a named vehicle predictor is destroyed
        return

    

    def update(self, env_state, t):
        self.last_env_state = env_state.copy()
        self._update_env_state(env_state, t)
        self._check_stale_vehicles(env_state, t)

        return self.predictions

    def _update_env_state(self, env_state, t):
        '''
        Updates all predictors for which data is available
        '''
        for vehicle_name in env_state.keys():
            self.last_update[vehicle_name] = t
            vehicle_state = env_state[vehicle_name]

            if vehicle_name not in self.predictions.keys():
                self._add_predictor(vehicle_name, vehicle_state, t)

            elif t - self.last_update[vehicle_name] > self.clear_timeout:
                self._restart_predictor(vehicle_name, vehicle_state, t)

            else:
                self._update_predictor(vehicle_name, vehicle_state, t)
        return

    # def _check_stale_vehicles(self, env_state, t):
    #     '''
    #     Remove predictors that haven't shown up in a long time
    #     '''
    #     missing_names = set(self.predictions).difference(set(env_state))
    #     for mn in missing_names:
    #         if t - self.last_update[mn] > self.destroy_timeout:
    #             self._destroy_predictor(vehicle_name)

    def _add_predictor(self, vehicle_name, vehicle_state, t):
        self.predictions[vehicle_name] = None
        self._restart_predictor(vehicle_name, vehicle_state, t)
        return

    def _restart_predictor(self, vehicle_name, vehicle_state, t):
        self.startup_cycles_done[vehicle_name] = 0
        self.memory_dict[vehicle_name] = None
        self._update_predictor(vehicle_name, vehicle_state, t)
        return

    @abstractmethod
    def _update_predictor(self, vehicle_name, vehicle_state, t):
        self.predictions[vehicle_name] = self.get_prediction(vehicle_state)
        return

    @abstractmethod
    def get_prediction(self, ego_state: VehicleState, target_state: VehicleState,
                       ego_prediction: VehiclePrediction, tar_prediction=None):
        return

    def _destroy_predictor(self, vehicle_name):
        self.startup_cycles_done.pop(vehicle_name)
        self.predictions.pop(vehicle_name)
        self.memory_dict.pop(vehicle_name)
        self.last_update.pop(vehicle_name)
        return


class ConstantVelocityPredictor(BasePredictor):
    '''
    extrapolates a line from current position using current linear velocity

    Does not implement state estimation or startup cycles - predicts directly off of passed state.
    '''

    def get_prediction(self, ego_state: VehicleState, target_state: VehicleState,
                       ego_prediction: VehiclePrediction, tar_prediction=None):
        target_state.update_global_velocity_from_body()
        t = target_state.t
        x = target_state.x.x
        y = target_state.x.y
        v_x = target_state.v_x
        v_y = target_state.v_y

        t_list = np.zeros((self.N))
        x_list = np.zeros((self.N))
        y_list = np.zeros((self.N))

        for i in range(self.N):
            x_list[i] = x
            y_list[i] = y
            t_list[i] = t
            t += self.dt
            x += self.dt * v_x
            y += self.dt * v_y

        pred = VehiclePrediction(t=target_state.t, x=x_list, y=y_list, psi=[target_state.e.psi]*self.N) # add same psi
        pred.xy_cov = np.repeat(np.diag([self.cov, self.cov])[np.newaxis, :, :], self.N, axis=0)
        return pred


class ConstantAngularVelocityPredictor(BasePredictor):
    '''
    extrapolates a curved line from current position using linear and angular velocity
    '''

    def get_prediction(self, ego_state: VehicleState, target_state: VehicleState,
                       ego_prediction: VehiclePrediction, tar_prediction=None):
        target_state.update_global_velocity_from_body()
        t = target_state.t
        x = target_state.x.x
        y = target_state.x.y
        v_x = target_state.v_x
        v_y = target_state.v_y
        psidot = target_state.w.w_psi
        psi = target_state.e.psi

        t_list = np.zeros((self.N))
        x_list = np.zeros((self.N))
        y_list = np.zeros((self.N))
        psi_list = np.zeros((self.N))

        delta_psi = 0
        for i in range(self.N):
            x_list[i] = x
            y_list[i] = y
            t_list[i] = t
            psi_list[i] = psi
            t += self.dt
            x += self.dt * (v_x * np.cos(delta_psi) - v_y * np.sin(delta_psi))
            y += self.dt * (v_y * np.cos(delta_psi) + v_x * np.sin(delta_psi))


            delta_psi += self.dt * psidot
            psi += self.dt * psidot

        pred = VehiclePrediction(t=target_state.t, x=x_list, y=y_list, psi=psi_list)
        pred.xy_cov = np.repeat(np.diag([self.cov, self.cov])[np.newaxis, :, :], self.N, axis=0)
        return pred

class NoPredictor(BasePredictor):
    """
    Actually predicts nothing.
    """
    def __init__(self, N:int):
        super(NoPredictor, self).__init__(N)

    def get_prediction(self, ego_state: VehicleState, target_state: VehicleState,
                       ego_prediction: VehiclePrediction, tar_prediction=None):
        return VehiclePrediction()

class MPCPredictor(BasePredictor):
    """
    Just passes through the target vehicle prediction.
    """
    def __init__(self, N:int):
        super(MPCPredictor, self).__init__(N)

    def get_prediction(self, ego_state: VehicleState, target_state: VehicleState,
                       ego_prediction: VehiclePrediction, tar_prediction=None):
        return tar_prediction



class GPPredictor(BasePredictor):
    def __init__(self, N: int, track : RadiusArclengthTrack, policy_name: str, use_GPU: bool, M: int, model=None, cov_factor: float = 1):
        super(GPPredictor, self).__init__(N, track)
        gc.collect()
        torch.cuda.empty_cache()
        # TODO: implement map from policy -> GP model
        self.gp = GPControllerTrained(policy_name, use_GPU, model)
        self.M = M  # number of samples
        self.cov_factor = cov_factor

    def get_prediction(self, ego_state: VehicleState, target_state: VehicleState,
                       ego_prediction: VehiclePrediction, tar_prediction=None):
        pred = self.gp.get_true_prediction_par(ego_state, target_state, ego_prediction, self.track, self.M)
        # fill in covariance transformation to x,y
        pred.track_cov_to_local(self.track, self.N, self.cov_factor)
        
        return pred
