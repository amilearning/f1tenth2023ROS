#!/usr/bin/env python3

from tqdm import tqdm
import seaborn as sns
from typing import List
import cProfile, pstats, io
from pstats import SortKey
import pandas as pd

import matplotlib.pyplot as plt

from barcgp.common.utils.file_utils import *
from barcgp.common.utils.scenario_utils import MultiPolicyEvalData, EvalData, PostprocessData

from barcgp.h2h_configs import *

total_runs = 200
track_width = width
blocking_threshold = 0.5  # Percentage of track x_tran movement to consider for blocking

policy_name = 'reverse'
policy_dir = os.path.join(eval_dir, policy_name)
scen_dir = os.path.join(policy_dir, 'track')

names = sorted(os.listdir(scen_dir), key=str)
print(names)
names_for_legend = names 
names_for_legend = ['NaiveGP', 'NMPC', 'CAV', 'Proposed']
colors = {"GP": "y", "NLMPC": "b", "CAV": "g", "CV": "m", "MPCC": "k", "STSP": "y", "TP": "r"}
find_color = lambda pred: [val for key, val in colors.items() if key in pred].__getitem__(0)


@dataclass
class Metrics:
    crash: bool = field(default=False)
    ego_crash: bool = field(default=False)
    tv_crash: bool = field(default=False)
    time_crash: float = field(default=0)
    ego_win: bool = field(default=False)
    left_track: bool = field(default=False)
    left_track_tv: bool = field(default=False)
    delta_s: float = field(default=0)
    time_overtake: float = field(default=-1)
    s_overtake: float = field(default=-1)
    xy_overtake: np.array = field(default=None)
    xy_crash: np.array = field(default=None)
    overtake_attempts_l: int = field(default=0)
    overtake_attempts_r: int = field(default=0)
    d_s_cont: np.array = field(default=None)
    col_d: np.array = field(default=None)
    init_ds: float = field(default=0)
    lateral_error: np.array = field(default=np.array([]))
    longitudinal_error: np.array = field(default=np.array([]))
    horizon_longitudinal_rmse: np.array = field(default=np.array([]))
    horizon_lateral_rmse: np.array = field(default=np.array([]))
    lateral_rmse: float = field(default=0.0)
    longitudinal_rmse: float = field(default=0.0)
    u_a: np.array = field(default=np.array([]))
    u_a_min: float = field(default=0)
    u_s: np.array = field(default=np.array([]))
    tv_inf: bool = field(default=None)
    ego_inf: bool = field(default=None)


@dataclass
class Vehicle:
    l: float = field(default=None)
    w: float = field(default=None)
    x: float = field(default=None)
    y: float = field(default=None)
    phi: float = field(default=None)


def collision_check(e1: Vehicle, e2: Vehicle):
    posx = e1.x
    posy = e1.y
    x_ob = e2.x
    y_ob = e2.y
    l_ob = e2.l
    w_ob = e2.w
    r = e1.w / 2
    dl = e1.l * 0.9 / 3
    s_e = np.sin(e1.phi)
    c_e = np.cos(e1.phi)
    s = np.sin(e2.phi)
    c = np.cos(e2.phi)
    dx1 = x_ob - (posx - 3 * dl * c_e / 2)
    dx2 = x_ob - (posx - dl * c_e / 2)
    dx3 = x_ob - (posx + dl * c_e / 2)
    dx4 = x_ob - (posx + 3 * dl * c_e / 2)
    dy1 = y_ob - (posy - 3 * dl * s_e / 2)
    dy2 = y_ob - (posy - dl * s_e / 2)
    dy3 = y_ob - (posy + dl * s_e / 2)
    dy4 = y_ob - (posy + 3 * dl * s_e / 2)
    a = (l_ob / np.sqrt(2) + r) * 1
    b = (w_ob / np.sqrt(2) + r) * 1

    i1 = (c * dx1 - s * dy1) ** 2 * 1 / a ** 2 + (s * dx1 + c * dy1) ** 2 * 1 / b ** 2
    i2 = (c * dx2 - s * dy2) ** 2 * 1 / a ** 2 + (s * dx2 + c * dy2) ** 2 * 1 / b ** 2
    i3 = (c * dx3 - s * dy3) ** 2 * 1 / a ** 2 + (s * dx3 + c * dy3) ** 2 * 1 / b ** 2
    i4 = (c * dx4 - s * dy4) ** 2 * 1 / a ** 2 + (s * dx4 + c * dy4) ** 2 * 1 / b ** 2
    return i1 > 1 and i2 > 1 and i3 > 1 and i4 > 1, min([i1, i2, i3, i4])


class CollisionChecker():
    def __init__(self, ego_win):
        self.t_col = 0
        self.col_d = []
        self.col = False
        self.s = []
        self.ego_win = ego_win
        self.collision_xy = None
        self.collision_y = 0
        self.tar_leading=True
        self.ego_leading_steps=0
        self.col_ego = False
        self.col_tv = False

    def next(self, e: VehicleState, t: VehicleState, e_p, t_p):
        if e.p.s > t.p.s:
            self.ego_leading_steps+=1
            if self.ego_leading_steps > 1:
                self.tar_leading=False
        e1 = Vehicle(l=ego_L, w=ego_W, x=e.x.x, y=e.x.y, phi=e.e.psi)
        e2 = Vehicle(l=tar_L, w=tar_W, x=t.x.x, y=t.x.y, phi=t.e.psi)
        res, d = collision_check(e1, e2)
        self.col_d.append(d)
        if self.ego_win:
            self.s.append(e.p.s)
        else:
            self.s.append(t.p.s)
        if not self.col and not res:
            self.t_col = e.t
            self.col = True
            if self.tar_leading:
                self.col_ego = True
            else:
                self.col_tv = True
            self.collision_xy = [e.x.x, e.x.y]

    def get_results(self):
        s = np.array(self.s)
        s = s / s[-1] * 100
        i = np.arange(100)
        interpolated = np.interp(i, s, self.col_d)
        return (self.col, self.t_col, interpolated, self.collision_xy, self.col_ego, self.col_tv)


class InterpolatedLead():
    def __init__(self, ego_win):
        self.ds = []
        self.s_p = []
        self.ds_percent = []
        self.ego_win = ego_win

    def next(self, e: VehicleState, t: VehicleState, e_p, t_p):
        self.ds.append(e.p.s - t.p.s)
        if self.ego_win:
            self.s_p.append(e.p.s)
        else:
            self.s_p.append(t.p.s)

    def get_results(self):
        s_p = np.array(self.s_p)
        s_p = s_p / s_p[-1] * 100
        i = np.arange(100)
        interpolated = np.interp(i, s_p, self.ds)
        return interpolated


class InterpolatedActuation():
    def __init__(self):
        self.u_a_min = np.inf
        self.u_a = []
        self.u_s = []
        self.s_p = []

    def next(self, e: VehicleState, t, e_p, t_p):
        if e.u.u_a < self.u_a_min:
            self.u_a_min = e.u.u_a
        self.u_a.append(e.u.u_a)
        self.u_s.append(e.u.u_steer)
        self.s_p.append(e.p.s)

    def get_results(self):
        s_p = np.array(self.s_p)
        s_p = s_p / s_p[-1] * 100
        i = np.arange(100)
        u_a = np.interp(i, s_p, self.u_a)
        u_s = np.interp(i, s_p, self.u_s)
        return [u_a, u_s, self.u_a_min]


class LeftTrack():
    def __init__(self, track_width):
        self.left = False
        self.left_tv = False
        self.track_wdth = track_width

    def next(self, e: VehicleState, t, e_p, t_p):
        if not self.left:
            if abs(e.p.x_tran) > self.track_wdth / 2 + ego_W:
                self.left = True
        if not self.left_tv:
            if abs(t.p.x_tran) > self.track_wdth / 2 + tar_W:
                self.left_tv = True

    def get_results(self):
        return self.left, self.left_tv


class Overtake():
    def __init__(self):
        self.tar_leading = True
        self.overtake_t = -1
        self.overtake_s = -1
        self.overtake_xy = [-1, -1]
        self.o_a_l = 0
        self.o_a_r = 0

    def next(self, e: VehicleState, t: VehicleState, e_p: VehiclePrediction, t_p: VehiclePrediction):
        if e.p.s > t.p.s and self.tar_leading:
            self.tar_leading = False
            self.overtake_t = e.t
            self.overtake_s = e.p.s
            self.overtake_xy = [e.x.x, e.x.y]
        elif e.p.s < t.p.s:
            if e_p is not None and e_p.x_tran is not None:
                if np.mean(e_p.x_tran) > t.p.x_tran:
                    self.o_a_r += 1
                else:
                    self.o_a_l += 1

    def get_results(self):
        all = max(1, self.o_a_l + self.o_a_r)
        o_a_l = self.o_a_l / all
        o_a_r = self.o_a_r / all
        return self.overtake_t, self.overtake_s, self.overtake_xy, o_a_l, o_a_r


def derive_lateral_long_error_from_true_traj(sim_data : EvalData, check_only_blocking=False):
    """
    @param sim_data: Input evaluation data where we are comparing against `tar_states` (true trajectory)
    @return:
    lateral_error (list of l_1 errors)
    longitudinal_error (list of l_1 errors)
    """
    lateral_error = []
    longitudinal_error = []
    track = sim_data.scenario_def.track
    samps = 0
    for timeStep in range(len(sim_data.tar_states)-1):
        pred = sim_data.tar_gp_pred[timeStep]  # (VehiclePrediction) at current timestep, what is GP prediction
        if pred is not None and (pred.x is not None or pred.s is not None):
            N = len(pred.s) if pred.s else len(pred.x)
            if N + timeStep - 1 < len(sim_data.tar_states):
                samps += 1
                for i in range(1, N):
                    tar_st = sim_data.tar_states[timeStep + i]  # (VehicleState) current target state from true traveled trajectory

                    if check_only_blocking:
                        xt = tar_st.p.x_tran
                        x_ref = np.sign(xt) * min(track.half_width, abs(float(xt)))

                        if pred.s is None and pred.x is not None:
                            f = track.global_to_local((pred.x[i], pred.y[i], pred.psi[i]))
                            if f is not None:
                                s, x_tran, e_psi = f
                            else:
                                return np.array([]), np.array([])
                            if s <= tar_st.p.s and np.fabs(x_ref) > blocking_threshold * track.track_width:
                                longitudinal = s - tar_st.p.s
                                lateral = x_tran - tar_st.p.x_tran
                                longitudinal_error.append(longitudinal)
                                lateral_error.append(lateral)
                        else:
                            if pred.s[i] <= tar_st.p.s and np.fabs(x_ref) > blocking_threshold * track.track_width:
                                longitudinal = pred.s[i] - tar_st.p.s
                                lateral = pred.x_tran[i] - tar_st.p.x_tran
                                longitudinal_error.append(longitudinal)
                                lateral_error.append(lateral)
                    else:
                        if not pred.s:
                            dx = tar_st.x.x - pred.x[i]
                            dy = tar_st.x.y - pred.y[i]
                            angle = sim_data.scenario_def.track.local_to_global((tar_st.p.s, 0, 0))[2]
                            longitudinal = dx * np.cos(angle) + dy * np.sin(angle)
                            lateral = -dx * np.sin(angle) + dy * np.cos(angle)
                        else:
                            longitudinal = pred.s[i] - tar_st.p.s
                            lateral = pred.x_tran[i] - tar_st.p.x_tran
                        longitudinal_error.append(longitudinal)
                        lateral_error.append(lateral)

    return np.array(lateral_error), np.array(longitudinal_error)


def multi_policy_lat_lon_error(sim_data : EvalData):
    """
    @param sim_data: Input evaluation data where we are comparing against `tar_states` (true trajectory)
    @return:
    lateral_error (list of l_1 errors)
    longitudinal_error (list of l_1 errors)
    """

    total_lateral_error = []
    total_longitunidal_error = []    
    track = sim_data.scenario_def.track
    samps = 0
    for timeStep in range(len(sim_data.tar_states)-1):
        lateral_error = []
        longitudinal_error = []
        pred = sim_data.tar_gp_pred[timeStep]  # (VehiclePrediction) at current timestep, what is GP prediction
        ego_states = sim_data.ego_states[timeStep]
        tar_states = sim_data.tar_states[timeStep] 
        data_skip = True
        # if (tar_states.p.s - ego_states.p.s) < 0.5 and  (tar_states.p.s - ego_states.p.s) > 0.0:
        if abs(tar_states.p.s - ego_states.p.s) < 1.0:
            data_skip = False
        if data_skip: 
            continue
        if pred is not None and (pred.x is not None or pred.s is not None):
            N = len(pred.s) if pred.s else len(pred.x)
            if N + timeStep - 1 < len(sim_data.tar_states):
                samps += 1
                for i in range(1, N):
                    tar_st = sim_data.tar_states[timeStep + i]  # (VehicleState) current target state from true traveled trajectory
                    if not pred.s:
                        dx = tar_st.x.x - pred.x[i]
                        dy = tar_st.x.y - pred.y[i]
                        angle = sim_data.scenario_def.track.local_to_global((tar_st.p.s, 0, 0))[2]
                        longitudinal = dx * np.cos(angle) + dy * np.sin(angle)
                        lateral = -dx * np.sin(angle) + dy * np.cos(angle)
                    else:
                        longitudinal = pred.s[i] - tar_st.p.s
                        lateral = pred.x_tran[i] - tar_st.p.x_tran
                    longitudinal_error.append(longitudinal)
                    lateral_error.append(lateral)
        
                total_longitunidal_error.append(np.array(longitudinal_error))
                total_lateral_error.append(np.array(lateral_error))
                
    return np.array(total_lateral_error), np.array(total_longitunidal_error)


def get_metrics(scen_data: EvalData):
    # TODO: This iterates over all times multiple times! Fix this
    metrics = Metrics()
    ego_win = scen_data.ego_states[-1].p.s > scen_data.tar_states[-1].p.s
    Col = CollisionChecker(ego_win)
    Lead = InterpolatedLead(ego_win)
    Act = InterpolatedActuation()
    L_Track = LeftTrack(scen_data.scenario_def.track.track_width)
    OT = Overtake()
    for (e, t, ep, tp) in zip(scen_data.ego_states, scen_data.tar_states, scen_data.ego_preds, scen_data.tar_preds):
        Col.next(e, t, ep, tp)
        Lead.next(e, t, ep, tp)
        L_Track.next(e, t, ep, tp)
        Act.next(e, t, ep, tp)
        OT.next(e, t, ep, tp)

    cr, t, inter, col_xy, metrics.ego_crash, metrics.tv_crash = Col.get_results()
    metrics.crash = cr
    if cr:
        metrics.crash_t = t
    metrics.delta_s = scen_data.ego_states[-1].p.s - scen_data.tar_states[-1].p.s
    o_t, o_s, o_xy, o_l, o_r = OT.get_results()
    metrics.time_overtake = o_t
    if hasattr(scen_data, 'tv_infeasible'):
        metrics.ego_inf, metrics.tv_inf = scen_data.ego_infeasible, scen_data.tv_infeasible
    while o_s > scen_data.scenario_def.track.track_length:
        o_s -= scen_data.scenario_def.track.track_length
    metrics.s_overtake = o_s
    metrics.xy_overtake = o_xy
    metrics.xy_crash = col_xy
    metrics.overtake_attempts_l = o_l
    metrics.overtake_attempts_r = o_r
    metrics.left_track, metrics.left_track_tv = L_Track.get_results()
    metrics.ego_win = ego_win and not cr and not metrics.left_track
    metrics.d_s_cont = Lead.get_results()
    metrics.col_d = inter
    metrics.init_ds = scen_data.ego_states[0].p.s - scen_data.tar_states[0].p.s
    # metrics.lateral_error, metrics.longitudinal_error = derive_lateral_long_error_from_true_traj(scen_data, check_only_blocking=False)
    metrics.lateral_error, metrics.longitudinal_error = multi_policy_lat_lon_error(scen_data)
    u = Act.get_results()
    metrics.lateral_rmse = np.sqrt(np.mean(metrics.lateral_error ** 2))
    
    # if sum(sum(np.isnan(metrics.lateral_error))) > 0:
    #     print(1)

    metrics.longitudinal_rmse = np.sqrt(np.mean(metrics.longitudinal_error ** 2))
    metrics.lateral_rmse = np.sqrt(np.mean(metrics.lateral_error ** 2))
    
    metrics.horizon_longitudinal_rmse = np.sqrt(np.mean((metrics.longitudinal_error ** 2),0))
    metrics.horizon_lateral_rmse = np.sqrt(np.mean((metrics.lateral_error ** 2),0))
    
    metrics.u_a = u[0]
    metrics.u_s = u[1]
    metrics.u_a_min = u[2]

    return metrics

def parse_metrics(metrics: Metrics, data: PostprocessData, i):
    # Counts
    data.num_wins += metrics.ego_win
    data.num_overtakes += metrics.ego_win  # TODO Replace this
    data.num_ego_inf += metrics.ego_inf
    data.num_tv_inf += metrics.tv_inf
    data.num_crashes += metrics.crash
    data.num_left_track += metrics.left_track
    # Averages
    data.avg_delta_s += metrics.delta_s/data.N
    data.avg_a += np.average(metrics.u_a)/data.N
    data.avg_min_a += metrics.u_a_min/data.N
    data.avg_abs_steer += np.average(np.abs(metrics.u_s))/data.N
    if metrics.ego_win:
        data.overtake_s.append(metrics.s_overtake)
        xy = metrics.xy_overtake
        data.overtake_x.append(xy[0])
        data.overtake_y.append(xy[1])
        data.win_ids.append(str(i) + '.pkl')
        data.overtake_ids.append(str(i) + '.pkl')
    if metrics.ego_inf:
        data.ego_infesible_ids.append(str(i) + '.pkl')
    if metrics.tv_inf:
        data.tv_infesible_ids.append(str(i) + '.pkl')
    if metrics.crash:
        data.crash_ids.append(str(i) + '.pkl')
        data.crash_x.append(metrics.xy_crash[0])
        data.crash_y.append(metrics.xy_crash[1])
        if metrics.ego_crash:
            data.crash_ids_ego.append(str(i) + '.pkl')
        else:
            data.crash_ids_tv.append(str(i) + '.pkl')
    if metrics.left_track:
        data.left_track_ids.append(str(i) + '.pkl')
    if not metrics.left_track_tv:
        data.lateral_errors.extend(metrics.lateral_error)
        data.longitudinal_errors.extend(metrics.longitudinal_error)        
        data.horizon_lateral_rmse.append(metrics.horizon_lateral_rmse)
        data.horizon_longitudinal_rmse.append(metrics.horizon_longitudinal_rmse)


def get_lat_lon_mse(processed_data, tmp_policy_name):        
    last_step_lon_mse = []        
    last_step_lat_mse = [] 
    for id, ctrl_name in enumerate(names):
        if ctrl_name == 'CAV_01' or ctrl_name == 'GP_2' or ctrl_name == 'NLMPC_01' or ctrl_name == 'TP_2 ':
            ctrl_data = processed_data[ctrl_name]                
            
            last_step_lon_data = np.array(ctrl_data.longitudinal_errors)[:,-1]            
            lon_mse_tmp = np.mean(np.square(last_step_lon_data))
            last_step_lon_mse.append(lon_mse_tmp)

            last_step_lat_data = np.array(ctrl_data.lateral_errors)[:,-1]
            lat_mse_tmp = np.mean(np.square(last_step_lat_data))
            last_step_lat_mse.append(lat_mse_tmp)
    
    
    last_step_lon_mse_np = np.transpose(np.array(last_step_lon_mse)).reshape(1, -1)   
   
 
    last_step_lon_mse_np_df = pd.DataFrame(last_step_lon_mse_np,columns=['CAV_01', 'GP_2', 'NLMPC_01', 'TP_2'])
    last_step_lon_mse_np_df['Policy'] = str(tmp_policy_name)

    last_step_lat_mse_np = np.transpose(np.array(last_step_lat_mse)).reshape(1, -1)    
    last_step_lat_mse_np_df = pd.DataFrame(last_step_lat_mse_np,columns=['CAV_01', 'GP_2', 'NLMPC_01', 'TP_2'])
    last_step_lat_mse_np_df['Policy'] = str(tmp_policy_name)

    return    last_step_lon_mse_np_df, last_step_lat_mse_np_df
            
def get_step_df(step_erros, min_idx, tmp_policy_name):
    a = []
    for i in range(len(step_erros)):
        tmp =  step_erros[i][:min_idx]
        a.append(tmp)
    a_np = np.transpose(np.array(a))    
    df_step = pd.DataFrame(a_np,columns=['CAV_01', 'GP_2', 'NLMPC_01', 'TP_2'])
    df_step['Policy'] = str(tmp_policy_name)
    return df_step

def get_lon_lat_df(processed_data, tmp_policy_name):
    first_step_erros_lon = []
    last_step_erros_lon = []
    
    first_step_erros_lat = []
    last_step_erros_lat = [] 
    for id, ctrl_name in enumerate(names):
        if ctrl_name == 'CAV_01' or ctrl_name == 'GP_2' or ctrl_name == 'NLMPC_01' or ctrl_name == 'TP_2 ':
            ctrl_data = processed_data[ctrl_name]                
            first_step_erros_lon.append(np.array(ctrl_data.longitudinal_errors)[:,2])       
            last_step_erros_lon.append(np.array(ctrl_data.longitudinal_errors)[:,-1])
            
            first_step_erros_lat.append(np.array(ctrl_data.lateral_errors)[:,2])       
            last_step_erros_lat.append(np.array(ctrl_data.lateral_errors)[:,-1])          
    
    
    # Initialize the minimum length as the maximum possible value
 
    # Iterate through the list and update the minimum length if a shorter length is found
    def get_min_idx(step_erros):
        idx = float('inf')
        for arr in step_erros:
            current_length = len(arr)
            if current_length < idx:
                idx = current_length
        return idx 

    first_step_erros_lon_min_length = get_min_idx(first_step_erros_lon)
    last_step_erros_lon_min_length = get_min_idx(last_step_erros_lon)
    first_step_erros_lat_min_length = get_min_idx(first_step_erros_lat)
    last_step_erros_lat_min_length = get_min_idx(last_step_erros_lat)


    
    first_step_erros_lon_df = get_step_df(first_step_erros_lon,first_step_erros_lon_min_length,tmp_policy_name)
    last_step_erros_lon_df = get_step_df(last_step_erros_lon,last_step_erros_lon_min_length,tmp_policy_name)
    first_step_erros_lat_df = get_step_df(first_step_erros_lat,first_step_erros_lat_min_length,tmp_policy_name)
    last_step_erros_lat_df = get_step_df(last_step_erros_lat,last_step_erros_lat_min_length,tmp_policy_name)

    return first_step_erros_lon_df, last_step_erros_lon_df, first_step_erros_lat_df, last_step_erros_lat_df
        
def main(args=None):
    

  
    def get_process(policy_name):
        
        policy_dir = os.path.join(eval_dir, policy_name)
        scen_dir = os.path.join(policy_dir, 'track')
        print(f"Evaluating data for policy: {policy_name} in {scen_dir}")
        n = len(names)
        # Initialize processed data container
        processed_data = {}
        for i in names:
            processed_data[i] = PostprocessData()
            processed_data[i].name = i
            processed_data[i].setup_id = policy_name
            processed_data[i].N = 100

        # Temporary variables
        counter = [0] * n
        counter_nc = [0] * n
        vals = [0] * n
        wins = [0] * n
        win_ids = [None] * n
        tv_inf = [0] * n
        ego_inf = [0] * n
        fair_counter = 0
        fair_vals = [0] * n
        crashes = [0] * n
        o_l = [0] * n
        o_r = [0] * n
        o_s = []
        o_xy = []
        for i in range(n):
            o_s.append([])
            o_xy.append([])
        l_t = [0] * n
        d_s_cont = [np.zeros((100,))]
        col_d_cont = [np.zeros((100,))]
        u_a = [np.zeros((100,))]
        u_s = [np.zeros((100,))]
        lat_rmse = [0] * n
        long_rmse = [0] * n
        lateral_errors, longitudinal_errors = dict(), dict()
        for i in range(n - 1):
            d_s_cont.append(np.zeros((100,)))
            col_d_cont.append(np.zeros((100,)))
            u_a.append(np.zeros((100,)))
            u_s.append(np.zeros((100,)))
    
        for i in tqdm(range(total_runs)):
            scores = [None] * n
            exists = True
            for a in names:
                name = os.path.join(os.path.join(scen_dir, a), str(i) + '.pkl')
                if not os.path.exists(name):
                    exists = False
                    break
            if exists:
                for id, a in enumerate(names):
                    if id not in lateral_errors:
                        lateral_errors[id] = []
                    if id not in longitudinal_errors:
                        longitudinal_errors[id] = []

                    name = os.path.join(os.path.join(scen_dir, a), str(i) + '.pkl')
                    if os.path.exists(name):
                        dbfile = open(name, 'rb')
                        multi_scenario_data: MultiPolicyEvalData = pickle.load(dbfile)
                        scenario_data = multi_scenario_data.evaldata
                        
                        if processed_data[a].track is None:
                            processed_data[a].track = scenario_data.scenario_def.track
                        metrics = get_metrics(scenario_data)
                        if metrics.lateral_error.shape[0] != 0:
                            parse_metrics(metrics, processed_data[a], i)
                        
                        
    
        post_path = os.path.join(policy_dir, policy_name + '.pkl')
        pickle_write(processed_data, post_path)

    
        
    
    
    # policy_names = ['timid', 'mild_200', 'aggressive_blocking',  'mild_5000' ,'reverse']
    policy_names = ['mild_200', 'aggressive_blocking', 'mild_5000', 'reverse']
    policy_names = ['mild_200', 'aggressive_blocking', 'mild_5000']
    # policy_names = [ 'mild_200', 'aggressive_blocking',  'mild_5000' ,'reverse']
    
    for j in range(len(policy_names)):        
        get_process(policy_names[j])

    

    
    
    first_step_erros_lon_df_set= []
    last_step_erros_lon_df_set = []
    first_step_erros_lat_df_set= []
    last_step_erros_lat_df_set= []

    lon_mse_df_set = []
    lat_mse_df_set = []
    for tmp_policy_name in policy_names:
        policy_dir = os.path.join(eval_dir, tmp_policy_name)
        tmp_post_path = os.path.join(policy_dir, tmp_policy_name + '.pkl')
        tmp_processed_data = pickle_read(tmp_post_path)                
        first_step_erros_lon_df, last_step_erros_lon_df, first_step_erros_lat_df, last_step_erros_lat_df = get_lon_lat_df(tmp_processed_data, tmp_policy_name)
        last_step_lon_mse_np_df, last_step_lat_mse_np_df = get_lat_lon_mse(tmp_processed_data, tmp_policy_name)
        lon_mse_df_set.append(last_step_lon_mse_np_df)
        lat_mse_df_set.append(last_step_lat_mse_np_df)
        first_step_erros_lon_df_set.append(first_step_erros_lon_df)
        last_step_erros_lon_df_set.append(last_step_erros_lon_df)
        first_step_erros_lat_df_set.append(first_step_erros_lat_df)
        last_step_erros_lat_df_set.append(last_step_erros_lat_df)
            
    def plot_step_error(df_set,value_name_):
        cat_df = pd.concat(df_set)
        cat_df_melted = pd.melt(cat_df, id_vars='Policy', var_name='Policy_Name', value_name=value_name_)
        error_plt= sns.catplot(x='Policy', y=value_name_, hue='Policy_Name', kind='box', data=cat_df_melted,showfliers = False)
        
        plt.xlabel('Policy')
        plt.ylabel(value_name_)
        plt.show()
        # plt.suptitle('Lateral Error', y=1.05)
        # plt.tight_layout()
        return error_plt

    def plot_mse_error(df_set,value_name_):
        cat_df = pd.concat(df_set)
        cat_df_melted = pd.melt(cat_df, id_vars='Policy', var_name='Policy_Name', value_name=value_name_)
        # error_plt= sns.catplot(x='Policy', y=value_name_, hue='Policy_Name', kind='box', data=cat_df_melted,showfliers = False)
        error_plt= sns.barplot(x='Policy', y=value_name_, hue='Policy_Name',  data=cat_df_melted)
        plt.xlabel('Policy')
        plt.ylabel(value_name_)
        plt.show()
        # plt.suptitle('Lateral Error', y=1.05)
        # plt.tight_layout()
        return error_plt
        
    # plot_step_error(first_step_erros_lon_df_set, value_name_ = 'Longitudinal_Error')
    # plot_step_error(first_step_erros_lat_df_set,value_name_ = 'Lateral_Error')

    plot_step_error(last_step_erros_lon_df_set,value_name_ = 'Longitudinal_Error')
    plot_step_error(last_step_erros_lat_df_set,value_name_ = 'Lateral_Error')
    
    plot_mse_error(lon_mse_df_set,value_name_ = 'Longitudinal_MSE')
    
    
    plot_mse_error(lat_mse_df_set,value_name_ = 'Lateral_MSE')
    plt.show()





 
if __name__ == '__main__':
    main()