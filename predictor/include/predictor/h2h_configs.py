from predictor.common.pytypes import *
from predictor.controllers.utils.controllerTypes import *
from predictor.dynamics.models.model_types import DynamicBicycleConfig
from enum import Enum
import math

class Predictor(Enum):
    GroundTruth = 0
    LSTM = 1
    NMPC = 2
    DirectGP = 3
    AutoGP = 4 
    ConstantInput = 5           

class Controllers(Enum):
    NMPC = 0
    MPPI = 1
    GPNMPC = 2




# Time discretization
dt = 0.1
# Horizon length
N = 10
# Number of iterations to run PID (need N+1 because of NLMPC predictor warmstart)
n_iter = N+1 
# Track width (should be pre-determined from track generation '.npz')
width = 2.2

# Force rebuild all FORCES code-gen controllers
rebuild = True
# Use a key-point lookahead strategy that is dynamic (all_tracks=True) or pre-generated (all_tracks=False)
all_tracks = True
offset = 32 if not all_tracks else 0

ego_L = 0.33
ego_W = 0.173

tar_L = 0.33
tar_W = 0.173

# Initial track conditions
factor = 1.3  # v_long factor
tarMin = VehicleState(t=0.0,
                      p=ParametricPose(s=offset + 2.0, x_tran=-.3 * width, e_psi=-0.02),
                      v=BodyLinearVelocity(v_long=0.8*factor))
tarMax = VehicleState(t=0.0,
                      p=ParametricPose(s=offset + 2.2, x_tran=.3* width, e_psi=0.02),
                      v=BodyLinearVelocity(v_long=1.0*factor))
egoMin = VehicleState(t=0.0,
                      p=ParametricPose(s=offset + 0.2, x_tran=-.3 * width, e_psi=-0.02),
                      v=BodyLinearVelocity(v_long=0.5*factor))
egoMax = VehicleState(t=0.0,
                      p=ParametricPose(s=offset + 0.4, x_tran=.3 * width, e_psi=0.02),
                      v=BodyLinearVelocity(v_long=1.0*factor))

IKD_width = 1.0
IKD_egoMin = VehicleState(t=0.0, p=ParametricPose(s=offset +0.2, x_tran=-0.8* IKD_width, e_psi=-math.pi/3), v=BodyLinearVelocity(v_long=0.1))
IKD_egoMax = VehicleState(t=0.0, p=ParametricPose(s=offset + 0.3, x_tran=0.8* IKD_width, e_psi=math.pi/3), v=BodyLinearVelocity(v_long=2.5))



tar_dynamics_config = DynamicBicycleConfig(dt=dt, model_name='dynamic_bicycle_full',
                                           wheel_dist_front=0.165, wheel_dist_rear=0.165, slip_coefficient=.9)
ego_dynamics_config = DynamicBicycleConfig(dt=dt, model_name='dynamic_bicycle_full',
                                           wheel_dist_front=0.165, wheel_dist_rear=0.165, slip_coefficient=.9)

# Controller parameters
gp_mpcc_ego_params = MPCCApproxFullModelParams(
    dt=dt,
    all_tracks=all_tracks,
    solver_dir='' if rebuild else '~/.mpclab_controllers/gp_mpcc_h2h_ego',
    # solver_dir='',
    optlevel=2,


    N=N,
    Qc=50.0, # e_cont , countouring error 
    Ql=500.0, #500.0  # e_lag, lag error 
    Q_theta= 200, # progress speed  v_proj_prev 


    Q_xref=0.0, #  reference tracking for blocking 
    R_d=2.0, # u_a, u_a_dot 
    R_delta=20.0, # 20.0 # u_delta, u_delta_dot

    slack=True,
    l_cs=5, # obstacle_slack
    Q_cs=2.0, # # obstacle_slack_e
    Q_vmax=200.0,
    vlong_max_soft=2.4, ## reference speed .. only activate if speed exceeds it 
    Q_ts=500.0, # track boundary
    Q_cs_e=8.0, # obstacle slack
    l_cs_e=35.0,  # obstacle slack

    num_std_deviations= 0.01,

    u_a_max=1.0,
    vx_max=2.6,
    u_a_min=-1.,
    u_steer_max=0.435,
    u_steer_min=-0.435,
    u_a_rate_max=10,
    u_a_rate_min=-10,
    u_steer_rate_max=2,
    u_steer_rate_min=-2
)

mpcc_ego_params = MPCCApproxFullModelParams(
    dt=dt,
    all_tracks=all_tracks,
    solver_dir='' if rebuild else '~/.mpclab_controllers/mpcc_h2h_ego',
    # solver_dir='',
    optlevel=2,

    N=N,
    Qc=50,
    Ql=500.0,
    Q_theta=200.0,
    Q_xref=0.0,
    R_d=2.0,
    R_delta=20.0,

    slack=True,
    l_cs=5,
    Q_cs=2.0,
    Q_vmax=200.0,
    vlong_max_soft=1.4,
    Q_ts=500.0,
    Q_cs_e=8.0,
    l_cs_e=35.0,

    u_a_max=0.55,
    vx_max=1.6,
    u_a_min=-1,
    u_steer_max=0.435,
    u_steer_min=-0.435,
    u_a_rate_max=10,
    u_a_rate_min=-10,
    u_steer_rate_max=2,
    u_steer_rate_min=-2
)

mpcc_tv_params = MPCCApproxFullModelParams(
    dt=dt,
    all_tracks=all_tracks,
    solver_dir='' if rebuild else '~/.mpclab_controllers/mpcc_h2h_tv',
    # solver_dir='',
    optlevel=2,

    N=N,
    Qc=50,
    Ql=500.0,
    Q_theta=200.0,
    Q_xref=500.0,
    R_d=5.0,
    R_delta=25.0,

    slack=True,
    l_cs=5,
    Q_cs=2.0,
    Q_vmax=200.0,
    vlong_max_soft=1.0,
    Q_ts=500.0,
    Q_cs_e=8.0,
    l_cs_e=35.0,

    u_a_max=0.45,
    vx_max=1.3,
    u_a_min=-1,
    u_steer_max=0.435,
    u_steer_min=-0.435,
    u_a_rate_max=10,
    u_a_rate_min=-10,
    u_steer_rate_max=2,
    u_steer_rate_min=-2
)


# For NLMPC predictor
nl_mpc_params = NLMPCParams(
        dt=dt,
        solver_dir='' if rebuild else '~/.mpclab_controllers/NL_MPC_solver_forces_pro',
        # solver_dir='',
        optlevel=2,
        slack=False,

        N=N,
        Q=[10.0, 0.2, 1, 15, 0.0, 25.0], # .5 10
        R=[0.1, 0.1],
        Q_f=[10.0, 0.2, 1, 17.0, 0.0, 1.0], # .5 10
        R_d=[5.0, 5.0],
        Q_s=0.0,
        l_s=50.0,

        x_tran_max=width/2,
        x_tran_min=-width/2,
        u_steer_max=0.3,
        u_steer_min=-0.3,
        u_a_max=1.0,
        u_a_min=-2.2,
        u_steer_rate_max=2,
        u_steer_rate_min=-2,
        u_a_rate_max=1.0,
        u_a_rate_min=-1.0
    )