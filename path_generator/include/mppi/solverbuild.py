#!/usr/bin python3

import numpy as np
import scipy.linalg as la

import shutil
import casadi as ca
import pathlib
import array
from typing import Tuple

from mppi.model_types import DynamicsConfig, DynamicBicycleConfig
from mppi.controllerTypes import MPCCApproxFullModelParams
from mppi.track import Track
from mppi.pytypes import VehicleState
import sys, os

sys.path.append(os.path.join(os.path.expanduser('~'), 'forces_pro_client'))
import forcespro
import forcespro.nlp


rebuild = True
mpcc_tv_params = MPCCApproxFullModelParams(
    dt=0.1,
    all_tracks=True,   
    solver_dir='' if rebuild else '~/.mpclab_controllers/',
    # solver_dir='',
    optlevel=2,

    N=10,
    

    slack=True,
   



)
def dir_exists(path=''):
    dest_path = pathlib.Path(path).expanduser()
    return dest_path.exists()



class SolverBuilder():
    def __init__(self, model_config: DynamicBicycleConfig, track=Track, control_params=MPCCApproxFullModelParams()):   
        self.x_ws = None
        self.u_ws = None
        self.initialized = False
        self.solver_name = 'FREN_KIN'
        DynamicBicycleConfig.__post_init__
        self.model_config = model_config
        self.track = track
        self.N = 10
        self.l_r = DynamicBicycleConfig.wheel_dist_center_rear
        self.l_f = DynamicBicycleConfig.wheel_dist_center_front
        self.mass = DynamicBicycleConfig.mass
        self.Izz = DynamicBicycleConfig.yaw_inertia
        self.g = DynamicBicycleConfig.gravity
        self.mu = DynamicBicycleConfig.wheel_friction
        self.Pb = DynamicBicycleConfig.pacejka_b
        self.Pc = DynamicBicycleConfig.pacejka_c
        self.Pdf = DynamicBicycleConfig.pacejka_d_front
        self.Pdr = DynamicBicycleConfig.pacejka_d_rear
        self.min_accel = -5.0
        self.max_accel = 5.0
        self.min_delta = -0.43 
        self.max_delta = 0.43
        self.slack = True        
        self.zvars = ['s', 'ey', 'epsi', 'vel','eyslack', 'accel', 'delta']
        self.optlevel = control_params.optlevel

        self.solver_dir = control_params.solver_dir
        self.n = 4  
        self.nslack = 1   
        self.nu = 2  
        self.vmax = 3.0
        self.Qprogress = 0.5
        self.Qey = 1.0
        self.Qepsi = 0.3
        self.R_accel = 0.01
        self.R_delta = 0.01
        self.Qeyslack = 10.0
        self.Qvmax = 1.0
        self.Qtrackcross = 1.0


        self.pvars = ['vmax', 'Qprogress', 'Qey', 'Qepsi', 'R_accel', 'R_delta', 'Qeyslack','Qvmax', 'Qtrackcross', 
                        'kp1_0', 'kp1_1',
                        'kp2_0', 'kp2_1',
                        'kp3_0', 'kp3_1',
                        'kp4_0', 'kp4_1']

                        # 'kw1_0', 'kw1_1',
                        # 'kw2_0', 'kw2_1',
                        # 'kw3_0', 'kw3_1',
                        # 'kw4_0', 'kw4_1']
        self.get_curvature_dyn = self.track.get_curvature_casadi_fn_dynamic()
        self.get_trackwidth_dyn = self.track.get_track_width_casadi_fn_dynamic()

    def dynamics_continous(self,z,p):
        # state x = [s, ey, epsi, vx, vy, wz]
        return 
    
    def _load_solver(self, solver_dir):
        solver_found = True
        self.solver = forcespro.nlp.Solver.from_directory(solver_dir)


    def kinemaic_continuous_dynamics(self,z,p,u):
        """Defines dynamics of the car, i.e. equality constraints.
        parameters:        
        state x = [s,ey,epsi,vel,slack(ey)]
        input u = [accel, delta]        
        """     
        accel = u[self.zvars.index('accel')-z.size()[0]-1]
        delta = u[self.zvars.index('delta')-z.size()[0]-1]
        s = z[self.zvars.index('s')]
        ey = z[self.zvars.index('ey')]
        epsi = z[self.zvars.index('epsi')]
        vel = z[self.zvars.index('vel')]

        kp1 = p[self.pvars.index('kp1_0'):self.pvars.index('kp1_0') + 2]
        kp2 = p[self.pvars.index('kp2_0'):self.pvars.index('kp2_0') + 2]
        kp3 = p[self.pvars.index('kp3_0'):self.pvars.index('kp3_0') + 2]
        kp4 = p[self.pvars.index('kp4_0'):self.pvars.index('kp4_0') + 2]
        curv = self.get_curvature_dyn(s,ca.transpose(ca.horzcat(kp1, kp2, kp3, kp4)))
        
        beta = ca.atan(self.l_r / (self.l_f + self.l_r) * ca.tan(delta))
        s_dynamics = vel * ca.cos(epsi+beta) / (1 - ey * curv)  
        ey_dynamics = vel * ca.sin(epsi + beta)
        
        dyawdt = vel / self.l_r * ca.sin(beta)
        dsdt = vel * ca.cos(epsi+beta) / (1 - ey * curv )   
        epsi_dynamics = dyawdt - dsdt * curv
        vel_dynamics = accel
        h = ca.vertcat(s_dynamics, ey_dynamics, epsi_dynamics, vel_dynamics)

        return h
        
    def obj(self,z,p):
        Qs = p[self.pvars.index('Qprogress')]
        Qey = p[self.pvars.index('Qey')]
        Qepsi = p[self.pvars.index('Qepsi')]
        R_accel = p[self.pvars.index('R_accel')]
        R_delta = p[self.pvars.index('R_delta')]
        Qtrackcross = p[self.pvars.index('Qtrackcross')]
        Qeyslack = p[self.pvars.index('Qeyslack')]        
        Qvmax = p[self.pvars.index('Qvmax')]
        
        vmax = p[self.pvars.index('vmax')]
        
        s = z[self.zvars.index('s')]
        ey = z[self.zvars.index('ey')]
        epsi = z[self.zvars.index('epsi')]
        vel = z[self.zvars.index('vel')]  # todo: replace
        
        eyslack = z[self.zvars.index('eyslack')]  # todo: replace
        accel = z[self.zvars.index('accel')]
        delta = z[self.zvars.index('delta')]


    
        track_width = 0.5 #self.get_trackwidth(s)

        cost = ca.bilin(Qey, ey, ey) + ca.bilin(Qepsi, epsi, epsi) + \
                ca.bilin(R_accel, accel, accel) + ca.bilin(R_delta, delta, delta) - Qs * s

        if self.slack:
        ## punish only if vel is greater than vmax
            cost += ca.fmax(vel - vmax, 0) ** 2 * Qvmax
            # Track boundary constraints
            cost += ca.fmax(ca.fabs(ey) - track_width, 0) ** 2 * Qtrackcross
            # Track boundary slack
            cost += ca.power(eyslack, 2) * Qeyslack
            
        
        return cost 
    

    def inequality_const(self,z):
        s = z[self.zvars.index('s')]
        width =0.5 # self.get_trackwidth(s)
        ey = z[self.zvars.index('ey')]
         
        eyslack = z[self.zvars.index('eyslack')]  # todo: replace
        # ey - track -slcak <= 0
        lw = ey-width-eyslack 
        uw = -ey-width-eyslack
        # ey + track -slcak <= 0 
        return ca.vertcat(lw,uw)
        

    def build_solver(self):
        # Forces model
        self.model = forcespro.nlp.SymbolicModel(self.N)

        self.model.N = 10  # horizon length

              
        self.model.nvar = self.n+self.nslack + self.nu  # number of variables [s,ey,epsi,vel,epsiSlack, accel, delta]
        ## dynamics constraint 
        self.model.neq  = 4  # number of equality constraints
        ## ey < track_width + track_slack 
        self.model.nh   = 2  # number of inequality constraint functions
      
        
        ## -3 < accel < 3 
        ## -0.43 < delta < 0.43 
        
        self.model.npar = len(self.pvars)  # number of runtime parameters

        self.model.objective = lambda z, p: self.obj(z, p)
        
        integrator_stepsize = 0.1
        self.model.eq = lambda z,p: forcespro.nlp.integrate(self.kinemaic_continuous_dynamics, z[:4],p,z[5:],
                                                integrator=forcespro.nlp.integrators.RK4,
                                                stepsize=integrator_stepsize)
        
        self.model.E = np.hstack([np.eye(self.n), np.zeros([ self.n,self.nu+self.nslack])])
        

        # Inequality constraint bounds
        self.model.ineq = lambda z: self.inequality_const(z)
        self.model.hu = np.array([0.0, 0.0])  # upper bound for nonlinear constraints, pg. 23 of FP manual
        self.model.hl = np.array([-ca.inf, -ca.inf])# upper bound for nonlinear constraints, pg. 23 of FP manual

        # s ey epsi, vel, eyslack(>0), accel, delta 
        self.model.lb = np.array([-np.inf, -np.inf, -np.inf, -np.inf, 0.0, self.min_accel, self.min_delta])
        self.model.ub = np.array([np.inf, np.inf, np.inf, np.inf, 10.0, self.max_accel, self.max_delta])

   
        # Put initial condition on all state variables x
        self.model.xinitidx = range(self.n+self.nslack)
      
        # Set solver options

        self.options = forcespro.CodeOptions(self.solver_name)
        self.options.maxit = 400
        self.options.overwrite = True
        self.options.printlevel = 0
        self.options.optlevel = self.optlevel
        self.options.BuildSimulinkBlock = False
        self.options.cleanup = True
        self.options.platform = 'Generic'
        self.options.gnu = True
        self.options.sse = True
        self.options.noVariableElimination = False
        # self.options.parallel = 1

        self.options.nlp.linear_solver = 'symm_indefinite'
        self.options.nlp.stack_parambounds = True
        self.options.nlp.TolStat = 1e-4
        self.options.nlp.TolEq = 1e-4
        self.options.nlp.TolIneq = 1e-4
        self.options.nlp.TolComp = 1e-4
        self.options.accuracy.ineq = 1e-4
        self.options.accuracy.eq = 1e-4
        self.options.accuracy.mu = 1e-4
        self.options.accuracy.rdgap = 1e-4
        self.options.solvemethod = 'PDIP_NLP'

        # Creates code for symbolic model formulation given above, then contacts server to generate new solver
        self.model.generate_solver(self.options)
        self.install_dir = self.install()  # install the model to ~/.mpclab_controllers
        self.solver = forcespro.nlp.Solver.from_directory(self.install_dir)
        # Copy solver config.
        # pickle_write([self.control_params, self.track_name], os.path.join(self.install_dir, 'params.pkl')) 

            # Method for installing generated solver files

    def initialize(self):
        # if self.solver_dir:
        #     self.solver_dir = pathlib.Path(self.solver_dir).expanduser()  # allow the use of ~
        #     self._load_solver(self.solver_dir)
        # else:
        rebuild =False 
        if rebuild:
            self.build_solver()
        else:
            self.install_dir = self.install()  # install the model to ~/.mpclab_controllers
            if self.install_dir is None:
                self.build_solver()
            else:
                self.solver = forcespro.nlp.Solver.from_directory(self.install_dir)
        self.initialized = True


    def install(self, path='~/mpclab_controllers/', verbose=False):
        src_path = pathlib.Path.cwd().joinpath(self.solver_name)
        # src_path = pathlib.Path(path).joinpath(self.solver_name)
        if src_path.exists():
            dest_path = pathlib.Path(path).expanduser()
            if not dest_path.exists():
                dest_path.mkdir(parents=True)
            if dest_path.joinpath(self.solver_name).exists():
                if verbose:
                    print('- Existing installation found, removing...')
                shutil.rmtree(dest_path.joinpath(self.solver_name))
            shutil.move(str(src_path), str(dest_path))
            if verbose:
                print('- Installed files from source: %s to destination: %s' % (str(src_path), str(dest_path.joinpath(self.solver_name))))
            return dest_path.joinpath(self.solver_name)
        else:
            if verbose:
                print('- The source directory %s does not exist, did not install' % str(src_path))
            return None

    def state2qu(self,state=VehicleState):
        ## s, ey, epsi, vel, slack, ua, udelta
        return np.array([state.p.s, state.p.x_tran, state.p.e_psi, state.v.v_long, 0.0])
             


    
    def set_warm_start(self, x_ws, u_ws):
        if x_ws.shape[0] != self.N or x_ws.shape[1] != self.n:  # TODO: self.N+1
            raise (RuntimeError(
                'Warm start state sequence of shape (%i,%i) is incompatible with required shape (%i,%i)' % (
                    x_ws.shape[0], x_ws.shape[1], self.N, self.n)))
        if u_ws.shape[0] != self.N or u_ws.shape[1] != self.nu:
            raise (RuntimeError(
                'Warm start input sequence of shape (%i,%i) is incompatible with required shape (%i,%i)' % (
                    u_ws.shape[0], u_ws.shape[1], self.N, self.nu)))

        self.x_ws = x_ws
        self.u_ws = u_ws

    def solve(self, state = VehicleState, key_pts = None):
        if not self.initialized:
            raise (RuntimeError(
                'MPCC controller is not initialized, run MPCC.initialize() before calling MPCC.solve()'))

        x = self.state2qu(state)
        
        if self.x_ws is None:
            print('Initial guess of open loop state sequence not provided, using zeros')
            self.x_ws = np.zeros((self.N, self.n))
        if self.u_ws is None:
            print('Initial guess of open loop input sequence not provided, using zeros')
            self.u_ws = np.zeros((self.N, self.nu))

        # Set up real-time parameters, warm-start
        parameters = []
        initial_guess = []
     
        current_s = state.p.s
        while current_s < 0: current_s += self.track.track_length
        while current_s >= self.track.track_length: current_s -= self.track.track_length
        
        if key_pts is None:
            key_pts = np.array([[current_s-1, 0.0],
                                [current_s+2, 0.0],
                                [current_s+4, 0.0],
                                [current_s+8, 0.0]])
         
        for stageidx in range(self.N):
            initial_guess.append(self.x_ws[stageidx])  # x
            initial_guess.append(np.zeros((1,))) ## for slack 
            initial_guess.append(self.u_ws[stageidx])  # u
            stage_p = []
            stage_p.extend([self.vmax, self.Qprogress, self.Qey, self.Qepsi, self.R_accel, self.R_delta, self.Qeyslack, self.Qvmax, self.Qtrackcross])
            stage_p.extend(key_pts[0])
            stage_p.extend(key_pts[1])
            stage_p.extend(key_pts[2])
            stage_p.extend(key_pts[3])

            parameters.append(stage_p)

         
        parameters = np.concatenate(parameters)
        initial_guess = np.concatenate(initial_guess)

        # problem dictionary, arrays have to be flattened
        problem = dict()
        problem["xinit"] = x
        problem["all_parameters"] = parameters
        problem["x0"] = initial_guess

        output, exitflag, solve_info = self.solver.solve(problem)

        if exitflag == 1:
            if exitflag == 0:
                info = {"success": False, "return_status": "Successfully Solved", "solve_time": solve_info.solvetime,
                        "info": solve_info}
            else:
                info = {"success": True, "return_status": "Successfully Solved", "solve_time": solve_info.solvetime,
                        "info": solve_info}

            # for k in range(self.N):
            #     sol = output["x%02d" % (k + 1)]
            #     self.x_pred[k, :] = sol[:self.n]
            #     self.u_pred[k, :] = sol[self.n:self.n + self.nu]

            # # Construct initial guess for next iteration
            # x_ws = self.x_pred[1:]
            # u_ws = self.u_pred[1:]
            
            # u_ws = np.vstack((u_ws, u_ws[-1]))  # stack previous input
            # self.set_warm_start(x_ws, u_ws)

            # u = self.u_pred[0]
            # self.n_sol_count = 0

        else:
            info = {"success": False, "return_status": 'Solving Failed, exitflag = %d' % exitflag, "solve_time": None,
                    "info": solve_info}
          
        # self.u_prev = u
        # control = VehicleActuation()
        # self.dynamics.u2input(control, u)

        return output, info, exitflag




