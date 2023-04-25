
from termios import CWERASE
import numpy as np
import math
import time, os

from utils.pycubicspline import Spline2D
from utils.f110 import F110
from utils.minimize_time import calcMinimumTime, calcMinimumTimeSpeedInputs
from utils.MAP import Track


import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch import fit_gpytorch_model

from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler

import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

class randomTrajectory:

	def __init__(self, track, n_waypoints):
		"""	track is an object with the following attributes/methods
				track_width (vector)	: width of the track in [m] (assumed constant at each way point)
				track_length (scaler)	: length of the track in [m]
				param_to_xy (function): converts arc length to x, y coordinates on the track
			n_waypoints is no of points used to fit cubic splines (equal to dim in BayesOpt)
		"""
		self.track = track
		self.n_waypoints = n_waypoints

	def sample_nodes(self, scale):
		""" sample width vector of length `n_waypoints`
		"""
		# shrink to prevent getting too close to corners
		track_width = self.track.track_width*scale
		width = -track_width/2 + track_width*np.random.rand(self.n_waypoints)
		return width

	def calculate_xy(self, width, last_index, theta=None):
		"""	compute x, y coordinates from sampled nodes (width) 
		"""
		track = self.track
		n_waypoints = width.shape[0]
		eps = 1/5/n_waypoints*track.track_length

		# starting and terminal points are fixed
		wx = np.zeros(n_waypoints+2)
		wy = np.zeros(n_waypoints+2)
		wx[0] = track.x_center[0]
		wy[0] = track.y_center[0]
		wx[-1] = track.x_center[last_index]
		wy[-1] = track.y_center[last_index]
		if theta is None:
			theta = np.linspace(0, track.track_length, n_waypoints+2)
		else:
			print(len(theta))
			print(width.shape[0])
			assert width.shape[0]==len(theta), 'dims not equal'
			theta_start = np.array([0])
			theta_end = np.array([self.track.theta_track[last_index]])
			theta = np.concatenate([theta_start, theta, theta_end])

		# compute x, y for every way point parameterized by arc length
		for idt in range(1,n_waypoints+1):
			x_, y_ = track._param2xy(theta[idt]+eps)
			_x, _y = track._param2xy(theta[idt]-eps)
			x, y = track._param2xy(theta[idt])
			norm = np.sqrt((y_-_y)**2 + (x_-_x)**2)
			wx[idt] = x - width[idt-1]*(y_-_y)/norm
			wy[idt] = y + width[idt-1]*(x_-_x)/norm
		return wx, wy

	def fit_cubic_splines(self, wx, wy, n_samples):
		"""	fit cubic splines on the waypoints
		"""
		sp = Spline2D(wx, wy)
		s = np.linspace(0, sp.s[-1]-1e-3, n_samples)
		x, y = [], []
		for i_s in s:
			ix, iy = sp.calc_position(i_s)
			x.append(ix)
			y.append(iy)
		return x, y
	

def evaluate_y(x_eval, mean_y=None, std_y=None):
    """ evaluate true output for given x (distance of nodes from center line)
        TODO: parallelize evaluations
    """
    if type(x_eval) is torch.Tensor:
        is_tensor = True
        x_eval = x_eval.cpu().numpy()
    else:
        is_tensor = False

    if len(x_eval.shape)==1:
        x_eval = x_eval.reshape(1,-1)
    n_eval = x_eval.shape[0]

    y_eval = np.zeros(n_eval)
    for ids in range(n_eval):
        wx, wy = rand_traj.calculate_xy(
            width=x_eval[ids],
            last_index=NODES[LASTIDX],
            theta=theta,
            )
        x, y = rand_traj.fit_cubic_splines(
            wx=wx, 
            wy=wy, 
            n_samples=N_WAYPOINTS,
            )
        y_eval[ids] = -calcMinimumTime(x, y, **params)       # we want to max negative lap times

    if mean_y and std_y:
        y_eval = normalize(y_eval, mean_y, std_y)

    if is_tensor:
        return torch.tensor(y_eval, device=device, dtype=dtype).unsqueeze(-1)
    else:
        return y_eval.ravel()

def generate_initial_data(n_samples=10):
    """ generate training data
    """
    train_x = np.zeros([n_samples, n_waypoints])
    train_y_ = np.zeros([n_samples, 1])

    for ids in range(n_samples):
        width_random = rand_traj.sample_nodes(scale=SCALE)
        t_random = evaluate_y(width_random)
        train_x[ids,:] = width_random
        train_y_[ids,:] = t_random

    mean_y, std_y = train_y_.mean(), train_y_.std()
    train_y = normalize(train_y_, mean_y, std_y)
    train_x = torch.tensor(train_x, device=device, dtype=dtype)
    train_y = torch.tensor(train_y, device=device, dtype=dtype)
    best_y = train_y.max().item()
    return train_x, train_y, best_y, mean_y, std_y

def normalize(y_eval, mean_y, std_y):
    """ normalize outputs for GP
    """
    return (y_eval - mean_y) / std_y

#####################################################################
# modeling and optimization functions called in closed-loop

def initialize_model(train_x, train_y, state_dict=None):
    """initialize GP model with/without initial states
    """
    model = SingleTaskGP(train_x, train_y).to(train_x)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model

def optimize_acqf_and_get_observation(acq_func, mean_y=None, std_y=None):
    """optimize acquisition function and evaluate new candidates
    """
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=10,
        raw_samples=512,  # used for intialization heuristic
    )

    # observe new values 
    new_x = candidates.detach()
    new_y = evaluate_y(new_x, mean_y=mean_y, std_y=std_y)
    return new_x, new_y

def sample_random_observations(mean_y, std_y):
    """sample a random trajectory
    """
    rand_x = torch.tensor(rand_traj.sample_nodes(scale=SCALE).reshape(1,-1), device=device, dtype=dtype)
    rand_y = evaluate_y(rand_x, mean_y=mean_y, std_y=std_y)  
    return rand_x, rand_y

#####################################################################
# main simulation loop


def calc_psi(traj):
    psi_ref = []

    y = traj[0,1] - traj[-1,1] 
    x = traj[0,0] - traj[-1,0]
    psi = math.atan2(y,x)
    psi_ref.append(psi)

    old_psi = psi
    for i in range(1,len(traj)):
        y = traj[i,1] - traj[i-2,1] 
        x = traj[i,0] - traj[i-2,0]
        psi = math.atan2(y,x)
        
        old_psi = psi
        psi_ref.append(psi)

    return psi_ref

def calc_curv(traj):
    dx = np.gradient(traj[:,0])
    dy = np.gradient(traj[:,1])

    d2x = np.gradient(dx)
    d2y = np.gradient(dy)

    curvature = (dx * d2y - d2x * dy) / (dx * dx + dy * dy)**1.5
    
    return curvature


if __name__ == '__main__':
	

	# set device in torch
	device = torch.device("cpu")
	dtype = torch.float


	# simulation settings
	SEED = np.random.randint(1000)
	torch.manual_seed(SEED)
	np.random.seed(SEED)

	BATCH_SIZE = 1              # useful for parallelization, DON'T change
	N_TRIALS =  1              # number of times bayesopt is run
	N_BATCH = 15          # new observations after initialization
	MC_SAMPLES = 64             # monte carlo samples
	N_INITIAL_SAMPLES = 15      # samples to initialize GP
	PLOT_RESULTS = False        # whether to plot results
	SAVE_RESULTS = True         # whether to save results
	N_WAYPOINTS = 300   # resampled waypoints
	ds = 0.1
	SCALE = 0.5      # shrinking factor for track width
	LASTIDX = 1                 # fixed node at the end DO NOT CHANGE
	params = F110()

	dir = os.path.expanduser('~') + '/F1tenth_track/'
	# track
	track = Track(dir+'result/center_traj_with_boundary.txt')
	inner = np.loadtxt(dir+"/result/innerwall.txt",  delimiter=",", dtype = float)  
	outer = np.loadtxt(dir+"/result/outerwall.txt",  delimiter=",", dtype = float)  


	while(1):
		xy_pts = []
		node_pts = []

		def add_point(event):
			if event.inaxes != ax:
				return
			
			if event.button == 1:
				x = event.xdata
				y = event.ydata

				old_dis = 10
				n_id = 0
				for i in range(len(track.x_center)):
					dis = math.sqrt( pow(track.x_center[i] - x,2 ) + pow(track.y_center[i] - y,2 ))
					if dis < old_dis:
						n_id = i
						old_dis = dis
				
				xy_pts.append([x ,y])
				plt.plot(x, y, '*r', lw=3)
				plt.draw()
				node_pts.append(n_id)
				print(node_pts)


		fig = plt.figure()
		ax = fig.add_subplot(111, facecolor='#FFFFCC')
		ax.plot(track.x_center, track.y_center, '--k')
		ax.plot(track.x_center[0], track.y_center[0], '*k')
		ax.plot(track.x_center[-1], track.y_center[-1], '*b')
		plt.plot(outer[:,0], outer[:,1], '-b', lw=1.5)
		plt.plot(inner[:,0], inner[:,1], '-b', lw=1.5)
		cursor = Cursor(ax, useblit=True, color='red', linewidth=2)
		cid = plt.connect('button_press_event', add_point)
		plt.show()

		node_pts.sort()
		# print('final selected pts', node_pts)
		NODES = node_pts
		NODES.append(NODES[0])

		cmd = input("(s): select once again, (.): else anything")

		if cmd == 's':
			continue
		else:
			break


	theta = track.theta_track[NODES]
	N_DIMS = len(NODES)
	n_waypoints = N_DIMS
 
	bounds = torch.tensor([[-track.track_width]*N_DIMS, [track.track_width]*N_DIMS], device=device, dtype=dtype)
	rand_traj = randomTrajectory(track=track,n_waypoints=n_waypoints)

	
	qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)

	verbose = True

	best_observed_all_ei, best_observed_all_nei, best_random_all = [], [], []
	train_x_all_ei, train_x_all_nei, train_x_all_random = [], [], []
	train_y_all_ei, train_y_all_nei, train_y_all_random = [], [], []

    # statistics over multiple trials
	for trial in range(1, N_TRIALS + 1):
		print('\nTrial {} of {}'.format(trial, N_TRIALS))
		best_observed_ei, best_observed_nei = [], []
		best_random = []
		
		# generate initial training data and initialize model
		print('\nGenerating {} random samples'.format(N_INITIAL_SAMPLES))
		train_x_ei, train_y_ei, best_y_ei, mean_y, std_y = generate_initial_data(n_samples=N_INITIAL_SAMPLES)
		denormalize = lambda x: -(x*std_y + mean_y)
		mll_ei, model_ei = initialize_model(train_x_ei, train_y_ei)
		
		train_x_nei, train_y_nei, best_y_nei = train_x_ei, train_y_ei, best_y_ei
		mll_nei, model_nei = initialize_model(train_x_nei, train_y_nei)

		train_x_random, train_y_random, best_y_random = train_x_ei, train_y_ei, best_y_ei

		best_observed_ei.append(denormalize(best_y_ei))
		best_observed_nei.append(denormalize(best_y_nei))
		best_random.append(denormalize(best_y_random))

		# run N_BATCH rounds of BayesOpt after the initial random batch
		for iteration in range(1, N_BATCH + 1):    
			
			print('\nBatch {} of {}\n'.format(iteration, N_BATCH))
			t0 = time.time()
			
			# fit the models
			try:
				fit_gpytorch_model(mll_ei)
				fit_gpytorch_model(mll_nei)
			except:
				break
			
			# update acquisition functions
			qEI = qExpectedImprovement(
				model=model_ei, 
				best_f=train_y_ei.max(),
				sampler=qmc_sampler,
			)
			
			qNEI = qNoisyExpectedImprovement(
				model=model_nei, 
				X_baseline=train_x_nei,
				sampler=qmc_sampler,
			)
			
			# optimize acquisition function and evaluate new sample
			new_x_ei, new_y_ei = optimize_acqf_and_get_observation(qEI, mean_y=mean_y, std_y=std_y)
			print('EI: time to traverse is {:.4f}s'.format(-(new_y_ei.numpy().ravel()[0]*std_y+mean_y)))
			new_x_nei, new_y_nei = optimize_acqf_and_get_observation(qNEI, mean_y=mean_y, std_y=std_y)
			print('NEI: time to traverse is {:.4f}s'.format(-(new_y_nei.numpy().ravel()[0]*std_y+mean_y)))
			new_x_random, new_y_random = sample_random_observations(mean_y=mean_y, std_y=std_y)
			print('Random: time to traverse is {:.4f}s'.format(-(new_y_random.numpy().ravel()[0]*std_y+mean_y)))

			# update training points
			train_x_ei = torch.cat([train_x_ei, new_x_ei])
			train_y_ei = torch.cat([train_y_ei, new_y_ei])

			train_x_nei = torch.cat([train_x_nei, new_x_nei])
			train_y_nei = torch.cat([train_y_nei, new_y_nei])

			train_x_random = torch.cat([train_x_random, new_x_random])
			train_y_random = torch.cat([train_y_random, new_y_random])

			# update progress
			best_value_ei = denormalize(train_y_ei.max().item())
			best_value_nei = denormalize(train_y_nei.max().item())
			best_value_random = denormalize(train_y_random.max().item())
			
			best_observed_ei.append(best_value_ei)
			best_observed_nei.append(best_value_nei)
			best_random.append(best_value_random)

			# reinitialize the models so they are ready for fitting on next iteration
			# use the current state dict to speed up fitting
			mll_ei, model_ei = initialize_model(
				train_x_ei,
				train_y_ei,
				model_ei.state_dict(),
			)
			mll_nei, model_nei = initialize_model(
				train_x_nei,
				train_y_nei, 
				model_nei.state_dict(),
			)
			t1 = time.time()

			if verbose:
				print(
					'best lap time (random, qEI, qNEI) = {:.2f}, {:.2f}, {:.2f}, time to compute = {:.2f}s'.format(
						best_value_random, 
						best_value_ei, 
						best_value_nei,
						t1-t0
						)
				)
			else:
				print(".")
		
		best_observed_all_ei.append(best_observed_ei)
		best_observed_all_nei.append(best_observed_nei)
		best_random_all.append(best_random)

		train_x_all_ei.append(train_x_ei.cpu().numpy())
		train_x_all_nei.append(train_x_nei.cpu().numpy())
		train_x_all_random.append(train_x_random.cpu().numpy())

		train_y_all_ei.append(denormalize(train_y_ei.cpu().numpy()))
		train_y_all_nei.append(denormalize(train_y_nei.cpu().numpy()))
		train_y_all_random.append(denormalize(train_y_random.cpu().numpy()))

	# iters = np.arange(N_BATCH + 1) * BATCH_SIZE
	iters = np.arange(iteration+1) * BATCH_SIZE
	y_ei = np.asarray(best_observed_all_ei)
	y_nei = np.asarray(best_observed_all_nei)
	y_rnd = np.asarray(best_random_all)
	
	
	np.savez(
		'./result/raceline.npz',
		y_ei=y_ei,
		y_nei=y_nei,
		y_rnd=y_rnd,
		iters=iters,
		train_x_all_ei=np.asarray(train_x_all_ei),
		train_x_all_nei=np.asarray(train_x_all_nei),
		train_x_all_random=np.asarray(train_x_all_random),
		train_y_all_ei=np.asarray(train_y_all_ei),
		train_y_all_nei=np.asarray(train_y_all_nei),
		train_y_all_random=np.asarray(train_y_all_random),
		SEED=SEED,
		)

	def ci(y):
		return 1.96 * y.std(axis=0) / np.sqrt(N_TRIALS)

	plt.figure()
	plt.gca().set_prop_cycle(None)
	plt.plot(iters, y_rnd.mean(axis=0), linewidth=1.5)
	plt.plot(iters, y_ei.mean(axis=0), linewidth=1.5)
	plt.plot(iters, y_nei.mean(axis=0), linewidth=1.5)
	plt.gca().set_prop_cycle(None)
	plt.fill_between(iters, y_rnd.mean(axis=0)-ci(y_rnd), y_rnd.mean(axis=0)+ci(y_rnd), label='random', alpha=0.2)
	plt.fill_between(iters, y_ei.mean(axis=0)-ci(y_ei), y_ei.mean(axis=0)+ci(y_ei), label='qEI', alpha=0.2)
	plt.fill_between(iters, y_nei.mean(axis=0)-ci(y_nei), y_nei.mean(axis=0)+ci(y_nei), label='qNEI', alpha=0.2)
	plt.xlabel('number of observations (beyond initial points)')
	plt.ylabel('best lap times')
	plt.grid(True)
	plt.legend(loc=0)
	plt.show()


	theta = track.theta_track[NODES]
	data = np.load("./result/raceline.npz", allow_pickle=True)

	y_ei = data['y_ei']
	y_nei = data['y_nei']
	y_rnd = data['y_rnd']
	iters = data['iters']
	train_x_all_ei = data['train_x_all_ei']
	train_x_all_nei = data['train_x_all_nei']
	train_x_all_random = data['train_x_all_random']
	train_y_all_ei = data['train_y_all_ei'].squeeze(-1)
	train_y_all_nei = data['train_y_all_nei'].squeeze(-1)
	train_y_all_random = data['train_y_all_random'].squeeze(-1)

	N_TRIALS = train_x_all_ei.shape[0]
	N_DIMS = train_x_all_ei.shape[-1]

	N_DIMS = len(NODES)
	n_waypoints = N_DIMS
	n_samples = 420
	sim = 0

	x_center, y_center = track.x_center, track.y_center
	rand_traj = randomTrajectory(track=track, n_waypoints=n_waypoints)

	def gen_traj(x_all, idx, sim):
		w_idx = x_all[sim][idx]
		wx, wy = rand_traj.calculate_xy(
			width=w_idx,
			last_index=NODES[LASTIDX],
			theta=theta,
			)
		sp = Spline2D(wx, wy)
		s = np.arange(0, sp.s[-1]-0.001, ds)
		x, y = [], []
		for i_s in s:
			ix, iy = sp.calc_position(i_s)
			x.append(ix)
			y.append(iy)
		return wx, wy, x, y

	fig = plt.figure()
	ax = plt.gca()
	ax.axis('equal')
	plt.plot(x_center, y_center, '--k', lw=0.5, alpha=0.5)

	EI = True
	if EI:
		train_x_all_nei = train_x_all_ei
		train_y_all_nei = train_y_all_ei

	# best trajectory
	sim, pidx = np.unravel_index(np.argmin(train_y_all_nei), train_y_all_nei.shape)
	wx_nei, wy_nei, x_nei, y_nei = gen_traj(train_x_all_nei, pidx, sim)
	plt.plot(wx_nei[:-2], wy_nei[:-2], linestyle='', marker='D', ms=5)

	time, speed, inputs = calcMinimumTimeSpeedInputs(x_nei, y_nei, **params)
	x = np.array(x_nei)
	y = np.array(y_nei)

	points = np.array([x, y]).T.reshape(-1, 1, 2)
	segments = np.concatenate([points[:-1], points[1:]], axis=1)
	norm = plt.Normalize(speed.min(), speed.max())
	lc = LineCollection(segments, cmap='viridis', norm=norm)
	lc.set_array(speed)
	lc.set_linewidth(2)
	line = ax.add_collection(lc)
	fig.colorbar(line, ax=ax)
	ax.set_xlabel('x [m]')
	ax.set_ylabel('y [m]')

	idx = -1
	dis_n = math.sqrt((x[0]-x[1])**2 + (y[0]-y[1])**2)
	for i in range(len(x)):
		dis = math.sqrt((x[-1]-x[i])**2 + (y[-1]-y[i])**2)
		if dis <= 3*dis_n:
			idx = i
			print(idx)
			break

	
	if idx >= 100:
		idx = 0


	traj = np.empty((len(x)-idx,4))
	traj[:,0] = x[idx:]
	traj[:,1] = y[idx:]
	traj[:,2] = speed[idx:]
	traj[:,3] = time[idx:]


	plt.plot(traj[0,0], traj[0,1],'bo')
	plt.plot(traj[20,0], traj[20,1],'ro')

	plt.text(traj[1,0], traj[1,1],
            "START",
            horizontalalignment ="center", 
            verticalalignment ="center", 
            wrap = True, fontsize = 12, 
            color ="blue")

	plt.text(traj[22,0], traj[22,1],
            "NEXT",
            horizontalalignment ="center", 
            verticalalignment ="center", 
            wrap = True, fontsize = 12, 
            color ="red")
	plt.show()

	dir = input("Direction is CW? or CCW? ")


	psi_c = calc_psi(traj[:,0:2])
	psi1 = np.mean(psi_c[:len(psi_c)//4])
	psi2 = np.mean(psi_c[len(psi_c)//4:len(psi_c)//2])
	if dir == "CCW":
		print("CCW")
	else:
		print("CW") 
		traj = np.flip(traj, axis = 0)

	curv = calc_curv(traj)
	traj[:,3] = np.abs(curv)
	np.savetxt("./result/opt_traj.txt",traj,delimiter=",")

	fig = plt.figure()
	ax = plt.gca()
	ax.axis('equal')
	plt.plot(x_center, y_center, '--k', lw=0.5, alpha=0.5)
	points = np.array([traj[:,0], traj[:,1]]).T.reshape(-1, 1, 2)
	segments = np.concatenate([points[:-1], points[1:]], axis=1)
	norm = plt.Normalize(traj[:,2].min(), traj[:,2].max())
	lc = LineCollection(segments, cmap='viridis', norm=norm)
	lc.set_array(traj[:,2])
	lc.set_linewidth(2)
	line = ax.add_collection(lc)
	fig.colorbar(line, ax=ax)
	ax.set_xlabel('x [m]')
	ax.set_ylabel('y [m]')

	

	plt.plot(outer[:,0], outer[:,1], '-b', lw=1.5)
	plt.plot(outer[0,0], outer[0,1], '*b', lw=1.5)
	plt.plot(inner[:,0], inner[:,1], '-b', lw=1.5)
	plt.plot(inner[0,0], inner[0,1], '*b', lw=1.5)
	plt.plot(traj[0,0], traj[0,1], '*r', lw=1.5)

	plt.show()


