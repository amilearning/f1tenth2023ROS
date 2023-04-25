""" Plot optimal racing lines from saved results.
    See generate_raceline_f110.py.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np


from trajectory import randomTrajectory
from hmctrack.utils.pycubicspline import Spline2D
from hmctrack.utils.f110 import F110
from hmctrack.utils.minimize_time import calcMinimumTime, calcMinimumTimeSpeed,calcMinimumTimeSpeedInputs
from hmctrack.utils.MAP import Track

import math
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


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


SAVE_RESULTS = True

params = F110()
track = Track("optimizetraj/centerline.txt")
NODES = []
# for i in range(0,len(track.center_line[0]),12):
# 	NODES.append(i)
# NODES.append(0)
NODES = []
for i in range(0,len(track.center_line[0]),20):
    NODES.append(i)
NODES.append(0)

# NODES = [0, 20, 35, 50, 70, 90, 115, 130, 160, 180, 195, 210, 230, 245, 255, 270, 290, 300, 315, 330]
LASTIDX = 1


theta = track.theta_track[NODES]





'''
centerline
'''

data = np.load("optimizetraj/raceline.npz", allow_pickle=True)

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


#####################################################################
# plot best trajectory
filepath = 'racetrack.png'

N_DIMS = len(NODES)
n_waypoints = N_DIMS
n_samples = 400
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
    s = np.linspace(0, sp.s[-1]-0.001, n_samples)
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
# plt.plot(x_outer, y_outer, 'k', lw=0.5, alpha=0.5)
# plt.plot(x_inner, y_inner, 'k', lw=0.5, alpha=0.5)

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
    if dis <= 1.5*dis_n:
        idx = i
        break

print(idx)


traj = np.empty((len(x)-idx,4))
traj[:,0] = x[idx:]
traj[:,1] = y[idx:]
traj[:,2] = speed[idx:]
traj[:,3] = time[idx:]

# idx = -1
# for i in range(1,len(x)):
#     dis = math.sqrt((x[0]-x[i])**2 + (y[0]-y[i])**2)
#     if i>= 5 and dis < 0.3:
#         idx = i
#         break

# print(idx)


# traj = np.empty((idx,4))
# traj[:,0] = x[:idx]
# traj[:,1] = y[:idx]
# traj[:,2] = speed[:idx]
# traj[:,3] = time[:idx]


psi_c = calc_psi(traj[:,0:2])
psi1 = np.mean(psi_c[:len(psi_c)//2])
psi2 = np.mean(psi_c[len(psi_c)//2:])
if psi1 < psi2:
    print("CCW")
else:
    print("CW") 
    traj = np.flip(traj, axis = 0)

np.savetxt("opt_traj.txt",traj,delimiter=",",)


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





'''
Boundary Line
'''
SCALE = 0
track = Track("map_data_0427-innerwall.csv")
NODES = []
for i in range(2,len(track.center_line[0]),2):
    NODES.append(i)
NODES.append(0)
LASTIDX = 0


theta = track.theta_track[NODES]
n_waypoints = len(theta)
rand_traj = randomTrajectory(
    track=track,
    n_waypoints=n_waypoints,
    )
width_random = rand_traj.sample_nodes(scale=SCALE)

wx_random, wy_random = rand_traj.calculate_xy(
    width_random,
    last_index=NODES[LASTIDX],
    theta=theta,
    )

x_inner, y_inner = rand_traj.fit_cubic_splines(
    wx=wx_random, 
    wy=wy_random, 
    n_samples=n_samples
    )

traj_i = np.empty((len(x_inner),2))
traj_i[:,0] = x_inner
traj_i[:,1] = y_inner
psi_i = calc_psi(traj_i[:,0:2])
psi1 = np.mean(psi_i[:len(psi_i)//2])
psi2 = np.mean(psi_i[len(psi_i)//2:])
if psi1 < psi2:
    print("CCW")
else:
    print("CW") 
    traj_i = np.flip(traj_i, axis = 0)    






# Outer boundary
track = Track("map_data_0427-outerwall.csv")
NODES = []
for i in range(2,len(track.center_line[0]),2):
    NODES.append(i)
NODES.append(0)
LASTIDX = 0

theta = track.theta_track[NODES]
n_waypoints = len(theta)
rand_traj = randomTrajectory(
    track=track,
    n_waypoints=n_waypoints,
    )
width_random = rand_traj.sample_nodes(scale=SCALE)

wx_random, wy_random = rand_traj.calculate_xy(
    width_random,
    last_index=NODES[LASTIDX],
    theta=theta,
    )
x_outer, y_outer = rand_traj.fit_cubic_splines(
		wx=wx_random, 
		wy=wy_random, 
		n_samples=n_samples
		)
traj_o = np.empty((len(x_outer),2))
traj_o[:,0] = x_outer
traj_o[:,1] = y_outer
psi_o = calc_psi(traj_o[:,0:2])
psi1 = np.mean(psi_o[:len(psi_o)//2])
psi2 = np.mean(psi_o[len(psi_o)//2:])
if psi1 < psi2:
    print("CCW")
else:
    print("CW") 
    traj_i = np.flip(traj_i, axis = 0) 



pt0 = [traj[0,0], traj[0,1]]
old_dis = 10
idx = -1

m = len(x_inner)
for i in range(m):
    pt = [traj_i[i,0], traj_i[i,1]]
    dis = math.sqrt((pt0[0]-pt[0])**2 + (pt0[1]-pt[1])**2)
    if dis < old_dis:
        old_dis = dis
        idx = i

inner = np.empty((m,2))
inner[:m-idx] = traj_i[idx:]
inner[m-idx:] = traj_i[:idx]


old_dis = 10
idx = -1
m = len(x_outer)
for i in range(m):
    pt = [traj_o[i,0], traj_o[i,1]]
    dis = math.sqrt((pt0[0]-pt[0])**2 + (pt0[1]-pt[1])**2)
    if dis < old_dis:
        old_dis = dis
        idx = i

outer = np.empty((m,2))
outer[:m-idx] = traj_o[idx:]
outer[m-idx:] = traj_o[:idx]


plt.plot(outer[:,0], outer[:,1], '-b', lw=1.5)
plt.plot(outer[0,0], outer[0,1], '*b', lw=1.5)
plt.plot(inner[:,0], inner[:,1], '-b', lw=1.5)
plt.plot(inner[0,0], inner[0,1], '*b', lw=1.5)
plt.plot(traj[0,0], traj[0,1], '*r', lw=1.5)


if SAVE_RESULTS:
    
    np.savez('optimized_traj.npz', x=x, y=y, time=time, speed=speed, inputs=inputs)


plt.show()
