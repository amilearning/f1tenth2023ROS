
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import math

import os
from utils.pycubicspline import Spline2D
from utils.f110 import F110
from utils.minimize_time import calcMinimumTimeSpeedInputs
from utils.MAP import Track



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
	


def calc_psi(traj):
    psi_ref = []

    y = traj[0,1] - traj[-1,1] 
    x = traj[0,0] - traj[-1,0]
    psi = math.atan2(y,x)
    # psi = (psi + np.pi)%(2*np.pi)
    psi_ref.append(psi)

    old_psi = psi
    for i in range(1,len(traj)):
        y = traj[i,1] - traj[i-1,1] 
        x = traj[i,0] - traj[i-1,0]
        psi = math.atan2(y,x)
        
        # psi = (psi + np.pi)%(2*np.pi)
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



#### MAIN FUNCTION
SCALE = 0
params = F110()
n_samples = 400
DiffN = 3
Dis_i = 1.5
Dis_o = 2.5

dir = os.path.expanduser('~') + '/F1tenth_track/'
CenterFile = dir+"track_extraction/map/map_data-centerline.csv"
InnerBoundaryFile = dir+"track_extraction/map/map_data-innerwall.csv"
OuterBoundaryFile = dir+"track_extraction/map/map_data-outerwall.csv"

fig = plt.figure()
ax = plt.gca()
ax.axis('equal')

inner = np.loadtxt(InnerBoundaryFile, delimiter=",", dtype = float)
outer = np.loadtxt(OuterBoundaryFile, delimiter=",", dtype = float)
center = np.loadtxt(CenterFile, delimiter=",", dtype = float)
# plt.plot(inner[10,0],inner[10,1], '*g')
# plt.plot(inner[20,0],inner[20,1], '.b')
# plt.plot(inner[:,0],inner[:,1], '--k')
# plt.plot(outer[10,0],outer[10,1], '*g')
# plt.plot(outer[20,0],outer[20,1], '.b')
# plt.plot(outer[:,0],outer[:,1], '--k')
# plt.plot(center[:,0],center[:,1], '--k')

# plt.text(center[1,0], center[1,1],
# 		"START",
# 		wrap = True, fontsize = 12,
# 		horizontalalignment ="center", 
#         verticalalignment ="center",  
# 		color ="blue")

# plt.text(center[22,0], center[22,1],
# 		"NEXT",
# 		wrap = True, fontsize = 12, 
# 		horizontalalignment ="center", 
#         verticalalignment ="center", 
# 		color ="red")

# plt.show()

# cmd = input("Direction is CW or CCW?? ")
InnerDirection = "CCW"
OuterDirection = "CCW"
centerDirection = "CCW"


## CENTER LINE
track = Track(CenterFile)
NODES = []
for i in range(2,len(track.center_line[0]),5):
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
x_ref, y_ref = rand_traj.fit_cubic_splines(
	wx=wx_random, 
	wy=wy_random, 
	n_samples=n_samples
	)

pt0 = [x_ref[0], y_ref[0]]
idx = -1
m = n_samples
n_dis = math.sqrt((x_ref[0]-x_ref[1])**2 + (y_ref[0]-y_ref[1])**2)
for i in range(1,m):
    pt = [x_ref[i], y_ref[i]]
    dis = math.sqrt((pt0[0]-pt[0])**2 + (pt0[1]-pt[1])**2)
    if i == 1:
        n_dis = dis
    if i > 10 and dis <= n_dis:
        idx = i
        break
x_center = x_ref[:idx]
y_center = y_ref[:idx]
center = np.empty((len(x_center),2))
center[:,0] = x_center
center[:,1] = y_center



psi = calc_psi(center[:,0:2])
# plt.plot(psi, 'b')
psi1 = np.mean(psi[:len(psi)//3])
psi2 = np.mean(psi[len(psi)//3:len(psi)//2])
if centerDirection == "CCW":
    print("CCW")
else:
    print("CW") 
    center = np.flip(center, axis = 0)  
psi = calc_psi(center[:,0:2])


curv = calc_curv(center[:,0:2])





## INNER BOUNDARY
n_samples_i = n_samples
track = Track(InnerBoundaryFile)
NODES = []
for i in range(2,len(track.center_line[0]),4):
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
	n_samples=n_samples_i
	)

pt0 = [x_inner[0], y_inner[0]]
idx = -1
m = n_samples_i
n_dis = 0.2

for i in range(1,m):
    pt = [x_inner[i], y_inner[i]]
    dis = math.sqrt((pt0[0]-pt[0])**2 + (pt0[1]-pt[1])**2)
    if i == 1:
        n_dis = dis

    if i > 10 and dis <= n_dis:
        idx = i
        break

x_inner = x_inner[:idx]
y_inner = y_inner[:idx]

traj_i = np.empty((len(x_inner),2))
traj_i[:,0] = x_inner
traj_i[:,1] = y_inner








## OUTER BOUNDARY

track = Track(OuterBoundaryFile)
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

pt0 = [x_outer[0], y_outer[0]]
idx = -1
m = n_samples
n_dis = math.sqrt((x_outer[0]-x_outer[1])**2 + (y_outer[0]-y_outer[1])**2)
for i in range(1,m):
    pt = [x_outer[i], y_outer[i]]
    dis = math.sqrt((pt0[0]-pt[0])**2 + (pt0[1]-pt[1])**2)
    if i == 1:
        n_dis = dis
    if i > 10 and dis <= n_dis:
        idx = i
        break

x_outer = x_outer[:idx]
y_outer = y_outer[:idx]

plt.plot(x_outer,y_outer, 'b')
plt.plot(x_outer[0],y_outer[0],'*b')
plt.plot(x_outer[-1],y_outer[-1],'*r')

traj_o = np.empty((len(x_outer),2))
traj_o[:,0] = x_outer
traj_o[:,1] = y_outer





# Start Line with Center line
pt0 = [center[0,0], center[0,1]]
old_dis = 10
idx = -1

m = len(traj_i)
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
m = len(traj_o)
for i in range(m):
    pt = [traj_o[i,0], traj_o[i,1]]
    dis = math.sqrt((pt0[0]-pt[0])**2 + (pt0[1]-pt[1])**2)
    if dis < old_dis:
        old_dis = dis
        idx = i

outer = np.empty((m,2))
outer[:m-idx] = traj_o[idx:]
outer[m-idx:] = traj_o[:idx]




if InnerDirection == "CCW":
	inner = np.flip(inner, axis = 0)
if OuterDirection == "CCW":
	outer = np.flip(outer, axis = 0)









## Compute the reference speed of center 
params = F110()
time, speed, inputs = calcMinimumTimeSpeedInputs(center[:,0], center[:,1], **params)
x = np.array(center[:,0])
y = np.array(center[:,1])


n = 20
if speed[0] < 1:
    speed_i = speed[-1]
    speed_o = speed[n]


    if speed_i < speed_o:
        new_speed = np.linspace(speed_i,speed_o,n)
        new_speed = np.flip(new_speed)
        for i in range(n):
            speed[i] = new_speed[i]
            
    elif speed_i > speed_o:
        new_speed = np.linspace(speed_o,speed_i,n)
        for i in reversed(range(n)):
            speed[i] = new_speed[i]

points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
norm = plt.Normalize(speed.min(), speed.max())
lc = LineCollection(segments, cmap='viridis', norm=norm)
lc.set_array(speed)
lc.set_linewidth(5)
line = ax.add_collection(lc)
# fig.colorbar(line, ax=ax)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')



plt.plot(inner[:,0],inner[:,1], '--k')
plt.plot(outer[:,0],outer[:,1], '--k')
plt.plot(center[:,0],center[:,1], '--k')

plt.text(center[1,0], center[1,1],
		"START",
		wrap = True, fontsize = 12,
		horizontalalignment ="center", 
        verticalalignment ="center",  
		color ="blue")

plt.text(center[22,0], center[22,1],
		"NEXT",
		wrap = True, fontsize = 12, 
		horizontalalignment ="center", 
        verticalalignment ="center", 
		color ="red")

plt.show()

cmd = input("Direction is CW or CCW?? ")

if cmd == "CW":
	center = np.flip(center, axis = 0)  
	psi = calc_psi(center[:,0:2])
	curv = calc_curv(center[:,0:2])






Li = 1
Lo = 1

RightTrack = []
plt.plot(inner[:,0],inner[:,1], '--k')
plt.plot(outer[:,0],outer[:,1], '--k')
plt.plot(center[:,0],center[:,1], '--k')

inner_track_width = []
inner_nearest_points = []
for i, center_point in enumerate(center):
	distances = []
	for j, boundary_point in enumerate(inner):
		distance = math.sqrt((center_point[0]-boundary_point[0])**2 + (center_point[1]-boundary_point[1])**2)
		distances.append(distance)
	min_distance = min(distances)
	min_index = distances.index(min_distance)
	inner_track_width. append(min_distance)
	inner_nearest_points.append(inner[min_index])
 
outer_track_width = []
outer_nearest_points = []
for i, center_point in enumerate(center):
	distances = []
	for j, boundary_point in enumerate(outer):
		distance = math.sqrt((center_point[0]-boundary_point[0])**2 + (center_point[1]-boundary_point[1])**2)
		distances.append(distance)
	min_distance = min(distances)
	min_index = distances.index(min_distance)
	outer_track_width. append(min_distance)
	outer_nearest_points.append(outer[min_index])

for i in range(len(inner_nearest_points)):
    plt.plot(inner_nearest_points[i][0],inner_nearest_points[i][1], 'r*')
    plt.plot(outer_nearest_points[i][0],outer_nearest_points[i][1], 'b*')
    A = np.array( [ [inner_nearest_points[i][0], inner_nearest_points[i][1]],  [center[i,0], center[i,1]], [outer_nearest_points[i][0],outer_nearest_points[i][1]] ])
    plt.plot(A[:,0],A[:,1])



final_opt_traj = np.empty((len(center),6))
final_opt_traj[:,:2] = center
final_opt_traj[:,2] = speed
final_opt_traj[:,3] = curv
final_opt_traj[:,4] = np.array(inner_track_width) #Left
final_opt_traj[:,5] = np.array(outer_track_width) #Right
# final_opt_traj[:,4:6] = 0
# final_opt_traj[:,6:8] = 0



dir = os.path.expanduser('~') + '/F1tenth_track/'

np.savetxt(dir+'result/center_traj_with_boundary.txt', final_opt_traj, delimiter=",")
np.savetxt(dir+'result/outerwall.txt', outer ,delimiter=",")
np.savetxt(dir+'result/innerwall.txt', inner ,delimiter=",")
np.savetxt(dir+'result/centerline.txt', center ,delimiter=",")

plt.show()
