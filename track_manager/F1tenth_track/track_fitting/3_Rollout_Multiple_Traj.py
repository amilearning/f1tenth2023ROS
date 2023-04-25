
# from asyncio import to_thread
from dis import dis
import numpy as np
import matplotlib.pyplot as plt
import math

from pycubicspline import Spline2D
from minimize_time import calcMinimumTimeSpeed
from f110 import F110



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


def generate_new_traj(Li,Lo,traj,psi):

    curv = calc_curv(traj)
    
    N = 4
    M = len(traj)//N
    new_traj1 = np.empty((M,2))
    for i in range(M): 
        if i*N < 1:
            psi[i*N] = psi[2]
        if i*N == len(traj)-1:
            psi[i*N] = psi[-2]

        if abs(curv[i*N]) >= 0.15:
            Xboundleft=traj[i*N,0]
            Yboundleft=traj[i*N,1]
        else:
            Xboundleft=traj[i*N,0] - Li*np.sin(psi[i*N])
            Yboundleft=traj[i*N,1] + Li*np.cos(psi[i*N])

        new_traj1[i,0] = Xboundleft
        new_traj1[i,1] = Yboundleft

    
    N = 4
    M = len(traj)//N
    new_traj2 = np.empty((M,2))
    for i in range(M): 
        if i*N < 1:
            psi[i*N] = psi[2]
        if i*N == len(traj)-1:
            psi[i*N] = psi[-2]

        if abs(curv[i*N]) >= 0.15:
            Xboundright=traj[i*N,0]
            Yboundright=traj[i*N,1]
        else:
            Xboundright=traj[i*N,0] + Lo*np.sin(psi[i*N])
            Yboundright=traj[i*N,1] - Lo*np.cos(psi[i*N])

        new_traj2[i,0] = Xboundright
        new_traj2[i,1] = Yboundright
    

    return new_traj1, new_traj2


def generate_new_traj2(traj,num1,num2):

    curv = calc_curv(traj)
    N = num1
    M = len(traj)//N
    new_traj1 = np.empty((M,2))
    for i in range(M):


        x = (traj[i*N,0] + traj[i*N,4])/2
        y = (traj[i*N,1] + traj[i*N,5])/2
        
        dis = np.sqrt( (traj[i*N-10:i*N,4]-x)**2 + (traj[i*N-10:i*N,5]-y )**2 )
        n_dis = np.where(dis<=0.7)
        if abs(curv[i*N]) >= 0.2:
            x = traj[i*N ,0]
            y = traj[i*N ,1]
        elif len(n_dis[0]) != 0:
            nn = np.argmin(dis)
            x = (traj[i*N ,0] + traj[i*N-10+nn ,4])/2
            y = (traj[i*N ,1] + traj[i*N-10+nn ,5])/2
        elif len(dis) != 0 and dis[-1] > 1:
            x = (traj[i*N, 0] + 2*traj[i*N, 4])/3
            y = (traj[i*N, 1] + 2*traj[i*N, 5])/3
        elif len(dis) != 0 and dis[-1] > 1.2:
            x = (traj[i*N, 0] + 3*traj[i*N, 4])/4
            y = (traj[i*N, 1] + 3*traj[i*N, 5])/4

        new_traj1[i,0] = x
        new_traj1[i,1] = y

    N = num2
    M = len(traj)//N
    new_traj2 = np.empty((M,2))
    for i in range(M):

        x = (traj[i*N,0] + traj[i*N,6])/2
        y = (traj[i*N,1] + traj[i*N,7])/2

        dis = np.sqrt( (traj[i*N-10:i*N,6]-x)**2 + (traj[i*N-10:i*N,7]-y )**2 )
        n_dis = np.where(dis<=0.6)

        if abs(curv[i*N]) >= 0.2:
            x = traj[i*N ,0]
            y = traj[i*N ,1]
        elif len(n_dis[0]) != 0:
            nn = np.argmin(dis)
            x = (traj[i*N,0] + traj[i*N-10+nn ,6])/2
            y = (traj[i*N,1] + traj[i*N-10+nn ,7])/2
        elif len(dis) != 0 and dis[-1] > 1:
            x = (traj[i*N, 0] + 2*traj[i*N, 6])/3
            y = (traj[i*N, 1] + 2*traj[i*N, 7])/3
        elif len(dis) != 0 and dis[-1] > 1.2:
            x = (traj[i*N, 0] + 3*traj[i*N, 6])/4
            y = (traj[i*N, 1] + 3*traj[i*N, 7])/4

        new_traj2[i,0] = x
        new_traj2[i,1] = y

    return new_traj1, new_traj2


def calc_speed(traj):
    new_traj = np.empty((len(traj),8))
    params = F110()
    time, speed = calcMinimumTimeSpeed(traj[:,0], traj[:,1], **params)
    speed = ModifyVelocity(speed)
    curv = calc_curv(traj)
    new_traj[:,:2] = traj[:,:2]
    new_traj[:,2] = speed
    new_traj[:,3] = curv

    return new_traj


def sample_nodes(n_waypoints):
    track_width = 0
    width = -track_width/2 + track_width*np.random.rand(n_waypoints)
    return width

def calculate_xy(traj, width, last_index, theta=None):
    n_waypoints = width.shape[0]
    track_length = calc_track_length(traj)
    theta_track = calc_theta_track(traj)
    eps = 1/5/n_waypoints*track_length

    # starting and terminal points are fixed
    wx = np.zeros(n_waypoints+2)
    wy = np.zeros(n_waypoints+2)
    wx[0] = traj[0,0]
    wy[0] = traj[0,1]
    wx[-1] = traj[last_index,0]
    wy[-1] = traj[last_index,0]
    if theta is None:
        theta = np.linspace(0, track_length, n_waypoints+2)
    else:
        assert width.shape[0]==len(theta), 'dims not equal'
        theta_start = np.array([0])
        theta_end = np.array([theta_track[last_index]])
        theta = np.concatenate([theta_start, theta, theta_end])
    
    for idt in range(1,n_waypoints+1):
        x_, y_ = param2xy(traj, theta_track, theta[idt]+eps)
        _x, _y = param2xy(traj, theta_track, theta[idt]-eps)
        x, y = param2xy(traj, theta_track,theta[idt])
        norm = np.sqrt((y_-_y)**2 + (x_-_x)**2)
        wx[idt] = x - width[idt-1]*(y_-_y)/norm
        wy[idt] = y + width[idt-1]*(x_-_x)/norm
    return wx, wy

def param2xy(traj, theta_track, theta):
    idt = 0
    while idt<theta_track.shape[0]-1 and theta_track[idt]<=theta:
        idt+=1
    deltatheta = (theta-theta_track[idt-1])/(theta_track[idt]-theta_track[idt-1])
    x = traj[idt-1,0] + deltatheta*(traj[idt,0]-traj[idt-1,0])
    y = traj[idt-1,1] + deltatheta*(traj[idt,1]-traj[idt-1,1])
    return x, y

def calc_theta_track(traj):
    traj = np.transpose(traj)
    diff = np.diff(traj)
    theta_track = np.cumsum(np.linalg.norm(diff, 2, axis=0))
    theta_track = np.concatenate([np.array([0]), theta_track])
    return theta_track

def calc_track_length(traj):
    center = np.transpose(traj)
    center = np.concatenate([center, center[:,0].reshape(-1,1)], axis=1) 
    diff = np.diff(center)
    track_length = np.sum(np.linalg.norm(diff, 2, axis=0))
    return track_length


def fit_cubic_splines(wx, wy, n_samples):
    sp = Spline2D(wx, wy)
    s = np.linspace(0, sp.s[-1]-1e-3, n_samples)
    x, y = [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        x.append(ix)
        y.append(iy)
    return x, y

def curvfitting(traj, n_samples):
    
    NODES = []
    for i in range(0,len(traj),5):
        NODES.append(i)

    NODES.append(0)
    LASTIDX = 1

    theta_track = calc_theta_track(traj)


    theta = theta_track[NODES]
    n_waypoints = len(theta)

    width_random = sample_nodes(n_waypoints)

    wx_random, wy_random = calculate_xy(
        traj = traj,
        width = width_random,
        last_index=NODES[LASTIDX],
        theta=theta,
        )
    x_ref, y_ref = fit_cubic_splines(
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
        if i > 10 and dis <= n_dis*1:
            idx = i
            break
    x_center = x_ref[:idx]
    y_center = y_ref[:idx]
    traj = np.empty((len(x_center),2))
    traj[:,0] = x_center
    traj[:,1] = y_center

    return traj


def ModifyVelocity(speed):
    n = 15

    speed_i = speed[-1]
    speed_o = speed[n]
    if speed_i < speed_o:
        new_speed = np.linspace(speed_i,speed_o,n)
        new_speed = np.flip(new_speed)
        speed[:n] = new_speed

    elif speed_i > speed_o:
        new_speed = np.linspace(speed_o,speed_i,n)
        speed[:n] = new_speed

    return speed


def Calc_Width(traj):
    ## Calculate the Width of centerline
    width_i = []
    width_o = []

    for i in range(len(traj)):
        dis_i = math.sqrt((traj[i,0]-traj[i,4])**2 + (traj[i,1]-traj[i,5])**2)
        dis_o = math.sqrt((traj[i,0]-traj[i,6])**2 + (traj[i,1]-traj[i,7])**2)
        width_i.append(dis_i)
        width_o.append(dis_o)

    return width_i, width_o


CenterFile = "./center_traj_with_boundary.txt"
InnerBoundaryFile = "./innerwall.txt"
OuterBoundaryFile = "./outerwall.txt"
OptimizeFile = "./opt_traj.txt"


traj = np.loadtxt(OptimizeFile, delimiter=",", dtype = float)
inner = np.loadtxt(InnerBoundaryFile, delimiter=",", dtype = float)
outer = np.loadtxt(OuterBoundaryFile, delimiter=",", dtype = float)
center = np.loadtxt(CenterFile, delimiter=",", dtype = float)


## Match the start line
dx = 0.5
idx = -1
pt = [traj[0,0], traj[0,1]]

idx =np.argmin(abs(np.linalg.norm(center[:,:2]-pt,axis=1)))


new_center = np.empty((len(center),8))
new_center[:len(center)-idx] = center[idx:]
new_center[len(center)-idx:] = center[:idx]


new_inner= np.empty((len(inner),2))
new_inner[:len(inner)-idx] = inner[idx:]
new_inner[len(inner)-idx:] = inner[:idx]
inner = new_inner


new_outer= np.empty((len(outer),2))
new_outer[:len(outer)-idx] = outer[idx:]
new_outer[len(outer)-idx:] = outer[:idx]
outer = new_outer





psi = calc_psi(traj)
psi_c = calc_psi(new_center)



plt.plot(inner[:,0], inner[:,1], '-k')
plt.plot(outer[:,0], outer[:,1],'-k')






## Generate New Trajectory
traj1,traj2 = generate_new_traj(0.3,0.3,traj,psi)
traj3,traj4 = generate_new_traj(0.4,0.4,new_center,psi_c)
# traj5,traj6 = generate_new_traj(0.5,0.5,new_center,psi_c)
# traj5,traj6 = generate_new_traj2(traj,5,5)


print("trajectory length: ", len(traj))
print("trajectory length: ", len(traj1), len(traj2), len(traj3), len(traj4))

Nx = int(input("Nx: "))
Nx2 = int(input("Nx2: "))

traj1 = curvfitting(traj1,len(traj)+Nx)
traj2 = curvfitting(traj2,len(traj)+Nx)
traj3 = curvfitting(traj3,len(traj)+Nx2)
traj4 = curvfitting(traj4,len(traj)+Nx2)
# traj5 = curvfitting(traj5,len(traj)+Nx2)
# traj6 = curvfitting(traj6,len(traj)+Nx2)

# traj5 = curvfitting(traj5,len(traj)+Nx2)
# traj6 = curvfitting(traj6,len(traj)+Nx2)
# traj7 = curvfitting(traj7,len(traj)+Nx2)
# traj8 = curvfitting(traj8,len(traj)+Nx2)


plt.plot(traj[:,0], traj[:,1], '-g', lw=2, alpha=0.5)
plt.plot(traj1[:,0], traj1[:,1],'-b', lw=2, alpha=0.5)
plt.plot(traj2[:,0], traj2[:,1], '-c', lw=2, alpha=0.5)
plt.plot(traj3[:,0], traj3[:,1], '-r', lw=2, alpha=0.5)
plt.plot(traj4[:,0], traj4[:,1], '-m', lw=2, alpha=0.5)
# plt.plot(traj5[:,0], traj5[:,1], '*b', lw=2, alpha=0.2)
# plt.plot(traj6[:,0], traj6[:,1], '*c', lw=2, alpha=0.2)

# plt.plot(traj7[:,0], traj7[:,1], '*r', lw=2, alpha=0.2)
# plt.plot(traj8[:,0], traj8[:,1], '*m', lw=2, alpha=0.2)


plt.show()

traj1 = calc_speed(traj1)
traj2 = calc_speed(traj2)
traj3 = calc_speed(traj3)
traj4 = calc_speed(traj4)

# traj5 = calc_speed(traj5)
# traj6 = calc_speed(traj6)
# traj7 = calc_speed(traj7)
# traj8 = calc_speed(traj8)

# traj1 = traj1[:len(traj)]
# traj2 = traj2[:len(traj)]
# traj3 = traj3[:len(new_center)]
# traj4 = traj4[:len(new_center)]

# traj5 = traj3[:len(new_center)]
# traj6 = traj4[:len(new_center)]
# traj7 = traj7[:len(new_center)]
# traj8 = traj8[:len(new_center)]

# traj1[:,4:6], traj1[:,6:8] = traj[:,4:6], traj[:,6:8]
# traj2[:,4:6], traj2[:,6:8] = traj[:,4:6], traj[:,6:8]
# traj3[:,4:6], traj3[:,6:8] = new_center[:,4:6], new_center[:,6:8]
# traj4[:,4:6], traj4[:,6:8] = new_center[:,4:6], new_center[:,6:8]


# traj5[:,4:6], traj5[:,6:8] = traj[:,4:6], traj[:,6:8]
# traj6[:,4:6], traj6[:,6:8] = traj[:,4:6], traj[:,6:8]
# traj7[:,4:6], traj7[:,6:8] = new_center[:,4:6], new_center[:,6:8]
# traj8[:,4:6], traj8[:,6:8] = new_center[:,4:6], new_center[:,6:8]

# for i in range(len(traj3)):
#     A = np.array( [ [traj3[i,4], traj3[i,5]],  [traj3[i,0], traj3[i,1]], [traj3[i,6], traj3[i,7]]  ])
#     plt.plot(A[:,0], A[:,1], 'r')
# traj3[:,4:6], traj3[:,6:8] = new_center[:len(traj),4:6], new_center[:len(traj),6:8]
# traj4[:,4:6], traj4[:,6:8] = new_center[:len(traj),4:6], new_center[:len(traj),6:8]


np.savetxt('./traj1_with_boundary.txt', traj1, delimiter=",")
np.savetxt('./traj2_with_boundary.txt', traj2, delimiter=",")
np.savetxt('./traj3_with_boundary.txt', traj3, delimiter=",")
np.savetxt('./traj4_with_boundary.txt', traj4, delimiter=",")
# np.savetxt('/home/hmcl/Desktop/F1tenth/traj5_with_boundary.txt', traj5, delimiter=",")
# np.savetxt('/home/hmcl/Desktop/F1tenth/traj6_with_boundary.txt', traj6, delimiter=",")
# np.savetxt('/home/hmcl/Desktop/F1tenth/traj7_with_boundary.txt', traj7, delimiter=",")
# np.savetxt('/home/hmcl/Desktop/F1tenth/traj8_with_boundary.txt', traj8, delimiter=",")

plt.show()