
# from asyncio import to_thread
import numpy as np
import matplotlib.pyplot as plt
import math
import os 

from utils.pycubicspline import Spline2D

from matplotlib.widgets import Cursor



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


def generate_new_traj(traj,psi, node_pts):

    N = 2
    M = len(traj)//N
    new_traj1 = np.empty((M,8))
    target = 0
    
    pass_target = False
    final_node = False
    pass_start = False
    
    for i in range(M): 

        if ~final_node:
            current_target2 = node_pts[target+1][0]
            current_target1 = node_pts[target][0]
            
            if current_target1 >= current_target2:
                current_target2 = M-1
                pass_start = True
        

        if i*N == len(traj)-1:
            psi[i*N] = psi[-2]

        if not pass_target or final_node:
            Xboundleft=traj[i*N,0] 
            Yboundleft=traj[i*N,1] 
            Li = 0
        else:
            Li = node_pts[target][1]   
            Xboundleft=traj[i*N,0] - Li*np.sin(psi[i*N])
            Yboundleft=traj[i*N,1] + Li*np.cos(psi[i*N])

        if pass_target and i*N >= current_target2:
            pass_target = False
            if target+2 == len(node_pts):
                final_node = True
            else:
                target += 2
        elif i*N >= current_target1:
            pass_target = True
            
        new_traj1[i,0] = Xboundleft
        new_traj1[i,1] = Yboundleft
        new_traj1[i,2] = traj[i*N,2]
        new_traj1[i,3] = traj[i*N,3]
        new_traj1[i,6] = traj[i*N,6]
        new_traj1[i,7] = Li

        plt.plot(Xboundleft,Yboundleft, 'r*')
    
    if pass_start:
        Li = node_pts[-1][1]

        for i in range(node_pts[-1][0]//N):
            new_traj1[i,0]=traj[i*N,0] - Li*np.sin(psi[i*N])
            new_traj1[i,1]=traj[i*N,1] + Li*np.cos(psi[i*N])
            
            plt.plot(new_traj1[i,0],new_traj1[i,1], 'g*')
        
    x,y = fit_cubic_splines(new_traj1[:,0],new_traj1[:,1],M)
    new_traj1[:,0] = x
    new_traj1[:,1] = y
    plt.plot(new_traj1[:,0],new_traj1[:,1], linewidth=2)
    
    
    return new_traj1

    new_traj = np.empty((len(traj),4))
    params = F110()
    time, speed = calcMinimumTimeSpeed(traj[:,0], traj[:,1], **params)
    speed = ModifyVelocity(speed)
    curv = calc_curv(traj)
    new_traj[:,:2] = traj[:,:2]
    new_traj[:,2] = speed
    new_traj[:,3] = curv

    return new_traj

def fit_cubic_splines(x, y, n_samples):
    sp = Spline2D(x, y)
    s = np.linspace(0, sp.s[-1]-1e-2, n_samples)
    x, y = [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        x.append(ix)
        y.append(iy)
    return x, y





dir = os.path.expanduser('~') + '/F1tenth_track/'
CenterFile = "result/center_traj_with_boundary.txt"
InnerBoundaryFile = "result/innerwall.txt"
OuterBoundaryFile = "result/outerwall.txt"
OptimizeFile = "result/opt_traj_with_boundary.txt"

traj = np.loadtxt(OptimizeFile, delimiter=",", dtype = float)
inner = np.loadtxt(InnerBoundaryFile, delimiter=",", dtype = float)
outer = np.loadtxt(OuterBoundaryFile, delimiter=",", dtype = float)
center = np.loadtxt(CenterFile, delimiter=",", dtype = float)


## Match the start line with centerline 
# dx = 0.5
# idx = -1
# pt = [traj[0,0], traj[0,1]]
# idx =np.argmin(abs(np.linalg.norm(center[:,:2]-pt,axis=1)))


# new_center = np.empty((len(center),6))
# new_center[:len(center)-idx] = center[idx:]
# new_center[len(center)-idx:] = center[:idx]


psi = calc_psi(traj)

#####
node_ids = []
node_pts_w_dis_set = []
dis_arr = []

while(1):
        node_pts = []

        def add_point(event):
            if event.inaxes != ax:
                return
            
            if event.button == 1:
                x = event.xdata
                y = event.ydata

                old_dis = 10
                n_id = 0
                for i in range(len(traj)):
                    dis = math.sqrt( pow(traj[i,0] - x,2 ) + pow(traj[i,1] - y,2 ))
                    if dis < old_dis:
                        n_id = i
                        old_dis = dis
            
                plt.plot(x, y, '*r', lw=3)
                plt.plot(traj[n_id,0], traj[n_id,1], '*g', lw=3)
                plt.draw()
                
                node_pts.append(n_id)
                print(node_pts)
                
            

        fig = plt.figure()
        ax = fig.add_subplot(111, facecolor='#FFFFCC')
        plt.plot(inner[:,0], inner[:,1], '-k')
        plt.plot(outer[:,0], outer[:,1],'-k')
        plt.plot(traj[:,0], traj[:,1],'-k')
        plt.text(traj[1,0], traj[1,1],
            "START",
            horizontalalignment ="center", 
            verticalalignment ="center", 
            wrap = True, fontsize = 12, 
            color ="blue")
        plt.text(traj[-1,0], traj[-1,1],
            "END",
            horizontalalignment ="center", 
            verticalalignment ="center", 
            wrap = True, fontsize = 12, 
            color ="red")


        cursor = Cursor(ax, useblit=True, color='red', linewidth=2)
        cid = plt.connect('button_press_event', add_point)

        plt.show()

        Dis = float(input("Rollout Distance is (+):Left (-): Right:: "))
        print("(a): add node, (n): new trajectory, (s): same trajectory with different rollout (.): end")
        cmd = input("enter::")


        for i in range(len(node_pts)):
            node_pts_w_dis_set.append([node_pts[i], Dis])

        if cmd == 'n':
            dis_arr.append(Dis)
            node_ids.append(node_pts_w_dis_set)
            node_pts_w_dis_set = []
            
            continue 
        elif cmd == 's':
            node_ids.append(node_pts_w_dis_set)
            dis_arr.append(Dis)
            Dis = float(input("Rollout Distance is (+):Right (-): Left:: "))
            dis_arr.append(Dis)
            node_pts_w_dis_set2 = [[0 for col in range(len(node_pts_w_dis_set[0]))] for row in range(len(node_pts_w_dis_set))]
            for j in range(len(node_pts_w_dis_set)):
                node_pts_w_dis_set2[j][0] = node_pts_w_dis_set[j][0]
                node_pts_w_dis_set2[j][1] = Dis
            node_ids.append(node_pts_w_dis_set2)
            node_pts_w_dis_set = []
            cmd = input("(c): continue, (.):end")
            if cmd != 'c':
                break
        elif cmd == 'a':
            continue 
        else:
            node_ids.append(node_pts_w_dis_set)
            break

print(node_ids)

new_node_ids = np.argsort(dis_arr)

## Arrage to biggest


## Generate New Trajectory


for j in range(len(node_ids)):

    traj_ = generate_new_traj(traj, psi, node_ids[new_node_ids[j]])

    plt.plot(traj[:,0], traj[:,1])
    np.savetxt(dir + 'result/traj'+str(j+1)+'.txt', traj_, delimiter=",")
    plt.draw()

plt.plot(inner[:,0], inner[:,1], '-k')
plt.plot(outer[:,0], outer[:,1],'-k')
plt.plot(center[:,0], center[:,1],'-k')
plt.show()