import random


import os
import math
import json
import errno
import shutil
import joblib
import random
import pyquaternion
import numpy as np
from visualization_msgs.msg import MarkerArray, Marker
import torch
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from scipy.signal import filtfilt, butter


def dist2d(point1, point2):
    """
    Euclidean distance between two points
    :param point1:
    :param point2:
    :return:
    """

    x1, y1 = point1[0:2]
    x2, y2 = point2[0:2]

    dist2 = (x1 - x2)**2 + (y1 - y2)**2

    return math.sqrt(dist2)

    
def b_to_g_rot(r,p,y): 
    # rot = torch.tensor([[torch.cos(p)*torch.cos(y), -1*torch.cos(p)*torch.sin(y), torch.sin(p)],
    #        [torch.cos(r)*torch.sin(y)+torch.cos(y)*torch.sin(r)*torch.sin(p), torch.cos(r)*torch.cos(y)-torch.sin(r)*torch.sin(p)*torch.sin(y), -torch.cos(p)*torch.sin(r)],
    #        [torch.sin(r)*torch.sin(y)-torch.cos(r)*torch.cos(y)*torch.sin(p), torch.cos(y)*torch.sin(r)+torch.cos(r)*torch.sin(p)*torch.sin(y), torch.cos(r)*torch.cos(p)]])
        
    row1 = torch.transpose(torch.stack([torch.cos(p)*torch.cos(y), -1*torch.cos(p)*torch.sin(y), torch.sin(p)]),0,1)
    row2 = torch.transpose(torch.stack([torch.cos(r)*torch.sin(y)+torch.cos(y)*torch.sin(r)*torch.sin(p), torch.cos(r)*torch.cos(y)-torch.sin(r)*torch.sin(p)*torch.sin(y), -torch.cos(p)*torch.sin(r)]),0,1)
    row3 = torch.transpose(torch.stack([torch.sin(r)*torch.sin(y)-torch.cos(r)*torch.cos(y)*torch.sin(p), torch.cos(y)*torch.sin(r)+torch.cos(r)*torch.sin(p)*torch.sin(y), torch.cos(r)*torch.cos(p)]),0,1)
    rot = torch.stack([row1,row2,row3],dim = 1)
    return rot


def wrap_to_pi(angle):
    """
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi-0.01:
        angle -= 2.0 * np.pi

    while angle < -np.pi+0.01:
        angle += 2.0 * np.pi

    return angle 


def angle_normalize(x):
    return (((x + math.pi) % (2 * math.pi)) - math.pi)

def wrap_to_pi_torch(angle):
    """
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    return (((angle + torch.pi) % (2 * torch.pi)) - torch.pi)
    
    
# def wrap_to_pi_torch(self, array):
#     array[array<-torch.pi] = array[array<-torch.pi] + 2*torch.pi
#     array[array>torch.pi] = array[array>torch.pi] - 2*torch.pi
#     return array 



def get_odom_euler(odom):    
    q = pyquaternion.Quaternion(w=odom.pose.pose.orientation.w, x=odom.pose.pose.orientation.x, y=odom.pose.pose.orientation.y, z=odom.pose.pose.orientation.z)
    yaw, pitch, roll = q.yaw_pitch_roll
    yaw = angle_normalize(yaw)
    return [roll, pitch, yaw]





def quaternion_to_euler(q):
    q = pyquaternion.Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
    yaw, pitch, roll = q.yaw_pitch_roll
    return [roll, pitch, yaw]


def unit_quat(q):
    """
    Normalizes a quaternion to be unit modulus.
    :param q: 4-dimensional numpy array or CasADi object
    :return: the unit quaternion in the same data format as the original one
    """

    # if isinstance(q, np.ndarray):
        # if (q == np.zeros(4)).all():
        #     q = np.array([1, 0, 0, 0])
    q_norm = np.sqrt(np.sum(q ** 2))
    # else:
    #     q_norm = cs.sqrt(cs.sumsqr(q))
    return 1 / q_norm * q

def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    
    return unit_quat(np.array([qw, qx, qy, qz]))



def q_to_rot_mat(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]    
    rot_mat = np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]])
    return rot_mat


def get_local_vel(odom, is_odom_local_frame = False):
    local_vel = np.array([0.0, 0.0, 0.0])
    if is_odom_local_frame is False: 
        # convert from global to local 
        q_tmp = np.array([odom.pose.pose.orientation.w,odom.pose.pose.orientation.x,odom.pose.pose.orientation.y,odom.pose.pose.orientation.z])
        euler = get_odom_euler(odom)
        rot_mat_ = q_to_rot_mat(q_tmp)
        inv_rot_mat_ = np.linalg.inv(rot_mat_)
        global_vel = np.array([odom.twist.twist.linear.x,odom.twist.twist.linear.y,odom.twist.twist.linear.z])
        local_vel = inv_rot_mat_.dot(global_vel)        
    else:
        local_vel[0] = odom.twist.twist.linear.x
        local_vel[1] = odom.twist.twist.linear.y
        local_vel[2] = odom.twist.twist.linear.z
    return local_vel 


def traj_to_markerArray(traj,color= [255,0,0]):

    marker_refs = MarkerArray() 
   
    for i in range(len(traj[:,0])):
        marker_ref = Marker()
        marker_ref.header.frame_id = "world"  
        marker_ref.ns = "ref_states"+str(i)
        marker_ref.id = i
        marker_ref.type = Marker.ARROW
        marker_ref.action = Marker.ADD                
        marker_ref.pose.position.x = traj[i,0] 
        marker_ref.pose.position.y = traj[i,1]              
        quat_tmp = euler_to_quaternion(0.0, 0.0, traj[i,2])     
        quat_tmp = unit_quat(quat_tmp)                 
        marker_ref.pose.orientation.w = quat_tmp[0]
        marker_ref.pose.orientation.x = quat_tmp[1]
        marker_ref.pose.orientation.y = quat_tmp[2]
        marker_ref.pose.orientation.z = quat_tmp[3]
        marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (color[0], color[1] , color[2])
        marker_ref.color.a = 1.0
        marker_ref.scale.x, marker_ref.scale.y, marker_ref.scale.z = (0.2, 0.2, 0.15)
        marker_refs.markers.append(marker_ref)
        

    return marker_refs

def predicted_distribution_traj_visualize(x_mean,x_var,y_mean,y_var,mean_predicted_state,color):
    marker_refs = MarkerArray() 
    for i in range(len(x_mean)):
        marker_ref = Marker()
        marker_ref.header.frame_id = "map"  
        marker_ref.ns = "gplogger_ref"+str(i)
        marker_ref.id = i
        marker_ref.type = Marker.SPHERE
        marker_ref.action = Marker.ADD                
        marker_ref.pose.position.x = x_mean[i]
        marker_ref.pose.position.y = y_mean[i]
        marker_ref.pose.position.z = mean_predicted_state[i,6]  
        quat_tmp = euler_to_quaternion(0.0, 0.0, mean_predicted_state[i,2])             
        marker_ref.pose.orientation.w = quat_tmp[0]
        marker_ref.pose.orientation.x = quat_tmp[1]
        marker_ref.pose.orientation.y = quat_tmp[2]
        marker_ref.pose.orientation.z = quat_tmp[3]
        marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (color[0], color[1], color[2])
        marker_ref.color.a = 0.5        
        # marker_ref.scale.x, marker_ref.scale.y, marker_ref.scale.z = (0.6, 0.4, 0.3)
        marker_ref.scale.x = 2*np.sqrt(x_var[i])
        marker_ref.scale.y = 2*np.sqrt(y_var[i])
        marker_ref.scale.z = 1
        marker_refs.markers.append(marker_ref)
        i+=1
    return marker_refs


def predicted_trj_visualize(predicted_state,color):        
    marker_refs = MarkerArray() 
    for i in range(len(predicted_state[:,0])):
        marker_ref = Marker()
        marker_ref.header.frame_id = "map"  
        marker_ref.ns = "gplogger_ref"+str(i)
        marker_ref.id = i
        marker_ref.type = Marker.ARROW
        marker_ref.action = Marker.ADD                
        marker_ref.pose.position.x = predicted_state[i,0] 
        marker_ref.pose.position.y = predicted_state[i,1]              
        marker_ref.pose.position.z = predicted_state[i,6]  
        quat_tmp = euler_to_quaternion(0.0, 0.0, predicted_state[i,2])     
        quat_tmp = unit_quat(quat_tmp)                 
        marker_ref.pose.orientation.w = quat_tmp[0]
        marker_ref.pose.orientation.x = quat_tmp[1]
        marker_ref.pose.orientation.y = quat_tmp[2]
        marker_ref.pose.orientation.z = quat_tmp[3]
        marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (color[0], color[1], color[2])
        marker_ref.color.a = 0.5        
        # marker_ref.scale.x, marker_ref.scale.y, marker_ref.scale.z = (0.6, 0.4, 0.3)
        marker_ref.scale.x = (i+1)/len(predicted_state[:,0])*0.1+0.1
        marker_ref.scale.y = (i+1)/len(predicted_state[:,0])*0.1+0.1
        marker_ref.scale.z = (i+1)/len(predicted_state[:,0])*0.1+0.1
        marker_refs.markers.append(marker_ref)
        i+=1
    return marker_refs


def ref_to_markerArray(traj):

    marker_refs = MarkerArray() 
    for i in range(len(traj[:,0])):
        marker_ref = Marker()
        marker_ref.header.frame_id = "map"  
        marker_ref.ns = "ref_states_"+str(i)
        marker_ref.id = i
        marker_ref.type = Marker.SPHERE
        marker_ref.action = Marker.ADD                
        marker_ref.pose.position.x = traj[i,0] 
        marker_ref.pose.position.y = traj[i,1]              
        marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (0, 0, 255)
        marker_ref.color.a = 0.5
        marker_ref.scale.x, marker_ref.scale.y, marker_ref.scale.z = (0.1, 0.1, 0.1)
        marker_refs.markers.append(marker_ref)
        

    return marker_refs



def multi_predicted_distribution_traj_visualize(x_mean_set,x_var_set,y_mean_set,y_var_set,mean_predicted_state_set):
    marker_refs = MarkerArray() 
    for j in range(len(x_mean_set)):
        for i in range(len(x_mean_set[j])):
            marker_ref = Marker()
            marker_ref.header.frame_id = "map"  
            marker_ref.ns = "gplogger_ref"+str(i)+str(j)
            marker_ref.id = j*len(x_mean_set[j])+i
            marker_ref.type = Marker.SPHERE
            marker_ref.action = Marker.ADD                
            marker_ref.pose.position.x = x_mean_set[j][i]
            marker_ref.pose.position.y =  y_mean_set[j][i]
            marker_ref.pose.position.z = mean_predicted_state_set[j,i,6]  
            quat_tmp = euler_to_quaternion(0.0, 0.0, mean_predicted_state_set[j,i,2])             
            marker_ref.pose.orientation.w = quat_tmp[0]
            marker_ref.pose.orientation.x = quat_tmp[1]
            marker_ref.pose.orientation.y = quat_tmp[2]
            marker_ref.pose.orientation.z = quat_tmp[3]
            marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (246, 229, 100 + 155/(len(x_mean_set)+0.01)*j)
            marker_ref.color.a = 0.5        
            # marker_ref.scale.x, marker_ref.scale.y, marker_ref.scale.z = (0.6, 0.4, 0.3)
            marker_ref.scale.x = 0.1 #2*np.sqrt(x_var_set[j][i])
            marker_ref.scale.y = 0.1 #2*np.sqrt(y_var_set[j][i])
            marker_ref.scale.z = 0.1
            marker_refs.markers.append(marker_ref)
            i+=1
    return marker_refs


def mean_multi_predicted_distribution_traj_visualize(mean_predicted_state_set):
    marker_refs = MarkerArray() 
    for j in range(mean_predicted_state_set.shape[0]):        
        for i in range(mean_predicted_state_set.shape[1]):
            marker_ref = Marker()
            marker_ref.header.frame_id = "map"  
            marker_ref.ns = "mean_ref"+str(i)+str(j)
            marker_ref.id = j*mean_predicted_state_set.shape[1]+i
            marker_ref.type = Marker.SPHERE
            marker_ref.action = Marker.ADD                
            marker_ref.pose.position.x = mean_predicted_state_set[j,i,0] 
            marker_ref.pose.position.y = mean_predicted_state_set[j,i,1]              
            marker_ref.pose.position.z = mean_predicted_state_set[j,i,6]  
            quat_tmp = euler_to_quaternion(0.0, 0.0, mean_predicted_state_set[j,i,2])     
            quat_tmp = unit_quat(quat_tmp)                 
            marker_ref.pose.orientation.w = quat_tmp[0]
            marker_ref.pose.orientation.x = quat_tmp[1]
            marker_ref.pose.orientation.y = quat_tmp[2]
            marker_ref.pose.orientation.z = quat_tmp[3]
            marker_ref.color.r = 0
            marker_ref.color.g = 255
            marker_ref.color.b = 0 
            marker_ref.color.a = 0.5    
            # marker_ref.scale.x, marker_ref.scale.y, marker_ref.scale.z = (0.6, 0.4, 0.3)
            marker_ref.scale.x = 0.1
            marker_ref.scale.y = 0.1
            marker_ref.scale.z = 0.1
            marker_refs.markers.append(marker_ref)
            i+=1
        

    return marker_refs



def nominal_multi_predicted_distribution_traj_visualize(mean_predicted_state_set):
    marker_refs = MarkerArray() 
    for j in range(mean_predicted_state_set.shape[0]):        
        for i in range(mean_predicted_state_set.shape[1]):
            marker_ref = Marker()
            marker_ref.header.frame_id = "map"  
            marker_ref.ns = "mean_ref"+str(i)+str(j)
            marker_ref.id = j*mean_predicted_state_set.shape[1]+i
            marker_ref.type = Marker.SPHERE
            marker_ref.action = Marker.ADD                
            marker_ref.pose.position.x = mean_predicted_state_set[j,i,0] 
            marker_ref.pose.position.y = mean_predicted_state_set[j,i,1]              
            marker_ref.pose.position.z = mean_predicted_state_set[j,i,6]  
            quat_tmp = euler_to_quaternion(0.0, 0.0, mean_predicted_state_set[j,i,2])     
            quat_tmp = unit_quat(quat_tmp)                 
            marker_ref.pose.orientation.w = quat_tmp[0]
            marker_ref.pose.orientation.x = quat_tmp[1]
            marker_ref.pose.orientation.y = quat_tmp[2]
            marker_ref.pose.orientation.z = quat_tmp[3]
            marker_ref.color.r = 255
            marker_ref.color.g = 0
            marker_ref.color.b = 0 
            marker_ref.color.a = 1.0    
            # marker_ref.scale.x, marker_ref.scale.y, marker_ref.scale.z = (0.6, 0.4, 0.3)
            marker_ref.scale.x = 0.05
            marker_ref.scale.y = 0.05
            marker_ref.scale.z = 0.05
            marker_refs.markers.append(marker_ref)
            i+=1
        

    return marker_refs

def dist3d(point1, point2):
    """
    Euclidean distance between two points 3D
    :param point1:
    :param point2:
    :return:
    """

    x1, y1, z1 = point1[0:3]
    x2, y2, z2 = point2[0:3]

    dist3d = (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2

    return math.sqrt(dist3d)

def gaussianKN2D(rl=4,cl=5, rsig=1.,csig=2.):
    """\
    creates gaussian kernel with side length `rl,cl` and a sigma of `rsig,csig`
    """
    rx = np.linspace(-(rl - 1) / 2., (rl - 1) / 2., rl)
    cx = np.linspace(-(cl - 1) / 2., (cl - 1) / 2., cl)
    gauss_x = np.exp(-0.5 * np.square(rx) / np.square(rsig))
    gauss_y = np.exp(-0.5 * np.square(cx) / np.square(csig))
    kernel = np.outer(gauss_x, gauss_y)
    return kernel / (np.sum(kernel)+1e-8)


# def tmp_dump1():
        ## #########
        #   || idx_l_top          ...   idx_r_min     ...    idx_r_top     || 
        #   ||   .                 .        .          .          .        ||        
        #   || idx_c_max           .       idx         .     idx_c_min     || 
        #   ||   .                 .        .          .          .        ||
        #   || idx_l_bottom       ...   idx_r_max     ...    idx_r_bottom  || 
        
        
        # idx_r_max = idx + half_row_map_size*self.c_size
        # markersidx_l_top = idx_r_min + half_col_map_size
        # idx_r_top = idx_r_min - half_col_map_size
        # idx_l_top, idx_r_top = np.clip([idx_l_top, idx_r_top], self.c_size*(idx_r_min%self.c_size), self.c_size*(idx_r_min%self.c_size)+ self.c_size-1)

        # idx_l_bottom = idx_r_max + half_col_map_size
        # idx_r_bottom = idx_r_max - half_col_map_size
        # idx_l_bottom, idx_r_bottom = np.clip([idx_l_bottom, idx_r_bottom], self.c_size*(idx_r_max%self.c_size), self.c_size*(idx_r_max%self.c_size)+ self.c_size-1)

        # # idx_c_min = idx - half_col_map_size
        # # idx_c_max = idx + half_col_map_size
        # # idx_c_min, idx_c_max = np.clip([idx_c_min, idx_c_max], self.c_size*(idx%self.c_size), self.c_size*(idx%self.c_size)+ self.c_size-1)
        
        # # map2d ###################33
        # #   || idx_r_top          ...   idx_r_min     ...    idx_l_top     || 
        # #   ||   .                 .        .          .          .        ||        
        # #   || idx_c_min           .       idx         .       idx_c_max   || 
        # #   ||   .                 .        .          .          .        ||
        # #   || idx_r_bottom       ...   idx_r_max     ...     idx_l_bottom || 

def torch_path_to_marker(path):
    path_numpy = path.cpu().numpy()
    marker_refs = MarkerArray() 
    marker_ref = Marker()
    marker_ref.header.frame_id = "map"  
    marker_ref.ns = "mppi_ref"
    marker_ref.id = 0
    marker_ref.type = Marker.LINE_STRIP
    marker_ref.action = Marker.ADD     
    marker_ref.scale.x = 0.1 
    for i in range(len(path_numpy[0,:])):                
        point_msg = Point()
        point_msg.x = path_numpy[0,i] 
        point_msg.y = path_numpy[1,i]              
        point_msg.z = path_numpy[3,i] 
        
        color_msg = ColorRGBA()
        color_msg.r = 0.0
        color_msg.g = 0.0
        color_msg.b = 1.0
        color_msg.a = 1.0
        marker_ref.points.append(point_msg)
        marker_ref.colors.append(color_msg)    
    marker_refs.markers.append(marker_ref)
    return marker_refs


def bound_angle_within_pi(angle):
	return (angle + np.pi) % (2.0 * np.pi) - np.pi # https://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap


def fix_angle_reference(angle_ref, angle_init):
	# This function returns a "smoothened" angle_ref wrt angle_init so there are no jumps.
	diff_angle = angle_ref - angle_init
	diff_angle = bound_angle_within_pi(diff_angle)
	diff_angle = np.unwrap(diff_angle) # removes jumps greater than pi
	return angle_init + diff_angle 


def compute_curvature(cdists, psis):
	# This function estimates curvature using finite differences (curv = dpsi/dcdist).
	diff_dists = np.diff(cdists)
	diff_psis  = np.diff(np.unwrap(psis))

	assert np.max( np.abs(diff_psis) )  < np.pi, "Detected a jump in the angle difference."

	curv_raw = diff_psis / np.maximum(diff_dists, 0.1) # use diff_dists where greater than 10 cm	
	curv_raw = np.insert(curv_raw, len(curv_raw), curv_raw[-1]) # curvature at last waypoint

	# Curvature Filtering: (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html)
	curv_filt =  filtfilt(np.ones((11,))/11, 1, curv_raw) 
	return curv_filt

def get_cdist_and_psis(xs,ys):
    cdists =[]
    psis = []
    cdists.append(0.0)  

    for i in range (len(xs)-1):    
        dist = math.sqrt( (xs[i+1] - xs[i])**2 + (ys[i+1] - ys[i])**2 ) + cdists[-1]
        cdists.append(dist)    
        psi = math.atan2((ys[i+1] - ys[i]),(xs[i+1] - xs[i]))
        psis.append(psi)
    
    psis.append(psis[-1])
    psis_unwrap = fix_angle_reference(np.array(psis).squeeze(), 0.0)
    # # define filter 
    # order = 2
    # cutoff = 5
    # fs = 1000
    # b,a = butter(order,cutoff/(fs/2),'low')
    
    unwraped_psis_filted = filtfilt(np.ones((11,))/11, 1, psis_unwrap) # curvature filter suggested by Jinkkwon Kim.
    
    wraped_psis_filted = bound_angle_within_pi(unwraped_psis_filted)
    return np.array(cdists).squeeze(), wraped_psis_filted


def get_psi(xs,ys):        
    psis = []
    for i in range (len(xs)-1):        
        psi = math.atan2((ys[i+1] - ys[i]),(xs[i+1] - xs[i]))
        psis.append(psi)
    
    psis.append(psis[-1])
    psis_unwrap = fix_angle_reference(np.array(psis).squeeze(), 0.0)    
    unwraped_psis_filted = filtfilt(np.ones((11,))/11, 1, psis_unwrap) # curvature filter suggested by Jinkkwon Kim.
    wraped_psis_filted = bound_angle_within_pi(unwraped_psis_filted)
    return wraped_psis_filted
    
def construct_frenet_centerline(xs,ys):
    s, psi = get_cdist_and_psis(xs,ys)    
    curvature = compute_curvature(s,psi)
    return np.transpose(np.vstack([s,psi,curvature]))

def global_to_frenet(x,y,psi, centerline_global,center_frenet):
    # centerline_global -> x_global, y_global
    # center_frenet -> s, s_psi, curvature
    closest_index = np.argmin(np.linalg.norm(centerline_global[:,0:2]-[x,y],axis=1))
    
    psi_centerline =center_frenet[closest_index,1]
    rot_global_to_frenet = np.array([[ np.cos(psi_centerline), np.sin(psi_centerline)], \
                                    [-np.sin(psi_centerline), np.cos(psi_centerline)]])
    error_xy = [x,y] - centerline_global[closest_index,:2] # xy deviation (global frame)
    error_frenet = np.dot(rot_global_to_frenet, error_xy[:]) # e_s, e_y deviation (Frenet frame)

    c_s = center_frenet[closest_index,0]
    c_e_y = error_frenet[1]
    c_e_psi = bound_angle_within_pi(psi-psi_centerline)
    c_curvature = center_frenet[closest_index,2]

    return np.array([c_s,c_e_y,c_e_psi, c_curvature])

def global_and_frenet_centerline(xs,ys):
    s, psi = get_cdist_and_psis(xs,ys)    
    curvature = compute_curvature(s,psi)
    return np.transpose(np.vstack([xs,ys,psi,s,curvature]))


def wrap_s(s_,track_length):
    while(np.min(s_) < 0):        
        s_[s_ < 0] += track_length    
    while(np.max(s_) >= track_length):        
        s_[s_ >= track_length] -= track_length    
    return s_


def torch_unwrap_s(s_,s_init,track_length):
    below_s_init_idx = (s_ < s_init).nonzero()
    s_[below_s_init_idx] +=track_length
    return s_


def torch_wrap_s(s_,track_length):
    # while(torch.min(s_) < 0):
    below_zero_idx = (s_ < 0).nonzero()
    s_[below_zero_idx] += track_length
    
    # while(torch.max(s_) >= track_length):
    above_tracklength_idx = (s_ >= track_length).nonzero()
    s_[above_tracklength_idx] -= track_length
    
    return s_

def torch_get_cuvature(s_, center_frenet,approx_range = None):    
    # center_frenet --> s(0), psi(1), curvature(2)
    if approx_range is not None:
        ref_ = center_frenet[approx_range[0]:approx_range[1],0]
    else:
        ref_ = center_frenet[:,0]
    s_ = torch_wrap_s(s_,center_frenet[-1,0])
    if s_.ndim > 1:
        s_s = s_.repeat(1,ref_.shape[0]).view(-1,s_.shape[0]).float()
        ref_s = ref_.repeat(s_.shape[0],1).view(-1,ref_s.shape[0]).float()
    else:        
        s_s = torch.transpose(s_.repeat(1,ref_.shape[0]).view(-1,2), 0,1).reshape(-1,1).float().squeeze()
        ref_s = ref_.repeat(1,s_.shape[0]).float()

    norm_vecs = torch.linalg.norm(s_s-ref_s,dim=0)
    row_norm_vecs = norm_vecs.view(-1,ref_.shape[0])
    min_index = row_norm_vecs.argmin(dim=1)
    curvatures = center_frenet[min_index,2]
    return curvatures

def torch_fren_to_global(fren_, center_global, center_frenet):
    s_ = fren_[:,0].squeeze()
    if not torch.is_tensor(center_global):
        center_global = torch.tensor(center_global).to(device="cuda")
    if not torch.is_tensor(center_frenet):
        center_frenet = torch.tensor(center_frenet).to(device="cuda")
    ref_ = center_frenet[:,0]
    s_ = torch_wrap_s(s_,center_frenet[-1,0])
    s_s = torch.transpose(s_.repeat(1,ref_.shape[0]).view(-1,2), 0,1).reshape(-1,1).float().squeeze()
    ref_s = ref_.repeat(s_.shape[0]).float()

    norm_vecs = torch.abs(s_s-ref_s)
    row_norm_vecs = norm_vecs.view(-1,ref_.shape[0])
    min_index = row_norm_vecs.argmin(dim=1)

    ey = fren_[:,1]
    epsi = fren_[:,2]
    # centerline_global -> x_global, y_global
    # center_frenet -> s, s_psi, curvature
    x_s = center_global[min_index,0]
    y_s = center_global[min_index,1]
    s_psi = center_frenet[min_index,2]    

    x = x_s +  ey*torch.cos(s_psi + torch.pi/2)
    y = y_s +  ey*torch.sin(s_psi + torch.pi/2)
    psi = wrap_to_pi_torch(s_psi+epsi)
    return x, y, psi
    


def local_to_global(s,ey,epsi,center_global,center_frenet):
    # centerline_global -> x_global, y_global
    # center_frenet -> s, s_psi, curvature
    x_s = np.interp(s,center_frenet[:,0],center_global[:,0])
    y_s = np.interp(s,center_frenet[:,0],center_global[:,1])
    s_psi = np.interp(s,center_frenet[:,0],center_frenet[:,1])

    x = x_s +  ey*np.cos(s_psi + np.pi/2)
    y = y_s +  ey*np.sin(s_psi + np.pi/2)
    psi = wrap_to_pi(s_psi+epsi)    

    return np.array([x,y,psi])

def fren_to_global(fren, center_global,center_frenet):
    s = fren[0]
    ey = fren[1]
    epsi = fren[2]
    return local_to_global(s,ey,epsi,center_global,center_frenet)

def torch_local_to_global(s_, ey_, epsi_, center_global, center_frenet):
    return 