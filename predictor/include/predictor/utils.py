
"""   
 Software License Agreement (BSD License)
 Copyright (c) 2022 Ulsan National Institute of Science and Technology (UNIST)
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************** 
  @author: Hojin Lee <hojinlee@unist.ac.kr>
  @date: September 10, 2022
  @copyright 2022 Ulsan National Institute of Science and Technology (UNIST)
  @brief: torch version of util functions
"""
#!/usr/bin/env python3
import tf 
import array
import rospy
import math
import pyquaternion
import numpy as np
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import torch
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from predictor.common.tracks.radius_arclength_track import RadiusArclengthTrack
from predictor.common.pytypes import VehicleState, VehiclePrediction
from typing import List
from hmcl_msgs.msg import VehiclePredictionROS

def shift_in_local_x(pose_msg: PoseStamped, dist = -0.13):
        # Convert orientation to a rotation matrix
    position = pose_msg.pose.position
    orientation = pose_msg.pose.orientation
    
    # orientation_q = pose.pose.orientation    
    # quat = [orientation_q.w, orientation_q.x, orientation_q.y, orientation_q.z]
    # (cur_roll, cur_pitch, cur_yaw) = quaternion_to_euler (quat)
    # cur_yaw = wrap_to_pi(cur_yaw)

    quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
    rotation_matrix = tf.transformations.quaternion_matrix(quaternion)[:3, :3]

    # Create a translation vector representing 1 meter displacement in the local coordinate system
    local_translation = np.array([dist, 0., 0.])  # Adjust the values based on your desired displacement

    # Transform the translation vector from local to global coordinate system
    global_translation = rotation_matrix.dot(local_translation)

    # Add the transformed translation vector to the original position
    new_position = np.array([position.x, position.y, position.z]) + global_translation

    # Update the position values in the PoseStamped message
    pose_msg.pose.position.x = new_position[0]
    pose_msg.pose.position.y = new_position[1]


def pose_to_vehicleState(track: RadiusArclengthTrack, state : VehicleState,pose : PoseStamped):
    state.x.x = pose.pose.position.x
    state.x.y = pose.pose.position.y
    orientation_q = pose.pose.orientation    
    quat = [orientation_q.w, orientation_q.x, orientation_q.y, orientation_q.z]
    (cur_roll, cur_pitch, cur_yaw) = quaternion_to_euler (quat)
    state.e.psi = cur_yaw
    xy_coord = (state.x.x, state.x.y, state.e.psi)
    cl_coord = track.global_to_local(xy_coord)
    if cl_coord is None:
        print('cl_coord is none')
        return
    state.t = pose.header.stamp.to_sec()
    state.p.s = cl_coord[0]
    state.p.x_tran = cl_coord[1]
    state.p.e_psi = cl_coord[2]
    
def odom_to_vehicleState(state:VehicleState, odom: Odometry):
    
    local_vel = get_local_vel(odom, is_odom_local_frame = False)
    if local_vel is None: 
        return 
    
    # local_vel[0] = max(0.2,local_vel[0]) ## limit the minimum velocity 
    if abs(local_vel[0]) > 0.0:
        local_vel[0] = max(0.5,local_vel[0]) ## limit the minimum velocity 
    else:
        local_vel[0] = min(-0.5,local_vel[0]) ## limit the minimum velocity 
    state.v.v_long = local_vel[0]
    state.v.v_tran = local_vel[1]
    state.w.w_psi = odom.twist.twist.angular.z


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
    
    



def get_odom_euler(odom):    
    q = pyquaternion.Quaternion(w=odom.pose.pose.orientation.w, x=odom.pose.pose.orientation.x, y=odom.pose.pose.orientation.y, z=odom.pose.pose.orientation.z)
    yaw, pitch, roll = q.yaw_pitch_roll
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


def traj_to_markerArray(traj):

    marker_refs = MarkerArray() 
    for i in range(len(traj[:,0])):
        marker_ref = Marker()
        marker_ref.header.frame_id = "map"  
        marker_ref.ns = "ref_states"+str(i)
        marker_ref.id = i
        marker_ref.type = Marker.ARROW
        marker_ref.action = Marker.ADD                
        marker_ref.pose.position.x = traj[i,0] 
        marker_ref.pose.position.y = traj[i,1]              
        quat_tmp = euler_to_quaternion(0.0, 0.0, traj[i,3])     
        quat_tmp = unit_quat(quat_tmp)                 
        marker_ref.pose.orientation.w = quat_tmp[0]
        marker_ref.pose.orientation.x = quat_tmp[1]
        marker_ref.pose.orientation.y = quat_tmp[2]
        marker_ref.pose.orientation.z = quat_tmp[3]
        marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (0, 255, 0)
        marker_ref.color.a = 0.2
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
    """
    creates gaussian kernel with side length `rl,cl` and a sigma of `rsig,csig`
    """
    rx = np.linspace(-(rl - 1) / 2., (rl - 1) / 2., rl)
    cx = np.linspace(-(cl - 1) / 2., (cl - 1) / 2., cl)
    gauss_x = np.exp(-0.5 * np.square(rx) / np.square(rsig))
    gauss_y = np.exp(-0.5 * np.square(cx) / np.square(csig))
    kernel = np.outer(gauss_x, gauss_y)
    return kernel / (np.sum(kernel)+1e-8)



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

def fill_global_info(track,pred):
    if pred.s is not None and len(pred.s) > 0 :
        pred.x = np.zeros(len(pred.s))
        pred.y = np.zeros(len(pred.s))
        pred.psi = np.zeros(len(pred.s))
        for i in range(len(pred.s)):
            cl_coord = [pred.s[i], pred.x_tran[i], pred.e_psi[i]]
            gl_coord = track.local_to_global(cl_coord)
            pred.x[i] = gl_coord[0]
            pred.y[i] = gl_coord[1]
            pred.psi[i] = gl_coord[2]


def prediction_to_marker(predictions):
    
    pred_path_marker_array = MarkerArray()
    if predictions is None or predictions.x is None:
        return pred_path_marker_array
    if len(predictions.x) <= 0:
        return pred_path_marker_array
    for i in range(len(predictions.x)):
  
        marker_ref = Marker()
        marker_ref.header.frame_id = "map"  
        marker_ref.header.stamp = rospy.Time.now()
        marker_ref.ns = "pred"
        marker_ref.id = i
        marker_ref.type = Marker.SPHERE
        marker_ref.action = Marker.ADD                
        marker_ref.pose.position.x = predictions.x[i]
        marker_ref.pose.position.y = predictions.y[i]
        marker_ref.pose.position.z = 0.0        
        marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (1.0, i/10.0, 0.0)
        marker_ref.color.a = 0.2     
        marker_ref.lifetime = rospy.Duration(0.2)
        # marker_ref.scale.x, marker_ref.scale.y, marker_ref.scale.z = (0.6, 0.4, 0.3)
        scale = 2
        x_cov = max(predictions.xy_cov[i][0,0],0.00001)
        y_cov = max(predictions.xy_cov[i][1,1],0.00001)
        marker_ref.scale.x = 2*np.sqrt(x_cov)*scale
        marker_ref.scale.y = 2*np.sqrt(y_cov)*scale
        marker_ref.scale.z = 0.1
        pred_path_marker_array.markers.append(marker_ref)
        
    return pred_path_marker_array



def prediction_to_rosmsg(vehicle_prediction_obj: VehiclePrediction):
    ros_msg = VehiclePredictionROS()
    ros_msg.header.stamp= rospy.Time.now()
    ros_msg.header.frame_id = "map"
    # Assign values from the VehiclePrediction object to the ROS message
    ros_msg.t = vehicle_prediction_obj.t
    
    if vehicle_prediction_obj.x is not None:
        ros_msg.x = array.array('f', vehicle_prediction_obj.x)
    if vehicle_prediction_obj.y is not None:
        ros_msg.y = array.array('f', vehicle_prediction_obj.y)
    if vehicle_prediction_obj.v_x is not None:
        ros_msg.v_x = array.array('f', vehicle_prediction_obj.v_x)
    if vehicle_prediction_obj.v_y is not None:
        ros_msg.v_y = array.array('f', vehicle_prediction_obj.v_y)
    if vehicle_prediction_obj.a_x is not None:
        ros_msg.a_x = array.array('f', vehicle_prediction_obj.a_x)
    if vehicle_prediction_obj.a_y is not None:
        ros_msg.a_y = array.array('f', vehicle_prediction_obj.a_y)
    if vehicle_prediction_obj.psi is not None:
        ros_msg.psi = array.array('f', vehicle_prediction_obj.psi)
    if vehicle_prediction_obj.psidot is not None:
        ros_msg.psidot = array.array('f', vehicle_prediction_obj.psidot)
    if vehicle_prediction_obj.v_long is not None:
        ros_msg.v_long = array.array('f', vehicle_prediction_obj.v_long)
    if vehicle_prediction_obj.v_tran is not None:
        ros_msg.v_tran = array.array('f', vehicle_prediction_obj.v_tran)
    if vehicle_prediction_obj.a_long is not None:
        ros_msg.a_long = array.array('f', vehicle_prediction_obj.a_long)
    if vehicle_prediction_obj.a_tran is not None:
        ros_msg.a_tran = array.array('f', vehicle_prediction_obj.a_tran)
    if vehicle_prediction_obj.e_psi is not None:
        ros_msg.e_psi = array.array('f', vehicle_prediction_obj.e_psi)
    if vehicle_prediction_obj.s is not None:
        ros_msg.s = array.array('f', vehicle_prediction_obj.s)
    if vehicle_prediction_obj.x_tran is not None:
        ros_msg.x_tran = array.array('f', vehicle_prediction_obj.x_tran)
    if vehicle_prediction_obj.u_a is not None:
        ros_msg.u_a = array.array('f', vehicle_prediction_obj.u_a)
    if vehicle_prediction_obj.u_steer is not None:
        ros_msg.u_steer = array.array('f', vehicle_prediction_obj.u_steer)

    if vehicle_prediction_obj.lap_num is not None:
        ros_msg.lap_num = int(vehicle_prediction_obj.lap_num)
    if vehicle_prediction_obj.sey_cov is not None:             
        ros_msg.sey_cov = vehicle_prediction_obj.sey_cov.tolist()
    if vehicle_prediction_obj.xy_cov is not None:            
        xy_cov_1d = np.array(vehicle_prediction_obj.xy_cov).reshape(-1)        
        ros_msg.xy_cov = xy_cov_1d
    
    return ros_msg  


def rosmsg_to_prediction(ros_msg: VehiclePredictionROS):
    vehicle_prediction_obj = VehiclePrediction()

    # Assign values from the ROS message to the VehiclePrediction object
    vehicle_prediction_obj.t = ros_msg.t
    vehicle_prediction_obj.x = array.array('f', ros_msg.x)
    vehicle_prediction_obj.y = array.array('f', ros_msg.y)
    vehicle_prediction_obj.v_x = array.array('f', ros_msg.v_x)
    vehicle_prediction_obj.v_y = array.array('f', ros_msg.v_y)
    vehicle_prediction_obj.a_x = array.array('f', ros_msg.a_x)
    vehicle_prediction_obj.a_y = array.array('f', ros_msg.a_y)
    vehicle_prediction_obj.psi = array.array('f', ros_msg.psi)
    vehicle_prediction_obj.psidot = array.array('f', ros_msg.psidot)
    vehicle_prediction_obj.v_long = array.array('f', ros_msg.v_long)
    vehicle_prediction_obj.v_tran = array.array('f', ros_msg.v_tran)
    vehicle_prediction_obj.a_long = array.array('f', ros_msg.a_long)
    vehicle_prediction_obj.a_tran = array.array('f', ros_msg.a_tran)
    vehicle_prediction_obj.e_psi = array.array('f', ros_msg.e_psi)
    vehicle_prediction_obj.s = array.array('f', ros_msg.s)
    vehicle_prediction_obj.x_tran = array.array('f', ros_msg.x_tran)
    vehicle_prediction_obj.u_a = array.array('f', ros_msg.u_a)
    vehicle_prediction_obj.u_steer = array.array('f', ros_msg.u_steer)
    vehicle_prediction_obj.lap_num = ros_msg.lap_num
    vehicle_prediction_obj.sey_cov = np.array(ros_msg.sey_cov)
    
    vehicle_prediction_obj.xy_cov = np.array(ros_msg.xy_cov).reshape(-1,2,2)

    return vehicle_prediction_obj