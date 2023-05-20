#!/usr/bin/env python3
import rospy
import numpy as np
import time
import math
import shutil
import copy
import re
import os

from geometry_msgs.msg import Pose
from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
from visualization_msgs.msg import *


wp_sampling = 10
class PublishTrack():
    def __init__(self):
        self.track_manager=MangeTrackHistroy(self)
        self.server = InteractiveMarkerServer("track_info_interative")
        self.menu_handler = MenuHandler()
        self.menu_list = [ "Anchor1","Anchor2","Vel_Smooth","Pos_Smooth","Vel_Multiply","Control-Z"]
        for menu in self.menu_list:
            self.menu_handler.insert( menu , callback=self.processFeedback)
        self.cst_pub = rospy.Publisher("track_info", MarkerArray, queue_size = 1)
        self.cst_pub2 = rospy.Publisher("track_info_1", MarkerArray, queue_size = 1)
        self.cst_pub3 = rospy.Publisher("track_info_2", MarkerArray, queue_size = 1)
        self.cst_pub4 = rospy.Publisher("track_info_3", MarkerArray, queue_size = 1)
        self.cst_pub5 = rospy.Publisher("track_info_4", MarkerArray, queue_size = 1)
        
        self.interative_marker_sub = rospy.Subscriber("/track_info_interative/update_full",InteractiveMarkerInit,self.InterativeMarkerCallback)

        self.anchor1 = np.nan
        self.anchor2 = np.nan
        self.vel_smooth_pivot = np.nan
        self.pos_smooth_pivot = np.nan
        self.vel_multiply_pivot = np.nan
        self.max_id = np.nan
        obstacle = True
        

        file_name1 = rospy.get_param('/pub_track_gui/filename')
        file_path = os.path.abspath(__file__)
        current_script_path = os.path.dirname(file_path)
        file_name = current_script_path + '/result/'+ file_name1
        print(file_name)
        file_name = current_script_path + '/result/'+ file_name1
        self.track =  self.read_file(file_name,0)
        # self.track =  np.loadtxt(file_name, delimiter=",", dtype = float)
        shutil.copy2(file_name,self.track_manager.name_current)
        self.publish_interative_marker()
  
        
        if obstacle:
            file_name = current_script_path+"/result/traj1_with_boundary.txt"
            self.track1 = self.read_file(file_name, 1000)
            
            file_name = current_script_path+"/result/traj2_with_boundary.txt"
            self.track2 = self.read_file(file_name, 2000)
            
            file_name = current_script_path+"/result/traj3_with_boundary.txt"
            self.track3 = self.read_file(file_name, 3000)
            
            file_name = current_script_path+"/result/traj4_with_boundary.txt"
            self.track4 = self.read_file(file_name, 4000)
        else:
            self.opt_track=self.read_file(file_name)
            self.track1 = self.opt_track
            self.track2 = self.opt_track
            self.track3 = self.opt_track
            self.track4 = self.opt_track
    
        
        self.rate = rospy.Rate(1)

        while not rospy.is_shutdown():
            self.cst_pub.publish(self.track)
            self.cst_pub2.publish(self.track1)
            self.cst_pub3.publish(self.track2)
            self.cst_pub4.publish(self.track3)
            self.cst_pub5.publish(self.track4)
            self.rate.sleep()
            self.server.applyChanges()
                

    def read_file(self, filename, id):
        track = np.loadtxt(filename, delimiter=",", dtype = float)
        
        track_markers = MarkerArray()
        for i in range(len(track)):
            track_marker = Marker()
            track_marker.header.frame_id = "map"  
            track_marker.header.stamp = rospy.Time.now()
            track_marker.ns = "track"
            track_marker.id = id
            track_marker.type = Marker.SPHERE
            track_marker.action = Marker.ADD          
            
            track_marker.pose.position.x = track[i,0] #position x
            track_marker.pose.position.y = track[i,1] #posiiton y
            track_marker.pose.position.z = track[i,2] #velocity
            track_marker.pose.orientation.x = track[i,3] #curvature
            track_marker.pose.orientation.y = track[i,4] #left_width
            track_marker.pose.orientation.z = track[i,5] #right_width
            track_marker.pose.orientation.w = track[i,7] #ey
            track_marker.color.r = track[i,6] # yaw
            track_marker.color.g =255
            track_marker.color.b =0 
            track_marker.color.a = 1
            track_marker.scale.x = 0.2
            track_marker.scale.y = 0.2
            track_marker.scale.z = 0.2
            track_markers.markers.append(track_marker)
            id += 1
        
        return track_markers
    
    def InterativeMarkerCallback(self,msg):
        marker_array = MarkerArray()
        for int_marker in msg.markers:
            marker = Marker()
            marker.header = int_marker.header
            marker.header.frame_id = "map"
            marker.id = int_marker.controls[0].markers[0].id
            marker.pose = int_marker.pose
            marker.pose.orientation.x = self.track.markers[marker.id].pose.orientation.x #curvature
            marker.pose.orientation.y = self.track.markers[marker.id].pose.orientation.y #left_width
            marker.pose.orientation.z = self.track.markers[marker.id].pose.orientation.z #right_width
            marker.pose.orientation.w = self.track.markers[marker.id].pose.orientation.w # ey 
            marker.ns= int_marker.controls[0].markers[0].ns
            marker.type = int_marker.controls[0].markers[0].type
            marker.action = int_marker.controls[0].markers[0].action
            marker.scale = int_marker.controls[0].markers[0].scale
            marker.color = int_marker.controls[0].markers[0].color
            marker.color.a = 1
            marker_array.markers.append(marker)
        # self.cst_pub.publish(marker_array)
        
        if self.track_manager.track_save_flag == True:
            self.track_manager.save_track(marker_array)
            self.publish_interative_marker()
            self.track_manager.track_save_flag = False
        
    def CreateMarkerControl(self, interaction_marker, interaction_mode, name, w,x,y,z):
        tarck_marker_control = InteractiveMarkerControl()
        tarck_marker_control.always_visible = True
        tarck_marker_control.name = name
        tarck_marker_control.orientation.w = w
        tarck_marker_control.orientation.x = x
        tarck_marker_control.orientation.y = y
        tarck_marker_control.orientation.z = z
        tarck_marker_control.interaction_mode = interaction_mode
        interaction_marker.controls.append(tarck_marker_control) 

    def path_contain_zero_wp(self,m1_id, m2_id):
        diff = abs(m1_id - m2_id)
        if diff > 0.5*self.max_id:
            return True
        else:
            return False
        
    def smooth_path_xy(self, marker1_name, marker2_name):
        m1_p = self.server.get(marker1_name).pose.position
        m2_p = self.server.get(marker2_name).pose.position
        m1_p = np.array([m1_p.x, m1_p.y])
        m2_p = np.array([m2_p.x, m2_p.y])
        m1_id = int(marker1_name[2:])
        m2_id = int(marker2_name[2:])
        direction = m2_p - m1_p
        distance = np.linalg.norm(direction)
        distance = np.linalg.norm(direction)
        if distance ==0:
            unit_vector = 0
        else:
            unit_vector = direction / distance
        
        if self.path_contain_zero_wp(m1_id, m2_id):
            step_size = distance / (self.max_id - abs(m1_id - m2_id))
            if m1_id < m2_id:
                tmp1 = range(m1_id, -1,-1)
                tmp2 = range(self.max_id,m2_id-1,-1)
                smooth_range = list(tmp1)+list(tmp2)
            else:
                tmp1 = range(m2_id,-1,-1)
                tmp2 = range(self.max_id,m1_id-1,-1)
                smooth_range = list(tmp1)+list(tmp2)
        else:
            if m1_id -m2_id ==0:
                step_size = 0
            else:
                step_size = distance/ abs(m1_id - m2_id)
            if m1_id < m2_id:
                smooth_range = range(m1_id, m2_id+1,1)
            else:
                smooth_range = range(m1_id, m2_id-1,-1)

        for i in smooth_range:
            if self.path_contain_zero_wp(i,m1_id):
                offset = (self.max_id - abs(i-m1_id)) * step_size
            else:
                offset = abs(i-m1_id) * step_size

            position = tuple(m1_p + offset*unit_vector)
            marker_name = "wp"+str(i)
            ori_pose = self.server.get(marker_name).pose
            new_pose = Pose()
            new_pose.position.x = position[0]
            new_pose.position.y = position[1]
            new_pose.position.z = ori_pose.position.z
            new_pose.orientation = ori_pose.orientation
            self.server.setPose(marker_name, new_pose)

        self.menu_handler.reApply( self.server )
        self.server.applyChanges()
        return
    
    def smooth_path_z(self, marker1_name, marker2_name):
        m1_p = self.server.get(marker1_name).pose.position
        m2_p = self.server.get(marker2_name).pose.position
        m1_p = np.array([m1_p.z])
        m2_p = np.array([m2_p.z])
        m1_id = int(marker1_name[2:])
        m2_id = int(marker2_name[2:])
        direction = m2_p - m1_p
        distance = np.linalg.norm(direction)
        if distance ==0:
            unit_vector = 0
        else:
            unit_vector = direction / distance
        
        if self.path_contain_zero_wp(m1_id, m2_id):
            step_size = distance / (self.max_id - abs(m1_id - m2_id))
            if m1_id < m2_id:
                tmp1 = range(m1_id, -1,-1)
                tmp2 = range(self.max_id,m2_id-1,-1)
                smooth_range = list(tmp1)+list(tmp2)
            else:
                tmp1 = range(m2_id,-1,-1)
                tmp2 = range(self.max_id,m1_id-1,-1)
                smooth_range = list(tmp1)+list(tmp2)
        else:
            if m1_id -m2_id ==0:
                step_size = 0
            else:
                step_size = distance/ abs(m1_id - m2_id)
            if m1_id < m2_id:
                smooth_range = range(m1_id, m2_id+1,1)
            else:
                smooth_range = range(m1_id, m2_id-1,-1)

        for i in smooth_range:
            if self.path_contain_zero_wp(i,m1_id):
                offset = (self.max_id - abs(i-m1_id)) * step_size
            else:
                offset = abs(i-m1_id) * step_size

            position = tuple(m1_p + offset*unit_vector)
            marker_name = "wp"+str(i)
            ori_pose = self.server.get(marker_name).pose
            new_pose = Pose()
            new_pose.position.x = ori_pose.position.x
            new_pose.position.y = ori_pose.position.y
            new_pose.position.z = position[0]
            new_pose.orientation = ori_pose.orientation
            self.server.setPose(marker_name, new_pose)

        self.menu_handler.reApply( self.server )
        self.server.applyChanges()
        return

    def multiply_path_z(self, marker1_name, marker2_name):
        file_name = self.track_manager.name_p1revious
        prev_track = np.loadtxt(file_name, delimiter=",", dtype = float)
        m1_p = self.server.get(marker1_name).pose.position
        m2_p = self.server.get(marker2_name).pose.position
        m1_p = np.array([m1_p.z])
        m2_p = np.array([m2_p.z])
        m1_id = int(marker1_name[2:])
        m2_id = int(marker2_name[2:])
        pivot_id = int(self.vel_multiply_pivot[2:])
        marker_name = self.vel_multiply_pivot
        pivot_p = self.server.get(marker_name).pose
        if self.path_contain_zero_wp(m1_id, m2_id):
            if m1_id < m2_id:
                tmp1 = range(m1_id, -1,-1)
                tmp2 = range(self.max_id,m2_id-1,-1)
                smooth_range = list(tmp1)+list(tmp2)
            else:
                tmp1 = range(m2_id,-1,-1)
                tmp2 = range(self.max_id,m1_id-1,-1)
                smooth_range = list(tmp1)+list(tmp2)
        else:
            if m1_id < m2_id:
                smooth_range = range(m1_id, m2_id+1,1)
            else:
                smooth_range = range(m1_id, m2_id-1,-1)

        rate = pivot_p.position.z / prev_track[pivot_id,2]
        for i in smooth_range:
            marker_name = "wp"+str(i)
            ori_pose = self.server.get(marker_name).pose
            new_pose = Pose()
            new_pose.position.x = ori_pose.position.x
            new_pose.position.y = ori_pose.position.y
            if i==pivot_id or i==m1_id or i == m2_id:
                new_pose.position.z = ori_pose.position.z
            else:
                new_pose.position.z =  prev_track[i,2]*rate
            new_pose.orientation = ori_pose.orientation
            self.server.setPose(marker_name, new_pose)

        self.menu_handler.reApply( self.server )
        self.server.applyChanges()
        return

    def processFeedback(self,feedback):
        p = feedback.pose.position

        if feedback.event_type == InteractiveMarkerFeedback.MENU_SELECT:
            selected_menu = self.menu_list[feedback.menu_entry_id-1]
            print(feedback.marker_name + ": menu item " + selected_menu + " clicked" )
            if selected_menu == "Anchor1":
                self.anchor1 = feedback.marker_name
            elif selected_menu == "Anchor2":
                self.anchor2 = feedback.marker_name
            elif selected_menu == "Vel_Smooth":
                self.vel_smooth_pivot = feedback.marker_name
                self.smooth_path_z(self.anchor1,self.vel_smooth_pivot)
                self.smooth_path_z(self.anchor2,self.vel_smooth_pivot)
                self.smooth_path_z(self.anchor1,self.vel_smooth_pivot)
                self.smooth_path_z(self.anchor2,self.vel_smooth_pivot)
                self.smooth_path_z(self.anchor1,self.vel_smooth_pivot)
                self.smooth_path_z(self.anchor2,self.vel_smooth_pivot)
                print("smooth vel finished")
                self.track_manager.track_save_flag = True
            elif selected_menu == "Pos_Smooth":
                self.pos_smooth_pivot = feedback.marker_name
                self.smooth_path_xy(self.anchor1,self.pos_smooth_pivot)
                self.smooth_path_xy(self.anchor2,self.pos_smooth_pivot)
                self.smooth_path_xy(self.anchor1,self.pos_smooth_pivot)
                self.smooth_path_xy(self.anchor2,self.pos_smooth_pivot)
                self.smooth_path_xy(self.anchor1,self.pos_smooth_pivot)
                self.smooth_path_xy(self.anchor2,self.pos_smooth_pivot)
                print("smooth pos finished")
                self.track_manager.track_save_flag = True
            elif selected_menu == "Vel_Multiply":
                self.vel_multiply_pivot = feedback.marker_name
                self.multiply_path_z(self.anchor1, self.anchor2)
                self.multiply_path_z(self.anchor1, self.anchor2)
                self.multiply_path_z(self.anchor1, self.anchor2)
                print("multiply vel finished")
                self.track_manager.track_save_flag = True
            elif selected_menu == "Control-Z":
                self.track_manager.control_z_track()
                print("load last track finished")
            else:
                print("something wrong..")
                exit()

        elif feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            print(feedback.marker_name + " is now at " + str(p.x) + ", " + str(p.y) + ", " + str(p.z) )
        
        if feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
             self.track_manager.track_save_flag = True

        return

    def publish_interative_marker(self):
        id = 0
        for marker in self.track.markers:
            int_marker = InteractiveMarker()
            int_marker.header.frame_id = "map"
            int_marker.name ="wp"+str(id)
            if id%wp_sampling ==0:
                int_marker.description = str(id)
            int_marker.header.stamp = rospy.Time.now()
            int_marker.scale = 0.5
            int_marker.pose.position.x = marker.pose.position.x #position x
            int_marker.pose.position.y = marker.pose.position.y #posiiton y
            int_marker.pose.position.z = marker.pose.position.z #velocity
            yaw = marker.color.r
            qx = 0.0
            qy = 0.0
            qz = math.sin(yaw / 2.0)
            qw = math.cos(yaw / 2.0)
            norm = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
            qx /= norm
            qy /= norm
            qz /= norm
            qw /= norm
            int_marker.pose.orientation.x = qx
            int_marker.pose.orientation.y = qy
            int_marker.pose.orientation.z = qz
            int_marker.pose.orientation.w = qw

            # # visualize marker
            tarck_marker_control = InteractiveMarkerControl()
            tarck_marker_control.always_visible = True
            track_marker = Marker()
            track_marker.id = id
            # track_marker.ns = "track"
            track_marker.header.stamp = rospy.Time.now()
            track_marker.type = Marker.SPHERE
            track_marker.ns = "track"
            track_marker.color.r = yaw # yaw
            track_marker.color.g =255
            track_marker.color.b =0 
            if id%wp_sampling ==0:
                track_marker.color.a = 1
            else:
                 track_marker.color.a = 0
            track_marker.scale.x = 0.2
            track_marker.scale.y = 0.2
            track_marker.scale.z = 0.2
            self.max_id = id
            id += 1
            tarck_marker_control.markers.append(track_marker)
            int_marker.controls.append(tarck_marker_control)
            if id%wp_sampling ==0:
                # move_x
                self.CreateMarkerControl(int_marker,InteractiveMarkerControl.MOVE_AXIS,"move_x",1,1,0,0)
                # move_y
                self.CreateMarkerControl(int_marker,InteractiveMarkerControl.MOVE_AXIS,"move_y",1,0,0,1)
                # move_z
                self.CreateMarkerControl(int_marker,InteractiveMarkerControl.MOVE_AXIS,"move_z",1,0,1,0)
                # menu
                self.CreateMarkerControl(int_marker,InteractiveMarkerControl.BUTTON,"menu",0,0,0,1)

            self.server.insert(int_marker, self.processFeedback)
            self.menu_handler.apply( self.server, int_marker.name )
        self.server.applyChanges()
            
class MangeTrackHistroy():
    def __init__(self,PublishTrack):
        file_path = os.path.abspath(__file__)
        current_script_path = os.path.dirname(file_path)
        directory = current_script_path + '/gui_tmp'
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print("Remove existing tmp folder: ",directory)
        os.makedirs(directory)
        print("Create tmp folder: ", directory)
        self.name_p1revious = directory + '/p1revious_track.txt'
        self.name_p2revious = directory + '/p2revious_track.txt'
        self.name_p3revious = directory + '/p2revious_track.txt'
        self.name_current = directory + '/current_track.txt'
        self.name_modified = current_script_path + '/result/rviz_modified_track.txt'
        self.PublishTrack = PublishTrack
        self.track_save_flag = False
    
    def save_track(self,marker_array):
        print("Start saving track...")
        self.shift_saved_track_forward()
        with open(self.name_current, 'w') as file:
        # Loop through each marker in the array
            for marker in marker_array.markers:
                # Write the pose information to a new line in the file
                file.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(
                    marker.pose.position.x, marker.pose.position.y, marker.pose.position.z,
                    marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z, marker.color.r, marker.pose.orientation.w))
        self.PublishTrack.track = self.PublishTrack.read_file(self.name_current,0)
        shutil.copy2(self.name_current,self.name_modified)
        print("End saving track...")

    def control_z_track(self):
        directory = self.name_p1revious
        if not os.path.exists(directory):
            print('\033[91m'+ "can not load last track more than 3 times!!"+'\033[0m')
            print('\033[91m'+ "can not load last track more than 3 times!!"+'\033[0m')
            print('\033[91m'+ "can not load last track more than 3 times!!"+'\033[0m')
            print('\033[91m'+ "can not load last track more than 3 times!!"+'\033[0m')
            return
        self.shift_saved_track_backward()
        self.PublishTrack.track = self.PublishTrack.read_file(self.name_current,0)
        self.PublishTrack.publish_interative_marker()
        shutil.copy2(self.name_current,self.name_modified)

    def shift_saved_track_forward(self):
        self.rename_txt(self.name_p2revious,self.name_p3revious)
        self.rename_txt(self.name_p1revious,self.name_p2revious)
        self.rename_txt(self.name_current,self.name_p1revious)

    def shift_saved_track_backward(self):
        self.rename_txt(self.name_p1revious,self.name_current)
        self.rename_txt(self.name_p2revious,self.name_p1revious)
        self.rename_txt(self.name_p3revious,self.name_p2revious)
       
    def rename_txt(self,src, dst):
        if os.path.exists(src):
            os.rename(src,dst)

def main():
    rospy.init_node("track_info_node")
    PublishTrack()

if __name__ == '__main__':
    main()
