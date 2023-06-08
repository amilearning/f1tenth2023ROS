#!/usr/bin/env python3
from re import L
import rospy

""" ROS node for the MPC GP in 3d offroad environment, to use in the Gazebo simulator and real world experiments.
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
import time
import math 
# import torch
from nav_msgs.msg import Odometry 
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped 
import rospkg
# from ov_ctrl.utils import construct_frenet_centerline,global_to_frenet, get_odom_euler, get_local_vel, local_to_global, traj_to_markerArray, fren_to_global, torch_fren_to_global, wrap_s, torch_wrap_s, torch_unwrap_s
# from prediction.VAEGPpredictor import VAEGPPredictor
# from prediction.RacingGP import RacingGP
# from ov_ctrl.race_model import VehicleModel, RaceModel
import matplotlib.pyplot as plt
import message_filters
import yaml

from dynamic_reconfigure.server import Server
from ov_ctrl.cfg import dynparamsConfig 
from mppi.model_types import DynamicsConfig, DynamicBicycleConfig
from mppi.solverbuild import SolverBuilder, mpcc_tv_params
from mppi.track import Track
from mppi.pytypes import VehicleState
from mppi.estimateyawcur import piecewise_constant_approximation



rospack = rospkg.RosPack()
pkg_dir = rospack.get_path('ov_ctrl')

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

def distance(p1,p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

class OVCTRL:
    def __init__(self):

        

            
        self.ego_vehicle_name = rospy.get_param("target_ov_ctrl_node/ego_vehicle_name", default='ego')
        self.tar_vehicle_name = rospy.get_param("target_ov_ctrl_node/target_vehicle_name", default='ego')        
        print(self.ego_vehicle_name + " is initializing")

        with open(pkg_dir+"/include/configs/simulation_config.yaml", "r") as f:
            params = yaml.safe_load(f)
        self.debug_accel_zero = params["debug_accel_zero"]
        self.debug_mode = params["debug_mode"]
        self.is_simulation = params["is_simulation"]         
        self.n_nodes = params["prediction_horizon"]         
        self.dt = params["dt"]        
        self.t_horizon = self.n_nodes * self.dt
       
        self.odom = Odometry  
        # s, ey, epsi, vx, vy, wz
        self.ego_cur_x = np.transpose(np.zeros([1,6]))      
        self.cur_state = VehicleState()
        
        self.prediction_switch = False
        self.vae_input_state = None 

        self.cur_odom = None
        self.centerline = None
        self.ego_frenets  = []
        self.tar_frenets  = []
        self.centerline_fren = None
        self.prev_vehicleCmd = None
        self.steering = None
        self.vehicle_status_available = False
        
        # Publishers                         
        self.control_pub = rospy.Publisher("/drive", AckermannDriveStamped, queue_size=1, tcp_nodelay=True)                
        self.debug_marker = rospy.Publisher("debug_path", MarkerArray, queue_size=2)       
        
        self.keypts_sub = rospy.Subscriber("/keypts_info", Marker, self.keypts_callback)                
         
        # set up center line 
        # self.centerline_sub = rospy.Subscriber("/center_line", Marker, self.centerline_callback)                        
        # centerline_msg = rospy.wait_for_message("/global_traj", MarkerArray)
        centerline_msg = rospy.wait_for_message("/centerline_info", Marker)
        if self.centerline_fren is None:
            centerline_s = []
            centerline_curv = []
            track_width = []            
            for i in range(len(centerline_msg.points)):                          
                centerline_s.append(centerline_msg.points[i].x)
                centerline_curv.append(centerline_msg.points[i].y)
                track_width.append(centerline_msg.points[i].z)

                # centerline.append([centerline_msg.points[i].x, centerline_msg.points[i].y])
                # centerline.append([centerline_msg.points[i].pose.position.x, centerline_msg.points[i].pose.position.y])
            # 
            self.centerline_s = np.array([centerline_s]).squeeze()
            self.centerline_curv = np.array([centerline_curv]).squeeze()
            
            self.track_width = np.array([track_width]).squeeze()

            
            # n_segments = 9
            # cur_pw_const,  seg_len, x_segments= piecewise_constant_approximation(self.centerline_curv, n_segments,2)
            # import matplotlib.pyplot as plt
            
            # plt.plot(self.centerline_curv)
            # plt.plot(cur_pw_const)
            # # plt.show()
            # self.centerline_s_apprx = self.centerline_s[seg_len]
            # self.centerline_curv_apprx = self.centerline_curv[seg_len]
            self.track_width = np.ones(len(self.centerline_curv))*0.5

            self.track = Track()
            self.track.initialize_dyn(track_s = self.centerline_s)
            self.mpc_solver = SolverBuilder(model_config = DynamicBicycleConfig, track = self.track, control_params = mpcc_tv_params)
            self.mpc_solver.initialize()

      
        # Subscriber 
        self.prev_odom_stamp = rospy.Time.now()
        
        self.odom_sub = rospy.Subscriber("/fren_pose", PoseStamped, self.fren_callback)                        
        # self.odom_sub = rospy.Subscriber("ground_truth/state_raw", Odometry, self.odom_callback)                        
        # self.vehicle_status_sub = rospy.Subscriber("chassisState", chassisState, self.vehicle_status_callback)                
        

        # 10Hz control callback         
        main_callback_hz = 20
        self.ctrl_counter = 0
        self.ctrl_timer = rospy.Timer(rospy.Duration(1/main_callback_hz), self.ctrl_callback) 


        rate = rospy.Rate(1.0)     
        while not rospy.is_shutdown():                                                         
            # if self.centerline_fren is not None: 
            #     plt.plot(self.centerline_fren[:,2])
            #     plt.show()
            rate.sleep()
    
 
    # def dyn_callback(self,config,level):
    #     self.dyn_params = torch.tensor([config.es_weight,
    #                                     config.obstacle_weight]).to(device=self.torch_device)
    #     self.ego_model.update_param(self.dyn_params)                
    #     return config
    
    def keypts_callback(self,msg):
        self.keypts = np.array([[point.x, point.y] for point in msg.points])
        self.keypts = np.zeros((4,2))
        for i in range(len(msg.points)):
            self.keypts[i,0] = msg.points[i].x
            self.keypts[i,1] = msg.points[i].y
        if i < 4:
            self.keypts[i:,0] = self.keypts[i,0]
            self.keypts[i:,1] = self.keypts[i,1]
     
        

    def fren_callback(self,fren_msg):
        # fren_msg.header
        self.cur_state.p.s = fren_msg.pose.position.x
        self.cur_state.p.x_tran  = fren_msg.pose.position.y
        self.cur_state.p.e_psi = fren_msg.pose.position.z
        self.cur_state.v.v_long = fren_msg.pose.orientation.x
        self.cur_state.v.v_tran = fren_msg.pose.orientation.y
        self.cur_state.w.w_psi = fren_msg.pose.orientation.z
        self.cur_state.u.u_steer = fren_msg.pose.orientation.w
        

              # for k in range(self.N):
            #     sol = output["x%02d" % (k + 1)]

 

    def ctrl_callback(self,timer):

        self.ctrl_counter = self.ctrl_counter + 1
        if self.ctrl_counter > 300: 
            self.ctrl_counter = 0
        start = time.perf_counter()
        current_s = self.cur_state.p.s
        # if self.ctrl_counter < 100:
        #     key_pts = np.array([[current_s-1, 0.0],
        #                             [current_s+2, 0.0],
        #                             [current_s+4, 0.0],
        #                             [current_s+8, 0.0]])
        # elif self.ctrl_counter >= 100 and self.ctrl_counter < 200:
        #     key_pts = np.array([[current_s-1, 3.0],
        #                             [current_s+2, 3.0],
        #                             [current_s+4, 3.0],
        #                             [current_s+8, 3.0]])
            
        # else: 
        #     key_pts = np.array([[current_s-1, -3.0],
        #                             [current_s+2, -3.0],
        #                             [current_s+4, -3.0],
        #                             [current_s+8, -3.0]])
            
        

        output, info, exitflag = self.mpc_solver.solve(self.cur_state.copy(),self.keypts)
        elapsed_time = (time.perf_counter() - start) * 1000
        print("mpc solve elapsed time = " + str(elapsed_time))
        x_pred = []
        u_pred = []
        for k in range(self.n_nodes):
            sol = output["x%02d" % (k + 1)]
            x_pred.append(sol[:4])
            u_pred.append(sol[5:7])
        x_pred_np = np.array(x_pred)
        u_pred_np = np.array(u_pred)
        print(u_pred)
        cmd_msg =  AckermannDriveStamped()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.drive.speed = u_pred_np[0,0]
        cmd_msg.drive.steering_angle = u_pred_np[0,1]
        self.control_pub.publish(cmd_msg)
        
        
        # self.control_pub.publish(ctrl_cmd)
        #######################################################
        # if self.debug_mode:
        #     ego_history = []
        #     ego_x_frenets = []        
        #     for i in range(10):            
        #         pred_ego_u = torch.tensor([ego_action[i,0],ego_action[i,1]]).repeat(2,1).to(device=self.torch_device)        
        #         ego_x = self.ego_model.vehicle_model.dynamics_update(ego_x, pred_ego_u,i)
        #         ego_x_frenet = ego_x[0,:].cpu().numpy()
        #         ego_x_frenets.append(ego_x_frenet)
        #         ego_x_global = fren_to_global(ego_x_frenet, self.centerline_global,self.centerline_fren)            
        #         ego_history.append(ego_x_global)
        #     ego_history = np.array(ego_history).squeeze()
        #     ego_x_frenets = np.array(ego_x_frenets).squeeze()
        #     pred_path_color = [255,255,0]
        #     paths_marker = traj_to_markerArray(ego_history,pred_path_color)
        #     self.debug_marker.publish(paths_marker)
        ############################################################################ 
        
        
        ############################################################################

        
        
        # def _thread_func():
        #     self.compute_action()            
        # self._thread = threading.Thread(target=_thread_func(), args=(), daemon=True)
        # self._thread.start()
        # self._thread.join()
              
        

###################################################################################

def main():
    rospy.init_node("ov_ctrl")
    
    OVCTRL()

if __name__ == "__main__":
    main()




 
    


