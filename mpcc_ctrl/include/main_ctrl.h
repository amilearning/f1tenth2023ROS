
//   Copyright (c) 2022 Ulsan National Institute of Science and Technology (UNIST)
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

//   Authour : Hojin Lee, hojinlee@unist.ac.kr


#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <cmath>
#include <cstdlib>
#include <chrono>


#include <boost/algorithm/string.hpp>
#include <boost/thread/thread.hpp>
#include <vector>

#include <sstream>
#include <string>
#include <list>
#include <queue>
#include <mutex> 
#include <thread> 
#include <numeric>
#include <boost/thread/thread.hpp>

#include <ros/ros.h>
#include <ros/time.h>
#include <std_msgs/Float64.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <eigen3/Eigen/Geometry>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>


#include <dynamic_reconfigure/server.h>
#include <mpcc_ctrl/testConfig.h>
#include <std_msgs/UInt8MultiArray.h>

#include <hmcl_msgs/obstacle.h>
#include <hmcl_msgs/TrackArray.h>
#include <hmcl_msgs/Track.h>


#include "hmcl_msgs/mpcc.h"

// #include "FORCESNLPsolver.h"
// #include "FORCESNLPsolver_memory.h"

#include "gp_mpcc_h2h_ego.h"
#include "gp_mpcc_h2h_ego_memory.h"


#define PI 3.14159265358979323846264338
/* AD tool to FORCESPRO interface */


class Ctrl 
{  
private:
ros::NodeHandle nh_ctrl_, nh_p;
bool manual_velocity;
dynamic_reconfigure::Server<mpcc_ctrl::testConfig> srv;
dynamic_reconfigure::Server<mpcc_ctrl::testConfig>::CallbackType f;
ros::ServiceServer mpcc_srv;

// gp_mpcc_h2h_ego_mem * mem_handle;

gp_mpcc_h2h_ego_params mpc_problem;
gp_mpcc_h2h_ego_info info;
gp_mpcc_h2h_ego_output output;
gp_mpcc_h2h_ego_mem * mem;
gp_mpcc_h2h_ego_extfunc extfunc_eval = &gp_mpcc_h2h_ego_adtool2forces;
Eigen::MatrixXd x0;
Eigen::MatrixXd x0i;
bool init_run;
int N;
int nvar;
int neq;    
int npar;
int exitflag;
int return_val;




public:
Ctrl(ros::NodeHandle& nh_ctrl, ros::NodeHandle& nh_p_);
~Ctrl();
void ControlLoop();
bool mpccService(hmcl_msgs::mpcc::Request  &req,hmcl_msgs::mpcc::Response &res);

void dyn_callback(mpcc_ctrl::testConfig& config, uint32_t level);
};



