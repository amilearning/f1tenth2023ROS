
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


#include "main_ctrl.h"


using namespace std;

Ctrl::Ctrl(ros::NodeHandle& nh_ctrl,ros::NodeHandle& nh_p_):
  nh_p(nh_p_),
  nh_ctrl_(nh_ctrl)
{
  mem = gp_mpcc_h2h_ego_internal_mem(0);
  exitflag = 0;
  return_val = 0;
  // nh_p.param<double>("odom_pose_diff_threshold", odom_pose_diff_threshold, 1.0);
   mpcc_srv = nh_ctrl_.advertiseService("compute_mpcc", &Ctrl::mpccService, this);
  boost::thread ControlLoopHandler(&Ctrl::ControlLoop,this);   
  

  f = boost::bind(&Ctrl::dyn_callback,this, _1, _2);
	srv.setCallback(f);
}



Ctrl::~Ctrl()
{}

bool Ctrl::mpccService(hmcl_msgs::mpcc::Request  &req,
         hmcl_msgs::mpcc::Response &res)
{
  
      
// /* vector of size 14 */ 
    // gp_mpcc_h2h_ego_float xinit[14];
        std::memcpy(mpc_problem.xinit, req.xinit.data(), req.xinit.size() * sizeof(double));
//       /* vector of size 190 */
    // gp_mpcc_h2h_ego_float x0[190];
        std::memcpy(mpc_problem.x0, req.x0.data(), req.x0.size() * sizeof(double));
// // /* vector of size 610 */
//     // gp_mpcc_h2h_ego_float all_parameters[610];
        std::memcpy(mpc_problem.all_parameters, req.all_parameters.data(), req.all_parameters.size() * sizeof(double));
        
        exitflag = gp_mpcc_h2h_ego_solve(&mpc_problem, &output, &info, mem, NULL, extfunc_eval);
        
//         if (exitflag !=1)
//         {
//             std::cout<< "/n/nmyMPC did not return optimla solution)" <<std::endl;
//         } 




  res.exitflag = 0;
    // Set the values for the output array
    res.output.resize(10);  // Assuming the output array has a size of 10
    for (int i = 0; i < 10; ++i) {
        res.output[i] = 1.23;  // Set a sample value for each element
    }

    return true;
}



void Ctrl::ControlLoop()
{ 
  
    ros::Rate loop_rate(1); // rate  
    
    while (ros::ok()){         
        
     loop_rate.sleep();
   
    }
}

void Ctrl::dyn_callback(mpcc_ctrl::testConfig &config, uint32_t level)
{
  ROS_INFO("Dynamiconfigure updated");  
  manual_velocity = config.manual_velocity;
  return ;
}




int main (int argc, char** argv)
{


  ros::init(argc, argv, "MpccCtrl");
  ros::NodeHandle nh_private("~");
  ros::NodeHandle nh_ctrl;
  Ctrl Ctrl_(nh_ctrl, nh_private);

  // ros::CallbackQueue callback_queue_ctrl, callback_queue_traj, callback_queue_state;
  // nh_ctrl.setCallbackQueue(&callback_queue_ctrl);
  // nh_traj.setCallbackQueue(&callback_queue_traj);
  // nh_state.setCallbackQueue(&callback_queue_state);
  

  // std::thread spinner_thread_ctrl([&callback_queue_ctrl]() {
  //   ros::SingleThreadedSpinner spinner_ctrl;
  //   spinner_ctrl.spin(&callback_queue_ctrl);
  // });


  // std::thread spinner_thread_traj([&callback_queue_traj]() {
  //   ros::SingleThreadedSpinner spinner_traj;
  //   spinner_traj.spin(&callback_queue_traj);
  // });

  //  std::thread spinner_thread_state([&callback_queue_state]() {
  //   ros::SingleThreadedSpinner spinner_state;
  //   spinner_state.spin(&callback_queue_state);
  // });


    ros::spin();

    // spinner_thread_ctrl.join();
    // spinner_thread_traj.join();
    // spinner_thread_state.join();


  return 0;

}
