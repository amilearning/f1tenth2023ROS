
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
  service_recieved = false;
  nh_p.param<std::string>("control_topic", control_topic, "/vesc/ackermann_cmd");
  // nh_p.param<double>("odom_pose_diff_threshold", odom_pose_diff_threshold, 1.0);
   mpcc_srv = nh_ctrl_.advertiseService("target_compute_mpcc", &Ctrl::mpccService, this);
   ackmanPub = nh_ctrl.advertise<ackermann_msgs::AckermannDriveStamped>(control_topic,2);    
   ego_pred_marker_pub = nh_ctrl.advertise<visualization_msgs::MarkerArray>("target_pred_marker",2);    
   
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
        
        
        // Calculate the total number of columns and rows
        const int numColumns = 10; 
        const int numRows = 19;
        // Create a single vector to store the stacked column vectors
        std::vector<double> stackedVector(numColumns * numRows);
        // Iterate over each column
        for (int column = 0; column < numColumns; ++column) {
        // Copy the column vector to the stacked vector
        std::copy(output.x01, output.x01 + numRows, stackedVector.begin() + column * numRows);
        }

        // Create an Eigen matrix and assign values row-wise
        Eigen::Matrix<gp_mpcc_h2h_ego_float, 10, 19> sol_matrix;
        // Eigen::MatrixXd sol_matrix(10, 19);

        // Assign values for each row
        for (int row = 0; row < 10; row++) {
        sol_matrix.col(row) << output.x01[row], output.x02[row], output.x03[row],
        output.x04[row], output.x05[row], output.x06[row], output.x07[row],
        output.x08[row], output.x09[row], output.x10[row];
        }
        visualization_msgs::MarkerArray ego_pred_marker;
        pred_eigen_to_markerArray(sol_matrix,  ego_pred_marker);
        ego_pred_marker_pub.publish(ego_pred_marker);
        // Print the Eigen matrix
        // std::cout << "Eigen matrix:\n";
        // std::cout << sol_matrix << std::endl;
      // self.zvars = ['vx', 'vy', 'psidot', 'posx', 'posy', 'psi', 'e_psi', 's', 'x_tran', 'theta', 's_prev', 'u_a',
      //                     'u_delta', 'v_proj',
      //                     'u_a_prev', 'u_delta_prev', 'v_proj_prev', 'obs_slack', 'obs_slack_e']
        
        if (exitflag ==1)
        { service_recieved = true;
            cur_cmd.header.stamp = ros::Time::now();
            cur_cmd.drive.steering_angle = output.x01[12];
            double cmd_accel = output.x01[11];
            if (cmd_accel < 0.0){
                // decel, increase lookahead 
                cur_cmd.drive.speed = output.x05[0]; // x04 working fine
            }else{
              // only compensate the pid delay in vesc 
              cur_cmd.drive.speed = output.x03[0]; //x02 working fine
            }
            
        }else{
          ROS_WARN("solver infeasible .. with exitflat = %d", exitflag);
          // roll_to_next_sequence
          cur_cmd.drive.speed = 0.0;
        }

        // return the solution 
    res.exitflag = exitflag;
    res.output.resize(stackedVector.size());
    for (size_t i = 0; i < stackedVector.size(); ++i) {
      res.output[i] = stackedVector[i];
    }


    return true;
}



void Ctrl::ControlLoop()
{ 
    double hz = 10;
    double ctrl_dt = 1/hz;
    ros::Rate loop_rate(10); // rate  

    
    while (ros::ok()){         
      if(service_recieved){
        ackmanPub.publish(cur_cmd);
        }
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


  ros::init(argc, argv, "TargetMpccCtrl");
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
