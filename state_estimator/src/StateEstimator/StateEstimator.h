/*
* Software License Agreement (BSD License)
* Copyright (c) 2013, Georgia Institute of Technology
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
* list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
/**********************************************
 * @file StateEstimator.cpp
 * @author Paul Drews <pdrews3@gatech.edu>
 * @author Edited by Hojin Lee <hojinlee@unist.ac.kr> 
 * @date May 1, 2017 / modified 2022
 * @copyright 2017 Georgia Institute of Technology
 * @brief ROS node to fuse information sources and create an accurate state estimation *
 * @details Subscribes to other pose estimate solution, GPS, IMU, and wheel odometry topics, claculates
 * an estimate of the car's current state using GTSAM, and publishes that data.
 ***********************************************/

#ifndef StateEstimator_H_
#define StateEstimator_H_

#include <gtsam/geometry/Quaternion.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/base/timing.h>
#include <GeographicLib/LocalCartesian.hpp>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/navigation/GPSFactor.h>

#include <list>
#include <iostream>
#include <fstream>
#include <queue>

#include <ros/ros.h>
#include <ros/time.h>
#include <std_msgs/Float64.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>

// #include "state_estimator/Diagnostics.h"
#include "BlockingQueue.h"

#include <autorally_msgs/wheelSpeeds.h>
#include <autorally_msgs/imageMask.h>
#include <autorally_msgs/stateEstimatorStatus.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <visualization_msgs/MarkerArray.h>

#define PI 3.14159265358979323846264338


namespace localization_core
{
  class StateEstimator
  {
  private:
    ros::NodeHandle nh_;
    ros::Subscriber gpsSub_, imuSub_, odomSub_, localPoseSub_;
    ros::Publisher  posePub_, estPosePub;
    ros::Publisher  biasAccPub_, biasGyroPub_;
    ros::Publisher  timePub_;
    ros::Publisher statusPub_;
    ros::Publisher gpsPosPub_;

    double lastImuT_, lastImuTgps_; 
    double local_pose_dt;
    unsigned char status_;
    double accelBiasSigma_, gyroBiasSigma_;
    double gpsSigma_, localPoseSigma_;
    int maxQSize_;

    BlockingQueue<sensor_msgs::NavSatFixConstPtr> gpsOptQ_;
    BlockingQueue<sensor_msgs::ImuConstPtr> imuOptQ_;
    BlockingQueue<nav_msgs::OdometryConstPtr> odomOptQ_;
    BlockingQueue<geometry_msgs::PoseStampedConstPtr> localPoseOptQ_;


    boost::mutex optimizedStateMutex_;
    gtsam::NavState optimizedState_;
    double optimizedTime_;
    std::shared_ptr<gtsam::PreintegratedImuMeasurements> imuPredictor_;
    double imuDt_;
    gtsam::imuBias::ConstantBias optimizedBias_, previousBias_;
    sensor_msgs::ImuConstPtr lastIMU_;
    std::shared_ptr<gtsam::PreintegrationParams> preintegrationParams_;

    std::list<sensor_msgs::ImuConstPtr> imuMeasurements_, imuGrav_;
    
    geometry_msgs::PoseStamped initialPose_;
    geometry_msgs::PoseStampedConstPtr ip;
    
    gtsam::Pose3 bodyPSensor_, carENUPcarNED_;
    gtsam::Pose3 imuPgps_;

    bool fixedOrigin_;
    
    bool gotFirstFix_, gotFirstLocalPose_;
    bool invertx_, inverty_, invertz_;
    bool usingOdom_, usingLocalPose_;
    double maxGPSError_, maxLocalPoseError_;

    gtsam::SharedDiagonal priorNoisePose_;
    gtsam::SharedDiagonal priorNoiseVel_;
    gtsam::SharedDiagonal priorNoiseBias_;
    gtsam::Vector noiseModelBetweenBias_sigma_;
    gtsam::ISAM2 *isam_;

    nav_msgs::OdometryConstPtr lastOdom_;
    geometry_msgs::PoseStampedConstPtr lastLocalPose_;
    geometry_msgs::PoseStampedConstPtr prev_local_pose;
  public:
    StateEstimator();
    ~StateEstimator();
    
    void ImuCallback(sensor_msgs::ImuConstPtr imu);
    
    void localPoseCallback(geometry_msgs::PoseStampedConstPtr pose);
    void MainLoop();
    
    
    
    void GetAccGyro(sensor_msgs::ImuConstPtr imu, gtsam::Vector3 &acc, gtsam::Vector3 &gyro);
    
  };
};

#endif /* StateEstimator_H_ */