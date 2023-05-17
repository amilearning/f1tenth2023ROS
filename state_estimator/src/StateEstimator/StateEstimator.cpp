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

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <cmath>

#include <boost/algorithm/string.hpp>
#include <boost/thread/thread.hpp>
#include <vector>
#include "StateEstimator.h"

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>


using namespace gtsam;
// Convenience for named keys
using symbol_shorthand::X;
using symbol_shorthand::V;
using symbol_shorthand::B;
using symbol_shorthand::G; // GPS pose


// macro for getting the time stamp of a ros message
#define TIME(msg) ( (msg)->header.stamp.toSec() )

namespace localization_core
{

  StateEstimator::StateEstimator() :
    // Diagnostics("StateEstimator", "none", ""),
    nh_("~"),
    lastImuT_(0.0),
    lastImuTgps_(0.0),
    maxQSize_(0),
    localPoseOptQ_(40),
    imuOptQ_(400),
    gotFirstFix_(false),
    gotFirstLocalPose_(false)
  {
    // temporary variables to retrieve parameters
    double accSigma, gyroSigma, initialVelNoise, initialBiasNoiseAcc, initialBiasNoiseGyro, initialRotationNoise,
        carXAngle, carYAngle, carZAngle, sensorX, sensorY, sensorZ, sensorXAngle, sensorYAngle, sensorZAngle,
        gravityMagnitude;

    nh_.param<double>("InitialRotationNoise", initialRotationNoise, 0.1);
    nh_.param<double>("InitialVelocityNoise", initialVelNoise, 0.1);
    nh_.param<double>("InitialBiasNoiseAcc", initialBiasNoiseAcc, 1e-1);
    nh_.param<double>("InitialBiasNoiseGyro", initialBiasNoiseGyro, 1e-2);
    nh_.param<double>("AccelerometerSigma", accSigma, 6.0e-2);
    nh_.param<double>("GyroSigma", gyroSigma, 2.0e-2);
    nh_.param<double>("AccelBiasSigma", accelBiasSigma_, 2.0e-4);
    nh_.param<double>("GyroBiasSigma", gyroBiasSigma_, 3.0e-5);
    nh_.param<double>("GPSSigma", gpsSigma_, 0.07);
    nh_.param<double>("localPoseSigma", localPoseSigma_, 0.01);
    nh_.param<double>("SensorTransformX", sensorX, 0.0);
    nh_.param<double>("SensorTransformY", sensorY, 0.0);
    nh_.param<double>("SensorTransformZ", sensorZ, 0.0);
    nh_.param<double>("SensorXAngle",  sensorXAngle, 0);
    nh_.param<double>("SensorYAngle", sensorYAngle, 0);
    nh_.param<double>("SensorZAngle",   sensorZAngle, 0);
    nh_.param<double>("CarXAngle",  carXAngle, 0);
    nh_.param<double>("CarYAngle",  carYAngle, 0);
    nh_.param<double>("CarZAngle",  carZAngle, 0);
    nh_.param<double>("Gravity",   gravityMagnitude, 9.81);
    nh_.param<bool>("InvertX", invertx_, false);
    nh_.param<bool>("InvertY", inverty_, false);
    nh_.param<bool>("InvertZ", invertz_, false);
    nh_.param<double>("Imudt", imuDt_, 1.0/200.0);
    nh_.param<double>("local_pose_dt", local_pose_dt, 1.0/20.0);

    double gpsx, gpsy, gpsz;
    nh_.param<double>("GPSX",  gpsx, 0);
    nh_.param<double>("GPSY",  gpsy, 0);
    nh_.param<double>("GPSZ",  gpsz, 0);
    imuPgps_ = Pose3(Rot3(), Point3(gpsx, gpsy, gpsz));
    imuPgps_.print("IMU->GPS");

    bool fixedInitialPose;
    double initialRoll, intialPitch, initialYaw;

    nh_.param<bool>("FixedInitialPose", fixedInitialPose, false);
    nh_.param<double>("initialRoll", initialRoll, 0);
    nh_.param<double>("intialPitch", intialPitch, 0);
    nh_.param<double>("initialYaw", initialYaw, 0);

    double latOrigin, lonOrigin, altOrigin;
    nh_.param<bool>("FixedOrigin", fixedOrigin_, false);
    nh_.param<double>("latOrigin", latOrigin, 0);
    nh_.param<double>("lonOrigin", lonOrigin, 0);
    nh_.param<double>("altOrigin", altOrigin, 0);

    nh_.param<bool>("UseOdom", usingOdom_, false);
    nh_.param<bool>("UseLocalPose", usingLocalPose_, true);
    
    nh_.param<double>("maxLocalPoseError", maxLocalPoseError_, 1.0);


    std::cout << "InitialRotationNoise " << initialRotationNoise << "\n"
    << "InitialVelocityNoise " << initialVelNoise << "\n"
    << "InitialBiasNoiseAcc " << initialBiasNoiseAcc << "\n"
    << "InitialBiasNoiseGyro " << initialBiasNoiseGyro << "\n"
    << "AccelerometerSigma " << accSigma << "\n"
    << "GyroSigma " << gyroSigma << "\n"
    << "AccelBiasSigma " << accelBiasSigma_ << "\n"
    << "GyroBiasSigma " << gyroBiasSigma_ << "\n"
    << "GPSSigma " << gpsSigma_ << "\n"
    << "SensorTransformX " << sensorX << "\n"
    << "SensorTransformY " << sensorY << "\n"
    << "SensorTransformZ " << sensorZ << "\n"
    << "SensorXAngle " <<  sensorXAngle << "\n"
    << "SensorYAngle " << sensorYAngle << "\n"
    << "SensorZAngle " <<   sensorZAngle << "\n"
    << "CarXAngle " <<  carXAngle << "\n"
    << "CarYAngle " <<  carYAngle << "\n"
    << "CarZAngle " <<  carZAngle << "\n"
    << "Gravity " <<   gravityMagnitude << "\n";

    // Use an ENU frame
    preintegrationParams_ =  PreintegrationParams::MakeSharedU(gravityMagnitude);
    preintegrationParams_->accelerometerCovariance = accSigma * I_3x3;
    preintegrationParams_->gyroscopeCovariance = gyroSigma * I_3x3;
    preintegrationParams_->integrationCovariance = 1e-5 * I_3x3;

    Vector biases((Vector(6) << 0, 0, 0, 0, 0, 0).finished());
    optimizedBias_ = imuBias::ConstantBias(biases);
    previousBias_ = imuBias::ConstantBias(biases);
    imuPredictor_ = std::make_shared<PreintegratedImuMeasurements>(preintegrationParams_, optimizedBias_);

    optimizedTime_ = 0;

    
    while (!ip)
      {
        ROS_WARN("Waiting for valid initial orientation");
        ip = ros::topic::waitForMessage<geometry_msgs::PoseStamped>("/tracked_pose", nh_, ros::Duration(15));
        prev_local_pose = ip;
      }

    initialPose_.pose.orientation.w = ip->pose.orientation.w;
    initialPose_.pose.orientation.x = ip->pose.orientation.x;
    initialPose_.pose.orientation.y = ip->pose.orientation.y;
    initialPose_.pose.orientation.z = ip->pose.orientation.z;


    Rot3 initRot(Quaternion(initialPose_.pose.orientation.w, initialPose_.pose.orientation.x, initialPose_.pose.orientation.y,
          initialPose_.pose.orientation.z));

    bodyPSensor_ = Pose3(Rot3::RzRyRx(sensorXAngle, sensorYAngle, sensorZAngle),
        Point3(sensorX, sensorY, sensorZ));
    carENUPcarNED_ = Pose3(Rot3::RzRyRx(carXAngle, carYAngle, carZAngle), Point3());

    bodyPSensor_.print("Body pose\n");
    carENUPcarNED_.print("CarBodyPose\n");

    posePub_ = nh_.advertise<nav_msgs::Odometry>("pose", 1);
    estPosePub = nh_.advertise<geometry_msgs::PoseStamped>("estimated_pose", 1);
    biasAccPub_ = nh_.advertise<geometry_msgs::Point>("bias_acc", 1);
    biasGyroPub_ = nh_.advertise<geometry_msgs::Point>("bias_gyro", 1);
    timePub_ = nh_.advertise<geometry_msgs::Point>("time_delays", 1);
    statusPub_ = nh_.advertise<autorally_msgs::stateEstimatorStatus>("status", 1);
    

    ISAM2Params params;
    params.factorization = ISAM2Params::QR; // TODO: should test with cholesky later 
    isam_ = new ISAM2(params);

    // prior on the first pose
    priorNoisePose_ = noiseModel::Diagonal::Sigmas(
         (Vector(6) << initialRotationNoise, initialRotationNoise, initialRotationNoise,
             localPoseSigma_, localPoseSigma_, localPoseSigma_).finished());

     // Add velocity prior
     priorNoiseVel_ = noiseModel::Diagonal::Sigmas(
         (Vector(3) << initialVelNoise, initialVelNoise, initialVelNoise).finished());

     // Add bias prior
     priorNoiseBias_ = noiseModel::Diagonal::Sigmas(
         (Vector(6) << initialBiasNoiseAcc,
             initialBiasNoiseAcc,
             initialBiasNoiseAcc,
             initialBiasNoiseGyro,
             initialBiasNoiseGyro,
             initialBiasNoiseGyro).finished());

     std::cout<<"checkpoint"<<std::endl;

     Vector sigma_acc_bias_c(3), sigma_gyro_bias_c(3);
     sigma_acc_bias_c << accelBiasSigma_,  accelBiasSigma_,  accelBiasSigma_;
     sigma_gyro_bias_c << gyroBiasSigma_, gyroBiasSigma_, gyroBiasSigma_;
     noiseModelBetweenBias_sigma_ = (Vector(6) << sigma_acc_bias_c, sigma_gyro_bias_c).finished();

     
     imuSub_ = nh_.subscribe("/imu/data", 10, &StateEstimator::ImuCallback, this);
     
     localPoseSub_ = nh_.subscribe("/tracked_pose", 2, &StateEstimator::localPoseCallback, this);
     boost::thread optimizer(&StateEstimator::MainLoop,this); // main loop
  }

  StateEstimator::~StateEstimator()
  {}
  
  void StateEstimator::localPoseCallback(geometry_msgs::PoseStampedConstPtr pose){
    if(!usingLocalPose_){
      return; 
    }
    
    if( TIME(pose) - TIME(prev_local_pose) > local_pose_dt){
      if (!localPoseOptQ_.pushNonBlocking(pose))
          ROS_WARN("Dropping a LocalPose measurement due to full queue!!");
      prev_local_pose = pose;
    }
    
    
  }

 

  void StateEstimator::GetAccGyro(sensor_msgs::ImuConstPtr imu, Vector3 &acc, Vector3 &gyro)
  {
    double accx, accy, accz;
    if (invertx_) accx = -imu->linear_acceleration.x;
    else accx = imu->linear_acceleration.x;
    if (inverty_) accy = -imu->linear_acceleration.y;
    else accy = imu->linear_acceleration.y;
    if (invertz_) accz = -imu->linear_acceleration.z;
    else accz = imu->linear_acceleration.z;
    acc = Vector3(accx, accy, accz);

    double gx, gy, gz;
    if (invertx_) gx = -imu->angular_velocity.x;
    else gx = imu->angular_velocity.x;
    if (inverty_) gy = -imu->angular_velocity.y;
    else gy = imu->angular_velocity.y;
    if (invertz_) gz = -imu->angular_velocity.z;
    else gz = imu->angular_velocity.z;

    gyro = Vector3(gx, gy, gz);
  }


  void StateEstimator::MainLoop()
  {
    double loop_dt = 0.1;
    ros::Rate loop_rate(1/loop_dt); // rate of main loop
    bool gotFirstFix = false;
    double startTime;
    int odomKey = 1;
    int imuKey = 1;
    int latestLocalPoseKey = 0;
    imuBias::ConstantBias prevBias;
    Vector3 prevVel = (Vector(3) << 0.0,0.0,0.0).finished();
    Pose3 prevPose;
    unsigned char status = autorally_msgs::stateEstimatorStatus::OK;

    while (ros::ok())
    {
      bool optimize = false;
      
      if (!gotFirstFix)
    {
        
        geometry_msgs::PoseStampedConstPtr pose_ = localPoseOptQ_.popBlocking();
        startTime = TIME(pose_);
        if(imuOptQ_.size() <= 0) {
          ROS_WARN_THROTTLE(1, "no IMU messages before first fix, continuing until there is");
          continue;
        }
        // errors out if the IMU and GPS are not close in timestamps
        double most_recent_imu_time = imuOptQ_.back()->header.stamp.toSec();
        if(std::abs(most_recent_imu_time - startTime) > 0.1) {
          ROS_ERROR_STREAM("There is a very large difference in the GPS and IMU timestamps " << most_recent_imu_time - startTime);
          exit(-1);
        }

        NonlinearFactorGraph newFactors;
        Values newVariables;
        gotFirstFix = true;
        

       
        
        // Add prior factors on pose, vel and bias
        Rot3 initialOrientation = Rot3::Quaternion(pose_->pose.orientation.w,
            pose_->pose.orientation.x,
            pose_->pose.orientation.y,
            pose_->pose.orientation.z);
        std::cout << "Initial orientation" << std::endl;
        std::cout << bodyPSensor_.rotation() * initialOrientation * carENUPcarNED_.rotation() << std::endl;
        Pose3 x0(bodyPSensor_.rotation() * initialOrientation * carENUPcarNED_.rotation(),
            Point3(pose_->pose.position.x, pose_->pose.position.y, 0.0));
        prevPose = x0;
        PriorFactor<Pose3> priorPose(X(0), x0, priorNoisePose_);
        newFactors.add(priorPose);
        PriorFactor<Vector3> priorVel(V(0), Vector3(0, 0, 0), priorNoiseVel_);
        newFactors.add(priorVel);
        Vector biases((Vector(6) << 0, 0, 0, 0.0,
            -0.0, -0.0).finished());
        prevBias = imuBias::ConstantBias(biases);
        PriorFactor<imuBias::ConstantBias> priorBias(B(0), imuBias::ConstantBias(biases), priorNoiseBias_);
        newFactors.add(priorBias);

        //Factor for imu->gps translation
        BetweenFactor<Pose3> imuPgpsFactor(X(0), G(0), imuPgps_,
            noiseModel::Diagonal::Sigmas((Vector(6) << 0.001,0.001,0.001,0.03,0.03,0.03).finished()));
        newFactors.add(imuPgpsFactor);

        // add prior values on pose, vel and bias
        newVariables.insert(X(0), x0);
        newVariables.insert(V(0), Vector3(0, 0, 0));
        newVariables.insert(B(0), imuBias::ConstantBias(biases));
        newVariables.insert(G(0), x0.compose(imuPgps_));

        isam_->update(newFactors, newVariables);
        //Read IMU measurements up to the first GPS measurement
        lastIMU_ = imuOptQ_.popBlocking();
        //If we only pop one, we need some dt
        lastImuTgps_ = TIME(lastIMU_) - imuDt_;
        while(TIME(lastIMU_) < TIME(pose_))
        {
          lastImuTgps_ = TIME(lastIMU_);
          lastIMU_ = imuOptQ_.popBlocking();
        }
        loop_rate.sleep();
      }

      else
      {
        NonlinearFactorGraph newFactors;
        Values newVariables;
        

        // add IMU measurements
        while (imuOptQ_.size() > 0 && (TIME(imuOptQ_.back()) > (startTime + imuKey * loop_dt)))
        {          
          double curTime = startTime + imuKey * loop_dt;
          PreintegratedImuMeasurements pre_int_data(preintegrationParams_, previousBias_);
          while(TIME(lastIMU_) < curTime)
          {
            Vector3 acc, gyro;
            GetAccGyro(lastIMU_, acc, gyro);
            double imuDT = TIME(lastIMU_) - lastImuTgps_;
            lastImuTgps_ = TIME(lastIMU_);
            
            pre_int_data.integrateMeasurement(acc, gyro, imuDT);
            lastIMU_ = imuOptQ_.popBlocking();
            
          }
          // adding the integrated IMU measurements to the factor graph
          ImuFactor imuFactor(X(imuKey-1), V(imuKey-1), X(imuKey), V(imuKey), B(imuKey-1), pre_int_data);
          newFactors.add(imuFactor);
          newFactors.add(BetweenFactor<imuBias::ConstantBias>(B(imuKey-1), B(imuKey), imuBias::ConstantBias(),
              noiseModel::Diagonal::Sigmas( sqrt(pre_int_data.deltaTij()) * noiseModelBetweenBias_sigma_)));

          // Predict forward to get an initial estimate for the pose and velocity
          NavState curNavState(prevPose, prevVel);
          NavState nextNavState = pre_int_data.predict(curNavState, prevBias);
          newVariables.insert(X(imuKey), nextNavState.pose());
          newVariables.insert(V(imuKey), nextNavState.v());
          newVariables.insert(B(imuKey), previousBias_);
          newVariables.insert(G(imuKey), nextNavState.pose().compose(imuPgps_));
          prevPose = nextNavState.pose();
          prevVel = nextNavState.v();
          ++imuKey;
          optimize = true;
        }
        

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
        // add LocalPose measurements that are not ahead of the imu messages         
        while (optimize && localPoseOptQ_.size() > 0 && TIME(localPoseOptQ_.front()) < (startTime + (imuKey-1)*loop_dt + 1e-2))
        {
          geometry_msgs::PoseStampedConstPtr local_pose_ = localPoseOptQ_.popBlocking();
          double timeDiff_ = (TIME(local_pose_) - startTime) / loop_dt; // 
          int key_ = round(timeDiff_);                       
          if (std::abs(timeDiff_ - key_) < 1e-1)
          { 
            latestLocalPoseKey = key_;
            gtsam::Pose3 local_pose3_ = Pose3(Rot3::Quaternion(local_pose_->pose.orientation.w, local_pose_->pose.orientation.x, local_pose_->pose.orientation.y, local_pose_->pose.orientation.z),Point3(local_pose_->pose.position.x, local_pose_->pose.position.y, local_pose_->pose.position.z));

            // check if the LocalPoseMsg message is close to our expected position
            Pose3 expectedState_;            
            if (newVariables.exists(X(key_)))
              expectedState_ = (Pose3) newVariables.at<Pose3>(X(key_));
            else
              expectedState_ = isam_->calculateEstimate<Pose3>(X(key_));            
            double dist_ = std::sqrt( std::pow(expectedState_.x() - local_pose_->pose.position.x, 2) + std::pow(expectedState_.y() - local_pose_->pose.position.y, 2) );
            
            if (dist_ < maxLocalPoseError_ || latestLocalPoseKey < imuKey-2)
            {
              SharedDiagonal LocalPoseNoise = noiseModel::Diagonal::Sigmas(
                      (Vector(6) << localPoseSigma_,localPoseSigma_,localPoseSigma_,localPoseSigma_,localPoseSigma_,localPoseSigma_).finished());

              PriorFactor<Pose3> localPosePrior_(X(key_), local_pose3_, LocalPoseNoise);
              newFactors.add(localPosePrior_);
              // newFactors.emplace_shared<PriorFactor<Pose3>>(X(key_), local_pose3_, LocalPoseNoise);
              // newVariables.insert(X(key_), local_pose3_);              
            }
            else
            {
              ROS_WARN("Received bad local Pose message");
              exit(0);
                while(localPoseOptQ_.size()>0)
                {        
                 localPoseOptQ_.popBlocking();
                }
                while(imuOptQ_.size()>0)
                {        
                 imuOptQ_.popBlocking();
                }
              ISAM2Params params;
              params.factorization = ISAM2Params::QR; // TODO: should test with cholesky later 
              isam_ = new ISAM2(params);
              gotFirstFix = false;
              status = autorally_msgs::stateEstimatorStatus::WARN;
               odomKey = 1;
               imuKey = 1;
               latestLocalPoseKey = 0;
               continue;
          
              
              // diag_warn("Received bad local Pose message");
              // TODO 
              // optimize = false;
            
            }
          }
        }

        if(!gotFirstFix){
          status = autorally_msgs::stateEstimatorStatus::WARN;
        loop_rate.sleep();
      continue;}

///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
        
        


        // if we processed imu - then we can optimize the state
        if (optimize)
        {
          try
          {
            isam_->update(newFactors, newVariables);
            Pose3 nextState = isam_->calculateEstimate<Pose3>(X(imuKey-1));

            prevPose = nextState;
            prevVel = isam_->calculateEstimate<Vector3>(V(imuKey-1));
            prevBias = isam_->calculateEstimate<imuBias::ConstantBias>(B(imuKey-1));

            // if we haven't added gps data for 2 message (0.2s) then change status
            if (latestLocalPoseKey + 3 < imuKey)
            {
              status = autorally_msgs::stateEstimatorStatus::WARN;
              // diag_warn("No gps");
            }
            else
            {
              status = autorally_msgs::stateEstimatorStatus::OK;
              // diag_ok("Still ok!");
            }

            double curTime = startTime + (imuKey-1) * 0.1;

            {
              boost::mutex::scoped_lock guard(optimizedStateMutex_);
              optimizedState_ = NavState(prevPose, prevVel);
              optimizedBias_ = prevBias;
              optimizedTime_ = curTime;
              status_ = status;
            }

            nav_msgs::Odometry poseNew;
            poseNew.header.stamp = ros::Time(curTime);

            geometry_msgs::Point ptAcc;
            ptAcc.x = prevBias.vector()[0];
            ptAcc.y = prevBias.vector()[1];
            ptAcc.z = prevBias.vector()[2];

            geometry_msgs::Point ptGyro;
            ptGyro.x = prevBias.vector()[3];
            ptGyro.y = prevBias.vector()[4];
            ptGyro.z = prevBias.vector()[5];

            biasAccPub_.publish(ptAcc);
            biasGyroPub_.publish(ptGyro);
          }
          catch(gtsam::IndeterminantLinearSystemException ex)
          {
            ROS_ERROR("Encountered Indeterminant System Error!");
            // diag_error("State estimator has encountered indeterminant system error");
            status = autorally_msgs::stateEstimatorStatus::ERROR;
            {
              boost::mutex::scoped_lock guard(optimizedStateMutex_);
              status_ = status;
            }
          }
        }
        loop_rate.sleep();
      }
    }
 
 
  }


  void StateEstimator::ImuCallback(sensor_msgs::ImuConstPtr imu)
  {
    double dt;
    if (lastImuT_ == 0) dt = imuDt_; // 200 Hz 
    else dt = TIME(imu) - lastImuT_;

    if(dt < imuDt_){
      return;
    }

    lastImuT_ = TIME(imu);
    //ros::Time before = ros::Time::now();

    // Push the IMU measurement to the optimization thread
    int qSize = imuOptQ_.size();
    if (qSize > maxQSize_)
      maxQSize_ = qSize;
    if (!imuOptQ_.pushNonBlocking(imu))
      ROS_WARN("Dropping an IMU measurement due to full queue!!");

    // Each time we get an imu measurement, calculate the incremental pose from the last GTSAM pose
    imuMeasurements_.push_back(imu);
    //Grab the most current optimized state
    double optimizedTime;
    NavState optimizedState;
    imuBias::ConstantBias optimizedBias;
    unsigned char status;
    {
      boost::mutex::scoped_lock guard(optimizedStateMutex_);
      optimizedState = optimizedState_;
      optimizedBias = optimizedBias_;
      optimizedTime = optimizedTime_;
      status = status_;
    }
    if (optimizedTime == 0) return; // haven't optimized first state yet

    if(status == autorally_msgs::stateEstimatorStatus::WARN){
      return;
    }

    bool newMeasurements = false;
    int numImuDiscarded = 0;
    double imuQPrevTime;
    Vector3 acc, gyro;
    while (!imuMeasurements_.empty() && (TIME(imuMeasurements_.front()) < optimizedTime))
    {
      imuQPrevTime = TIME(imuMeasurements_.front());
      imuMeasurements_.pop_front();
      newMeasurements = true;
      numImuDiscarded++;
    }

    if(newMeasurements)
    {
      //We need to reset integration and iterate through all our IMU measurements
      imuPredictor_->resetIntegration();
      int numMeasurements = 0;
      for (auto it=imuMeasurements_.begin(); it!=imuMeasurements_.end(); ++it)
      {
        double dt_temp =  TIME(*it) - imuQPrevTime;
        imuQPrevTime = TIME(*it);
        GetAccGyro(*it, acc, gyro);
        imuPredictor_->integrateMeasurement(acc, gyro, dt_temp);
        numMeasurements++;
        // ROS_INFO("IMU time %f, dt %f", (*it)->header.stamp.toSec(), dt_temp);
      }
      // ROS_INFO("Resetting Integration, %d measurements integrated, %d discarded", numMeasurements, numImuDiscarded);
    }
    else
    {
      //Just need to add the newest measurement, no new optimized pose
      GetAccGyro(imu, acc, gyro);
      imuPredictor_->integrateMeasurement(acc, gyro, dt);
      // ROS_INFO("Integrating %f, dt %f", m_lastImuT, dt);
    }

   
    // predict next state given the imu measurements
    NavState currentPose = imuPredictor_->predict(optimizedState, optimizedBias);
    nav_msgs::Odometry poseNew;
    poseNew.header.stamp = imu->header.stamp;

    Vector4 q = currentPose.quaternion().coeffs();
    poseNew.pose.pose.orientation.x = q[0];
    poseNew.pose.pose.orientation.y = q[1];
    poseNew.pose.pose.orientation.z = q[2];
    poseNew.pose.pose.orientation.w = q[3];

    poseNew.pose.pose.position.x = currentPose.position().x();
    poseNew.pose.pose.position.y = currentPose.position().y();
    poseNew.pose.pose.position.z = currentPose.position().z();

    poseNew.twist.twist.linear.x = currentPose.velocity().x();
    poseNew.twist.twist.linear.y = currentPose.velocity().y();
    poseNew.twist.twist.linear.z = currentPose.velocity().z();
    
    poseNew.twist.twist.angular.x = gyro.x() + optimizedBias.gyroscope().x();
    poseNew.twist.twist.angular.y = gyro.y() + optimizedBias.gyroscope().y();
    poseNew.twist.twist.angular.z = gyro.z() + optimizedBias.gyroscope().z();

    poseNew.child_frame_id = "base_link";
    poseNew.header.frame_id = "map";

    posePub_.publish(poseNew);
    
    geometry_msgs::PoseStamped est_pose_new;
    est_pose_new.header = poseNew.header;
    est_pose_new.pose.position = poseNew.pose.pose.position;
    est_pose_new.pose.orientation = poseNew.pose.pose.orientation;
    est_pose_new.header.frame_id = "map";
    estPosePub.publish(est_pose_new);

    geometry_msgs::Point delays;
    delays.x = TIME(imu);
    delays.y = (ros::Time::now() - imu->header.stamp).toSec();
    delays.z = TIME(imu) - optimizedTime;
    timePub_.publish(delays);

    // publish the status of the estimate - set in the gpsHelper thread
    autorally_msgs::stateEstimatorStatus statusMsgs;
    statusMsgs.header.stamp = imu->header.stamp;
    statusMsgs.status = status;
    statusPub_.publish(statusMsgs);
    return;
  }

  


  
};

int main (int argc, char** argv)
{
  ros::init(argc, argv, "StateEstimator");
  //ros::NodeHandle n;
  localization_core::StateEstimator wpt;
  ros::spin();
}
