/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#ifndef UTILHEADER
#define UTILHEADER

#include "state_and_types.h"
#include "json.hpp"
#include <torch/torch.h>
#include <torch/script.h>
// #include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <numeric>
#include <cmath>
#include <vector>
#include <ros/ros.h>
#include <eigen3/Eigen/Core>
#include "hmcl_msgs/Lane.h"
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <std_msgs/Float64MultiArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>


torch::Tensor listToTensor(const nlohmann::json& list);

// Function to read a JSON file and convert it to a torch::Tensor
std::tuple<torch::Tensor, torch::Tensor> jsonFileToTensor(const std::string& filePath);



double wrapAngle(double theta);

double computeAnglePose(const Pose& point_0, const Pose& point_1, const Pose& point_2);


double computeAngle(const Point& point_0, const Point& point_1, const Point& point_2);


 double dist(double x1, double y1, double x2, double y2);

// Function to calculate the projection of a point onto a line
 double projectToLine(double x, double y, double cx1, double cy1, double cx2, double cy2, double& px, double& py);



 double normalizeRadian(const double _angle);

 void convertEulerAngleToMonotonic(std::vector<double> &a);

 double find_distance(geometry_msgs::Point p1,geometry_msgs::Point p2);

#endif