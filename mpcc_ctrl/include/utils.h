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

#pragma once
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
#include "state_and_types.h"


double wrapAngle(double theta) {
    while (theta < -M_PI) {
        theta += 2 * M_PI;
    }
    while (theta > M_PI) {
        theta -= 2 * M_PI;
    }
    return theta;
}

double computeAnglePose(const Pose& point_0, const Pose& point_1, const Pose& point_2) {
    // Vector from point_0 to point_1
    double v1_x = point_1.x - point_0.x;
    double v1_y = point_1.y - point_0.y;

    // Vector from point_0 to point_2
    double v2_x = point_2.x - point_0.x;
    double v2_y = point_2.y - point_0.y;

    // Dot product and determinant (for cross product in 2D)
    double dot = v1_x * v2_x + v1_y * v2_y;
    double det = v1_x * v2_y - v1_y * v2_x;

    // Angle between vectors
    double theta = std::atan2(det, dot);

    return theta;
}


double computeAngle(const Point& point_0, const Point& point_1, const Point& point_2) {
    // Vector from point_0 to point_1
    double v1_x = point_1.first - point_0.first;
    double v1_y = point_1.second - point_0.second;

    // Vector from point_0 to point_2
    double v2_x = point_2.first - point_0.first;
    double v2_y = point_2.second - point_0.second;

    // Dot product and determinant (for cross product in 2D)
    double dot = v1_x * v2_x + v1_y * v2_y;
    double det = v1_x * v2_y - v1_y * v2_x;

    // Angle between vectors
    double theta = std::atan2(det, dot);

    return theta;
}


double dist(double x1, double y1, double x2, double y2)
{
  return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

// Function to calculate the projection of a point onto a line
double projectToLine(double x, double y, double cx1, double cy1, double cx2, double cy2, double& px, double& py)
{
  double r_numerator = (x - cx1)*(cx2 - cx1) + (y - cy1)*(cy2 - cy1);
  double r_denomenator = pow(cx2 - cx1, 2) + pow(cy2 - cy1, 2);
  double r = r_numerator / r_denomenator;

  double px_temp = cx1 + r*(cx2 - cx1);
  double py_temp = cy1 + r*(cy2 - cy1);

  double s = r*sqrt(r_denomenator);
  double distance = dist(x, y, px_temp, py_temp);

  if (distance < 1e-6) {
    px = x;
    py = y;
    s = 0.0;
  }
  else {
    px = px_temp;
    py = py_temp;
  }
  return distance;
}



double normalizeRadian(const double _angle)
{
  double n_angle = _angle;
  // double n_angle = std::fmod(_angle, 2 * M_PI);
  // n_angle = n_angle > M_PI ? n_angle - 2 * M_PI : n_angle < -M_PI ? 2 * M_PI + n_angle : n_angle;
  while(n_angle > M_PI){
      n_angle = n_angle-2*M_PI;
  }
   while(n_angle < -1*M_PI){
      n_angle = n_angle+2*M_PI;
  }

  // another way
  // Math.atan2(Math.sin(_angle), Math.cos(_angle));
  return n_angle;
}

void convertEulerAngleToMonotonic(std::vector<double> &a)
{
  for (unsigned int i = 1; i < a.size(); ++i)
  {
    const double da = a[i] - a[i - 1];
    a[i] = a[i - 1] + normalizeRadian(da);
  }
}



double find_distance(geometry_msgs::Point p1,geometry_msgs::Point p2){    
    return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y) +(p1.z - p2.z)*(p1.z - p2.z));
} 
