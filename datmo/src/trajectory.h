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
#include <vector>
#include <iostream>
#include <cmath>

/** 
 * @class trajectory class for mpc follower
 * @brief calculate control command to follow reference waypoints
 */
class Trajectory
{
public:
  std::vector<double> x;             //!< @brief x position x vector
  std::vector<double> y;             //!< @brief y position y vector
  std::vector<double> z;             //!< @brief z position z vector
  std::vector<double> yaw;           //!< @brief yaw pose yaw vector
  std::vector<double> vx;            //!< @brief vx velocity vx vector(local)
  std::vector<double> vy;            //!< @brief vy velocity vy vector(local) 
  std::vector<double> k;             //!< @brief k curvature k vector
  std::vector<double> relative_time; //!< @brief relative_time duration time from start point
  std::vector<double> s; // progress 
  std::vector<double> ey_l;  
  std::vector<double> ey_r;  

  void erase_to(const int & idx);
  /**
   * @brief push_back for all values
   */
  void push_back(const double &xp, const double &yp, const double &zp,
                 const double &yawp, const double &vxp, const double &vyp, const double &kp,
                 const double &tp, const double &sp, const double &ey_lp, const double &ey_rp);
  /**
   * @brief clear for all values
   */
  void clear();

  Trajectory get_segment(size_t start_idx, size_t end_idx) const;
  

  /**
   * @brief check size of Trajectory
   * @return size, or 0 if the size for each components are inconsistent
   */
  unsigned int size() const;
  double dist( double x_ref,  double y_ref, int traj_index) const;
};
