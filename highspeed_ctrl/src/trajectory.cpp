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

#include "trajectory.h"

void Trajectory::push_back(const double &xp, const double &yp, const double &zp,
                              const double &yawp, const double &vxp, const double &vyp, const double &kp,
                              const double &tp)
{
  x.push_back(xp);
  y.push_back(yp);
  z.push_back(zp);
  yaw.push_back(yawp);
  vx.push_back(vxp);
  vy.push_back(vyp);
  k.push_back(kp);
  relative_time.push_back(tp);
};

void Trajectory::clear()
{
  x.clear();
  y.clear();
  z.clear();
  yaw.clear();
  vx.clear();
  vy.clear();
  k.clear();
  relative_time.clear();
};

void Trajectory::erase_to(const int & idx){
  if(idx < 1 || idx > x.size()-1){
    return;
  }
  x.erase(x.begin(), x.begin() + idx);
  y.erase(y.begin(), y.begin() + idx);
  z.erase(z.begin(), z.begin() + idx);
  yaw.erase(yaw.begin(), yaw.begin() + idx);
  vx.erase(vx.begin(), vx.begin() + idx);
  vy.erase(vx.begin(), vy.begin() + idx);
  k.erase(k.begin(), k.begin() + idx);
  relative_time.erase(relative_time.begin(), relative_time.begin() + idx);
  
}


unsigned int Trajectory::size() const
{
  if (x.size() == y.size() && x.size() == z.size() && x.size() == yaw.size() &&
      x.size() == vx.size() && x.size() == k.size() && x.size() == relative_time.size())
  {
    return x.size();
  }
  else
  {
    std::cerr << "[MPC trajectory] trajectory size is inappropriate" << std::endl;
    return 0;
  }
}


Trajectory Trajectory::get_segment(size_t start_idx, size_t end_idx) const
{
  Trajectory segment;
  
  if (start_idx < end_idx && end_idx <= size())
  {
    for (size_t i = start_idx; i < end_idx; ++i)
    {
      segment.push_back(
        x[i],
        y[i],
        z[i],
        yaw[i],
        vx[i],
        vy[i],
        k[i],
        relative_time[i]
      );
    }
  }
  else
  {
    std::cerr << "[MPC trajectory] Invalid indices for get_segment" << std::endl;
  }

  return segment;
}