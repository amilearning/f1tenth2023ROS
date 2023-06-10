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
                              const double &tp, const double &sp, const double &ey_lp, const double &ey_rp, const double &lkhp, const double &ovtp)
{
  x.push_back(xp);
  y.push_back(yp);
  z.push_back(zp);
  yaw.push_back(yawp);
  vx.push_back(vxp);
  vy.push_back(vyp);
  k.push_back(kp);
  relative_time.push_back(tp);
  s.push_back(sp);
  ey_l.push_back(ey_lp);
  ey_r.push_back(ey_rp);
  lkh.push_back(lkhp);
  ovt.push_back(ovtp);
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
  s.clear();
  ey_l.clear();
  ey_r.clear();
  lkh.clear();
  ovt.clear();
};

void Trajectory::encode_traj_to_path_info(){

  path.clear();
  path.push_back(x);
  path.push_back(y);
  path.push_back(z);
  path.push_back(yaw);
  path.push_back(vx);
  path.push_back(vy);
  path.push_back(k);
  path.push_back(relative_time);
  path.push_back(s);
  path.push_back(ey_l);
  path.push_back(ey_r);
  path.push_back(lkh);
  path.push_back(ovt);
  

 
}

void Trajectory::encode_from_path_to_traj(std::vector<std::vector<double>> path){
  clear();  
  x = path[0];
  y = path[1];
  z = path[2];
  yaw = path[3];
  vx = path[4];
  vy = path[5];
  k = path[6];
  relative_time = path[7];
  s = path[8];
  ey_l = path[9];
  ey_r = path[10];
  lkh = path[11];
  ovt = path[12];

}


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
  s.erase(s.begin(), s.begin() + idx);  
  ey_l.erase(ey_l.begin(), ey_l.begin() + idx);  
  ey_r.erase(ey_r.begin(), ey_r.begin() + idx);  
  lkh.erase(lkh.begin(), lkh.begin() + idx);  
  ovt.erase(ovt.begin(), ovt.begin() + idx);  
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

double Trajectory::dist( double x_ref,  double y_ref, int traj_index) const{
   return sqrt(pow(x_ref - x[traj_index], 2) + pow(y_ref - y[traj_index], 2));
   
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
        relative_time[i],
        s[i],
        ey_l[i],
        ey_r[i],
        lkh[i],
        ovt[i]
      );
    }
  }
 else if(start_idx > end_idx)
  { // start_idx >= end_idx  --> (start ~ end of traj) + (begining_of_traj + end) 
         for (size_t i = start_idx; i < x.size(); ++i){
           segment.push_back(x[i],
                                y[i],
                                z[i],
                                yaw[i],
                                vx[i],
                                vy[i],
                                k[i],
                                relative_time[i],
                                s[i],
                                ey_l[i],
                                ey_r[i],
                                lkh[i],
                                ovt[i]
          );
         }

         for (size_t i = 0; i <= end_idx; ++i){
           segment.push_back(x[i],
                                y[i],
                                z[i],
                                yaw[i],
                                vx[i],
                                vy[i],
                                k[i],
                                relative_time[i],
                                s[i],
                                ey_l[i],
                                ey_r[i],
                                lkh[i],
                                ovt[i]
          );
         }

    
  }
  else{
    segment.push_back(x[start_idx],
                                y[start_idx],
                                z[start_idx],
                                yaw[start_idx],
                                vx[start_idx],
                                vy[start_idx],
                                k[start_idx],
                                relative_time[start_idx],
                                s[start_idx],
                                ey_l[start_idx],
                                ey_r[start_idx],
                                lkh[start_idx],
                                ovt[start_idx]
          );
  }

  return segment;
}