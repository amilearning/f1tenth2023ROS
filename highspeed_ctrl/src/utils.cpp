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

#include "utils.h"


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
  double n_angle = std::fmod(_angle, 2 * M_PI);
  n_angle = n_angle > M_PI ? n_angle - 2 * M_PI : n_angle < -M_PI ? 2 * M_PI + n_angle : n_angle;

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


template <typename T1, typename T2>
bool interp1d(const T1 &index, const T2 &values, const double &ref, double &ret)
{
  ret = 0.0;
  if (!((int)index.size() == (int)values.size()))
  {
    printf("index and values must have same size, return false. size : idx = %d, values = %d\n", (int)index.size(), (int)values.size());
    return false;
  }
  if (index.size() == 1)
  {
    printf("index size is 1, too short. return false.\n");
    return false;
  }
  unsigned int end = index.size() - 1;
  if (ref < index[0])
  {
    ret = values[0];
    // printf("ref point is out of index (low), return false.\n");
    return true;
  }
  if (index[end] < ref)
  {
    ret = values[end];
    // printf("ref point is out of index (high), return false.\n");
    return true;
  }

  for (unsigned int i = 1; i < index.size(); ++i)
  {
    if (!(index[i] > index[i - 1]))
    {
      printf("index must be monotonically increasing, return false. index[%d] = %f, but index[%d] = %f\n", i, index[i], i-1, index[i - 1]);
      return false;
    }
  }
  unsigned int i = 1;
  while (ref > index[i])
  {
    ++i;
  }
  const double a = ref - index[i - 1];
  const double d_index = index[i] - index[i - 1];
  ret = ((d_index - a) * values[i - 1] + a * values[i]) / d_index;
  return true;
}
template bool interp1d<std::vector<double>, std::vector<double>>(const std::vector<double> &, const std::vector<double> &, const double &, double &);
template bool interp1d<std::vector<double>, Eigen::VectorXd>(const std::vector<double> &, const Eigen::VectorXd &, const double &, double &);
template bool interp1d<Eigen::VectorXd, std::vector<double>>(const Eigen::VectorXd &, const std::vector<double> &, const double &, double &);
template bool interp1d<Eigen::VectorXd, Eigen::VectorXd>(const Eigen::VectorXd &, const Eigen::VectorXd &, const double &, double &);

// 1D interpolation
bool interp1dTraj(const std::vector<double> &index, const Trajectory &values,
                               const std::vector<double> &ref_time, Trajectory &ret)
{
  if (!(index.size() == values.size()))
  {
    printf("index and values must have same size, return false.\n");
    return false;
  }
  if (index.size() == 1)
  {
    printf("index size is 1, too short. return false.\n");
    return false;
  }

  for (unsigned int i = 1; i < index.size(); ++i)
  {
    if (!(index[i] > index[i - 1]))
    {
      printf("index must be monotonically increasing, return false. index[%d] = %f, but index[%d] = %f\n", i, index[i], i-1, index[i - 1]);
      return false;
    }
  }

  for (unsigned int i = 1; i < ref_time.size(); ++i)
  {
    if (!(ref_time[i] > ref_time[i - 1]))
    {
      printf("reference point must be monotonically increasing, return false. ref_time[%d] = %f, but ref_time[%d] = %f\n", i, ref_time[i], i-1, ref_time[i - 1]);
      return false;
    }
  }

  ret.clear();
  unsigned int i = 1;
  for (unsigned int j = 0; j < ref_time.size(); ++j)
  {
    double a, d_index;
    if (ref_time[j] > index.back())
    {
      a = 1.0;
      d_index = 1.0;
      i = index.size() - 1;
    }
    else if (ref_time[j] < index.front())
    {
      a = 0.0;
      d_index = 1.0;
      i = 1;
    }
    else
    {
      while (ref_time[j] > index[i])
      {
        ++i;
      }
      a = ref_time[j] - index[i - 1];
      d_index = index[i] - index[i - 1];
    }
    const double x = ((d_index - a) * values.x[i - 1] + a * values.x[i]) / d_index;
    const double y = ((d_index - a) * values.y[i - 1] + a * values.y[i]) / d_index;
    const double z = ((d_index - a) * values.z[i - 1] + a * values.z[i]) / d_index;
    const double yaw = ((d_index - a) * values.yaw[i - 1] + a * values.yaw[i]) / d_index;
    const double vx = ((d_index - a) * values.vx[i - 1] + a * values.vx[i]) / d_index;
    const double k = ((d_index - a) * values.k[i - 1] + a * values.k[i]) / d_index;
    const double t = ref_time[j];
    double vy = 0.0;
    double s = 0.0;
    ret.push_back(x, y, z, yaw, vx, vy, k, t, s, 3.0, 3.0);
  }
  return true;
}

void calcTrajectoryYawFromXY(Trajectory &traj)
{
  if (traj.yaw.size() == 0)
    return;

  for (unsigned int i = 1; i < traj.yaw.size() - 1; ++i)
  {
    const double dx = traj.x[i + 1] - traj.x[i - 1];
    const double dy = traj.y[i + 1] - traj.y[i - 1];
    traj.yaw[i] = std::atan2(dy, dx);
  }
  if (traj.yaw.size() > 1)
  {
    traj.yaw[0] = traj.yaw[1];
    traj.yaw.back() = traj.yaw[traj.yaw.size() - 2];
  }
}

double find_distance(geometry_msgs::Point p1,geometry_msgs::Point p2){    
    return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y) +(p1.z - p2.z)*(p1.z - p2.z));
} 
void calcTrajectoryCurvature(Trajectory &traj, int curvature_smoothing_num)
{

  
  std::vector<double> x = traj.x;
  std::vector<double> y = traj.y;
  std::vector<double> curv = compute_curvature(x,y);
  
  
      std::cout << "curv = [";
    for (std::vector<double>::const_iterator it = curv.begin(); it != curv.end(); ++it) {
        std::cout << *it;
        if (it != curv.end() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

  traj.k = curv;

  // unsigned int traj_k_size = traj.x.size();
  // traj.k.clear();

  // /* calculate curvature by circle fitting from three points */
  // geometry_msgs::Point p1, p2, p3;
  // for (unsigned int i = curvature_smoothing_num; i < traj_k_size - curvature_smoothing_num; ++i)
  // {
  //   p1.x = traj.x[i - curvature_smoothing_num];
  //   p2.x = traj.x[i];
  //   p3.x = traj.x[i + curvature_smoothing_num];
  //   p1.y = traj.y[i - curvature_smoothing_num];
  //   p2.y = traj.y[i];
  //   p3.y = traj.y[i + curvature_smoothing_num];
  //   double den = std::max(find_distance(p1, p2) * find_distance(p2, p3) * find_distance(p3, p1), 0.0001);
  //   const double curvature = 2.0 * ((p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)) / den;
  //   traj.k.push_back(curvature);
  // }

  /* first and last curvature is copied from next value */
  // for (int i = 0; i < curvature_smoothing_num; ++i)
  // {
  //   traj.k.insert(traj.k.begin(), traj.k.front());
  //   traj.k.push_back(traj.k.back());
  // }
}

void calcTrajectoryS(Trajectory &traj)
{
  unsigned int traj_k_size = traj.x.size();
  traj.s.clear();
  double accumulated_dist = 0.0;
  traj.s.push_back(0.0);
  for (int i = 1; i < traj_k_size; i++) {
    accumulated_dist +=     dist(traj.x[i],traj.y[i], traj.x[i-1], traj.y[i-1]);
    traj.s.push_back(accumulated_dist);
  }

}

void calcTrajectoryFrenet(Trajectory &traj, int curvature_smoothing_num){
  calcTrajectoryS(traj);
  calcTrajectoryCurvature(traj, curvature_smoothing_num);
  
}



void convertWaypointsToMPCTraj(const hmcl_msgs::Lane &lane, Trajectory &mpc_traj)
{
  mpc_traj.clear();
  const double k_tmp = 0.0;
  const double t_tmp = 0.0;
  for (const auto &wp : lane.waypoints)
  {
    const double x = wp.pose.pose.position.x;
    const double y = wp.pose.pose.position.y;
    const double z = wp.pose.pose.position.z;
    const double yaw = tf2::getYaw(wp.pose.pose.orientation);
    const double vx = wp.twist.twist.linear.x;
    double vy = 0.0;
    double s_tmp= 0.0;
    mpc_traj.push_back(x, y, z, yaw, vx, vy, k_tmp, t_tmp,s_tmp, 3.0, 3.0);
  }
}

void convertWaypointsToMPCTrajWithDistanceResample(const hmcl_msgs::Lane &path, const std::vector<double> &path_time,
                                                             const double &dl, Trajectory &ref_traj)
{
  ref_traj.clear();
  double dist = 0.0;
  std::vector<double> dists;
  dists.push_back(0.0);

  for (int i = 1; i < (int)path_time.size(); ++i)
  {
    double dx = path.waypoints.at(i).pose.pose.position.x - path.waypoints.at(i - 1).pose.pose.position.x;
    double dy = path.waypoints.at(i).pose.pose.position.y - path.waypoints.at(i - 1).pose.pose.position.y;
    dist += sqrt(dx * dx + dy * dy);
    dists.push_back(dist);
  }

  convertWaypointsToMPCTrajWithResample(path, path_time, dists, dl, ref_traj);
}


void convertWaypointsToMPCTrajWithTimeResample(const hmcl_msgs::Lane &path, const std::vector<double> &path_time,
                                                         const double &dt, Trajectory &ref_traj)
{
  ref_traj.clear();
  convertWaypointsToMPCTrajWithResample(path, path_time, path_time, dt, ref_traj);
}

void convertWaypointsToMPCTrajWithResample(const hmcl_msgs::Lane &path, const std::vector<double> &path_time,
                                                     const std::vector<double> &ref_index, const double &d_ref_index, Trajectory &ref_traj)
{
  if (ref_index.size() == 0) {
    return;
  }

  for (unsigned int i = 1; i < ref_index.size(); ++i)
  {
    if (ref_index[i] < ref_index[i - 1])
    {
      ROS_ERROR("[convertWaypointsToMPCTrajWithResample] resampling index must be monotonically increasing. idx[%d] = %f, idx[%d+1] = %f",
                i, ref_index[i], i, ref_index[i + 1]);
      return;
    }
  }

  double point = ref_index[0];
  while (point < ref_index.back())
  {
    unsigned int j = 1;
    while (point > ref_index.at(j))
    {
      ++j;
    }

    const double a = point - ref_index.at(j - 1);
    const double ref_index_dist = ref_index.at(j) - ref_index.at(j - 1);
    const geometry_msgs::Pose pos0 = path.waypoints.at(j - 1).pose.pose;
    const geometry_msgs::Pose pos1 = path.waypoints.at(j).pose.pose;
    const geometry_msgs::Twist twist0 = path.waypoints.at(j - 1).twist.twist;
    const geometry_msgs::Twist twist1 = path.waypoints.at(j).twist.twist;
    const double x = ((ref_index_dist - a) * pos0.position.x + a * pos1.position.x) / ref_index_dist;
    const double y = ((ref_index_dist - a) * pos0.position.y + a * pos1.position.y) / ref_index_dist;
    const double z = ((ref_index_dist - a) * pos0.position.z + a * pos1.position.z) / ref_index_dist;

    /* for singular point of euler angle */
    const double yaw0 = tf2::getYaw(pos0.orientation);
    const double dyaw = normalizeRadian(tf2::getYaw(pos1.orientation) - yaw0);
    const double yaw1 = yaw0 + dyaw;
    const double yaw = ((ref_index_dist - a) * yaw0 + a * yaw1) / ref_index_dist;
    const double vx = ((ref_index_dist - a) * twist0.linear.x + a * twist1.linear.x) / ref_index_dist;
    const double curvature_tmp = 0.0;
    const double t = ((ref_index_dist - a) * path_time.at(j - 1) + a * path_time.at(j)) / ref_index_dist;
    double vy = 0.0;
    double s= 0.0;
    ref_traj.push_back(x, y, z, yaw, vx, vy, curvature_tmp, t, s, 3.0, 3.0);
    point += d_ref_index;
  }
}

void calcPathRelativeTime(const hmcl_msgs::Lane &path, std::vector<double> &path_time)
{
  double t = 0.0;
  path_time.clear();
  path_time.push_back(t);
  for (int i = 0; i < (int)path.waypoints.size() - 1; ++i)
  {
    const double x0 = path.waypoints.at(i).pose.pose.position.x;
    const double y0 = path.waypoints.at(i).pose.pose.position.y;
    const double z0 = path.waypoints.at(i).pose.pose.position.z;
    const double x1 = path.waypoints.at(i + 1).pose.pose.position.x;
    const double y1 = path.waypoints.at(i + 1).pose.pose.position.y;
    const double z1 = path.waypoints.at(i + 1).pose.pose.position.z;
    const double dx = x1 - x0;
    const double dy = y1 - y0;
    const double dz = z1 - z0;
    const double dist = sqrt(dx * dx + dy * dy + dz * dz);
    double v = std::max(std::fabs(path.waypoints.at(i).twist.twist.linear.x), 1.0);
    t += (dist / v);
    path_time.push_back(t);
  }
}

bool calcNearestPose(const Trajectory &traj, const geometry_msgs::Pose &self_pose, geometry_msgs::Pose &nearest_pose,
                               unsigned int &nearest_index, double &min_dist_error, double &nearest_yaw_error, double &nearest_time)
{
  int nearest_index_tmp = -1;
  double min_dist_squared = std::numeric_limits<double>::max();
  nearest_yaw_error = std::numeric_limits<double>::max();
  for (uint i = 0; i < traj.size(); ++i)
  {
    const double dx = self_pose.position.x - traj.x[i];
    const double dy = self_pose.position.y - traj.y[i];
    const double dist_squared = dx * dx + dy * dy;

    /* ignore when yaw error is large, for crossing path */
    const double err_yaw = normalizeRadian(tf2::getYaw(self_pose.orientation) - traj.yaw[i]);
    if (fabs(err_yaw) < (M_PI / 3.0))
    {
      if (dist_squared < min_dist_squared)
      {
        /* save nearest index */
        min_dist_squared = dist_squared;
        nearest_yaw_error = err_yaw;
        nearest_index_tmp = i;
      }
    }
  }
  if (nearest_index_tmp == -1)
  {
    ROS_WARN("[calcNearestPose] yaw error is over PI/3 for all waypoints. no closest waypoint found.");
    return false;
  }

  nearest_index = nearest_index_tmp;

  min_dist_error = std::sqrt(min_dist_squared);
  nearest_time = traj.relative_time[nearest_index];
  nearest_pose.position.x = traj.x[nearest_index];
  nearest_pose.position.y = traj.y[nearest_index];

    tf2::Quaternion q;
  q.setRPY(0, 0, traj.yaw[nearest_index]);  

  nearest_pose.orientation = tf2::toMsg(q); 
  return true;
};


bool calcNearestPoseInterp(const Trajectory &traj, const geometry_msgs::Pose &self_pose, geometry_msgs::Pose &nearest_pose,
                                     unsigned int &nearest_index, double &min_dist_error, double &nearest_yaw_error, double &nearest_time)
{

  if (traj.size() == 0)
  {
    ROS_WARN("[calcNearestPoseInterp] trajectory size is zero");
    return false;
  }
  const double my_x = self_pose.position.x;
  const double my_y = self_pose.position.y;
  const double my_yaw = normalizeRadian(tf2::getYaw(self_pose.orientation));

  int nearest_index_tmp = -1;
  double min_dist_squared = std::numeric_limits<double>::max();
  for (uint i = 0; i < traj.size(); ++i)
  {
    const double dx = my_x - traj.x[i];
    const double dy = my_y - traj.y[i];
    const double dist_squared = dx * dx + dy * dy;

    /* ignore when yaw error is large, for crossing path */
    double err_yaw = normalizeRadian(my_yaw - traj.yaw[i]);
    if (fabs(err_yaw) < (M_PI / 3.0))
    {
      if (dist_squared < min_dist_squared)
      {
        /* save nearest index */
        min_dist_squared = dist_squared;
        nearest_index_tmp = i;
      }
    }
  }
  if (nearest_index_tmp == -1)
  {
    ROS_WARN("[calcNearestPoseInterp] yaw error is over PI/3 for all waypoints. no closest waypoint found.");
    return false;
  }

  nearest_index = nearest_index_tmp;

  if (traj.size() == 1)
  {
    nearest_pose.position.x = traj.x[nearest_index];
    nearest_pose.position.y = traj.y[nearest_index];
    tf2::Quaternion q;
    q.setRPY(0, 0, traj.yaw[nearest_index]);
    nearest_pose.orientation = tf2::toMsg(q);
    nearest_time = traj.relative_time[nearest_index];
    min_dist_error = std::sqrt(min_dist_squared);
    nearest_yaw_error = normalizeRadian(my_yaw - traj.yaw[nearest_index]);
    return true;
  }

  /* get second nearest index = next to nearest_index */
  int second_nearest_index = 0;
  if (nearest_index == traj.size() - 1)
    second_nearest_index = nearest_index - 1;
  else if (nearest_index == 0)
    second_nearest_index = 1;
  else
  {
    double dx1, dy1, dist_squared1, dx2, dy2, dist_squared2;
    dx1 = my_x - traj.x[nearest_index + 1];
    dy1 = my_y - traj.y[nearest_index + 1];
    dist_squared1 = dx1 * dx1 + dy1 * dy1;
    dx2 = my_x - traj.x[nearest_index - 1];
    dy2 = my_y - traj.y[nearest_index - 1];
    dist_squared2 = dx2 * dx2 + dy2 * dy2;
    if (dist_squared1 < dist_squared2)
      second_nearest_index = nearest_index + 1;
    else
      second_nearest_index = nearest_index - 1;
  }

  const double a_sq = min_dist_squared;

  /* distance between my position and second nearest position */
  const double dx2 = my_x - traj.x[second_nearest_index];
  const double dy2 = my_y - traj.y[second_nearest_index];
  const double b_sq = dx2 * dx2 + dy2 * dy2;

  /* distance between first and second nearest position */
  const double dx3 = traj.x[nearest_index] - traj.x[second_nearest_index];
  const double dy3 = traj.y[nearest_index] - traj.y[second_nearest_index];
  const double c_sq = dx3 * dx3 + dy3 * dy3;

  /* if distance between two points are too close */
  if (c_sq < 1.0E-5)
  {
    nearest_pose.position.x = traj.x[nearest_index];
    nearest_pose.position.y = traj.y[nearest_index];
    tf2::Quaternion q;
    q.setRPY(0, 0, traj.yaw[nearest_index]);
    nearest_pose.orientation = tf2::toMsg(q);
    nearest_time = traj.relative_time[nearest_index];
    min_dist_error = std::sqrt(min_dist_squared);
    nearest_yaw_error = normalizeRadian(my_yaw - traj.yaw[nearest_index]);
    return true;
  }

  /* linear interpolation */
  const double alpha = 0.5 * (c_sq - a_sq + b_sq) / c_sq;
  nearest_pose.position.x = alpha * traj.x[nearest_index] + (1 - alpha) * traj.x[second_nearest_index];
  nearest_pose.position.y = alpha * traj.y[nearest_index] + (1 - alpha) * traj.y[second_nearest_index];
  double tmp_yaw_err = traj.yaw[nearest_index] - traj.yaw[second_nearest_index];
  if (tmp_yaw_err > M_PI)
  {
    tmp_yaw_err -= 2.0 * M_PI;
  }
  else if (tmp_yaw_err < -M_PI)
  {
    tmp_yaw_err += 2.0 * M_PI;
  }
  const double nearest_yaw = traj.yaw[second_nearest_index] + alpha * tmp_yaw_err;
  tf2::Quaternion q;
  q.setRPY(0, 0, nearest_yaw);
  nearest_pose.orientation = tf2::toMsg(q);
  nearest_time = alpha * traj.relative_time[nearest_index] + (1 - alpha) * traj.relative_time[second_nearest_index];

  /* calcuate the perpendicular distance from ego position to the line joining
     2 nearest way points. */
  auto min_dist_err_sq = b_sq - c_sq * alpha * alpha;

  /* If ego vehicle is very close to or on the line, min_dist_err_sq would be
     very close to 0, any rounding error in the floating point arithmetic
     could cause it to become negative. Hence its value is limited to 0
     in order to perform sqrt. */
  if (min_dist_err_sq < 0) {
    min_dist_err_sq = 0;
  }

  min_dist_error = std::sqrt(min_dist_err_sq);

  nearest_yaw_error = normalizeRadian(my_yaw - nearest_yaw);
  return true;
}




//////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

// Function to calculate the Frenet coordinates
void computeFrenet(VehicleState & state, const Trajectory& center_traj){
  // double x, double y, const vector<double>& cx, const vector<double>& cy, double& s, double& ey, double& epsi)
  if(center_traj.size() < 5){    
    return;
  }
  // Find the nearest point on the centerline to the vehicle position  
  double x = state.pose.position.x;
  double y = state.pose.position.y;
  double min_dist = center_traj.dist(x,y,0);
  int min_index = 0;
  for (int i = 1; i < center_traj.size(); i++) {
    double d = center_traj.dist(x,y,i);
    if (d < min_dist) {
      min_dist = d;
      min_index = i;
    }
  }
  // Calculate the projection of the vehicle position onto the centerline
  double px = 0.0, py = 0.0;
  double fey = 0.0;
  
  if (min_index == 0) {
    fey = projectToLine(x, y, center_traj.x[min_index],
    center_traj.y[min_index ],
    center_traj.x[min_index+1],
    center_traj.y[min_index+1], px, py);
  }else {    
    fey = projectToLine(x, y, center_traj.x[min_index - 1],
    center_traj.y[min_index - 1],
    center_traj.x[min_index],
    center_traj.y[min_index], px, py);
  }
  double ey = 0.0;
    Eigen::Vector2d xy_query(x, y);
    Eigen::Vector2d xy_waypoint(px, py);
     Eigen::Matrix2d rot_global_to_frenet;
    rot_global_to_frenet << std::cos(center_traj.yaw[min_index]), std::sin(center_traj.yaw[min_index]),
                            -std::sin(center_traj.yaw[min_index]), std::cos(center_traj.yaw[min_index]);

    // Calculate the error in xy deviation in global frame
    Eigen::Vector2d error_xy = xy_query - xy_waypoint;
    // Calculate the error in Frenet frame
    Eigen::Vector2d error_frenet = rot_global_to_frenet * error_xy;

    ey =  error_frenet(1);


  // calculate interpolated 's' 
    // state.s = center_traj.s[]
    double a_dist, b_dist;
    double s= 0.0;
    if(min_index ==0){
         s= 0.0;
    }else{  
      s=center_traj.s[min_index-1]+ dist(px,py,center_traj.x[min_index-1],center_traj.y[min_index-1]);
    }
  


  state.ey = ey;
  state.s = s;
  state.epsi = normalizeRadian(state.yaw-center_traj.yaw[min_index]);
  state.k = center_traj.k[min_index];

}



// Function to calculate the Frenet coordinates
void frenToCartician(VehicleState & cart_state,const VehicleState & fren_state, const Trajectory& center_traj){
  // double x, double y, const vector<double>& cx, const vector<double>& cy, double& s, double& ey, double& epsi)
  if(center_traj.size() < 5){    
    return;
  }



  double s = fren_state.s;
  double ey =fren_state.ey;
  double epsi = fren_state.epsi;

  unsigned int s_closest_idx = 1;
  while (s > center_traj.s[s_closest_idx])
  { 
    if(s_closest_idx < center_traj.s.size()-1)
    ++s_closest_idx;
    else{
      break;
    }
  }

  
  double x_on_centerline, y_on_centerline, yaw_on_centerline;
  interp1d(center_traj.s, center_traj.x, s, x_on_centerline);
  interp1d(center_traj.s, center_traj.y, s, y_on_centerline);
  // x_on_centerline = center_traj.x[s_closest_idx];
  // y_on_centerline = center_traj.y[s_closest_idx];
  
  yaw_on_centerline = center_traj.yaw[s_closest_idx];
  // interp1d(center_traj.s, center_traj.yaw, s, yaw_on_centerline);
  // yaw_on_centerline = normalizeRadian(yaw_on_centerline);
  
  double x,y;
 
  if(ey >= 0){
x = x_on_centerline+ fabs(ey)*cos(M_PI/2.0+yaw_on_centerline); 
y = y_on_centerline+ fabs(ey)*sin(M_PI/2.0+yaw_on_centerline);
  }else{
x = x_on_centerline+ fabs(ey)*cos(-M_PI/2.0+yaw_on_centerline); 
y = y_on_centerline+ fabs(ey)*sin(-M_PI/2.0+yaw_on_centerline);
  }
   

  double yaw =  normalizeRadian(yaw_on_centerline + epsi);
 
  cart_state.pose.position.x = x;
  cart_state.pose.position.y = y;
  tf2::Quaternion q;
  q.setRPY(0, 0, yaw);  
  q.normalize();
  cart_state.pose.orientation = tf2::toMsg(q); 
  cart_state.yaw = yaw;


}

void frenToCarticians(std::vector<VehicleState> & states, const Trajectory& center_traj){
  // std::cout << "frentocart"<<std::endl;
  for(int i=0;i < states.size(); i++){
    VehicleState tmp_state = states[i];
    // std::cout << "ey = " << tmp_state.ey << std::endl;
    frenToCartician(states[i], tmp_state, center_traj);
  }
}
  
std::vector<double> moving_average(std::vector<double> data, int window_size) {
    std::vector<double> weights(window_size, 1.0 / window_size);
    // std::vector<double> filtered_data(data.size() - window_size + 1, 0.0);
    std::vector<double> filtered_data(data.size(), 0.0);
    for (int i = 0; i < filtered_data.size(); i++) {
        filtered_data[i] = std::inner_product(data.begin() + i, data.begin() + i + window_size, weights.begin(), 0.0);
    }

    for (int i=data.size()-window_size-1; i < data.size(); i++){
        filtered_data[i] = data[data.size()-1];
        
    }
    return filtered_data;
}


std::vector<double> compute_curvature(std::vector<double>& x_unfiltered, std::vector<double>& y_unfiltered) {
    int window_size = 10;
    std::vector<double> x = moving_average(x_unfiltered, window_size);
    std::vector<double> y = moving_average(y_unfiltered, window_size);

    // the distance between x, y should be evenly spaced.
    // Resample input data to ensure evenly spaced points

    int n_interp =x.size();
    std::vector<double> t(n_interp);
    for (int i = 0; i < n_interp; i++) {
        t[i] = static_cast<double>(i) / static_cast<double>(n_interp-1);
    }

    gsl_interp_accel* accel_x = gsl_interp_accel_alloc();
    gsl_interp_accel* accel_y = gsl_interp_accel_alloc();
    gsl_spline* spline_x = gsl_spline_alloc(gsl_interp_cspline, x.size());
    gsl_spline* spline_y = gsl_spline_alloc(gsl_interp_cspline, y.size());
  
    gsl_spline_init(spline_x,  t.data(),x.data(), x.size());
       
    gsl_spline_init(spline_y,  t.data(), y.data(), y.size());
       

    std::vector<double> fx(n_interp);
    std::vector<double> fy(n_interp);
    for (int i = 0; i < n_interp; i++) {
      try {
        fx[i] = gsl_spline_eval(spline_x, t[i], accel_x);           
         } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
    }

       try {
        fy[i] = gsl_spline_eval(spline_y, t[i], accel_y);  
         } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
    }


    }

    // Compute first and second derivatives
    std::vector<double> dx(n_interp);
    std::vector<double> dy(n_interp);
    std::vector<double> ddx(n_interp);
    std::vector<double> ddy(n_interp);

    for (int i = 0; i < n_interp; i++) {
        dx[i] = gsl_spline_eval_deriv(spline_x, t[i], accel_x);
        dy[i] = gsl_spline_eval_deriv(spline_y, t[i], accel_y);
    }

    for (int i = 1; i < n_interp-1; i++) {
        ddx[i] = gsl_spline_eval_deriv2(spline_x, t[i], accel_x);
        ddy[i] = gsl_spline_eval_deriv2(spline_y, t[i], accel_y);
    }

    // Compute curvature
    std::vector<double> curvature(n_interp);
    for (int i = 1; i < n_interp-1; i++) {
        curvature[i] = (dx[i]*ddy[i] - dy[i]*ddx[i]) / pow(dx[i]*dx[i] + dy[i]*dy[i], 1.5);
    }

    gsl_spline_free(spline_x);
    gsl_spline_free(spline_y);
    gsl_interp_accel_free(accel_x);
    gsl_interp_accel_free(accel_y);

    std::vector<double> filtered_curvature = moving_average(curvature, window_size);

    return filtered_curvature;
}



void genInterpolatedGrid(const std::vector<double>& x_min_max, 
                         const std::vector<double>& y_min_max,
                         const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z){

          const gsl_interp2d_type *T = gsl_interp2d_bicubic;
  double xa[] = {-2.0, -1.0, 0.0, 1.0, 2.0};
  double ya[] = {-2.0, -1.0, 0.0, 1.0, 2.0};
  double za[] = {0.0, 3.0, 4.0, 3.0, 0.0, -3.0, 0.0, 1.0, 0.0, -3.0, -4.0, -1.0, 0.0, -1.0, -4.0, -3.0, 0.0, 1.0, 0.0, -3.0, 0.0, 3.0, 4.0, 3.0, 0.0};
  // double xval[] = {1.0, 1.5, 2.0};
  // double yval[] = {1.0, 1.5, 2.0};
  // double zval[] = {1.2, 1.3, 1.4};
  size_t nx = sizeof(xa) / sizeof(xa[0]);
  size_t ny = sizeof(ya) / sizeof(ya[0]);
  // size_t test_size = sizeof(xval) / sizeof(xval[0]);

  gsl_spline2d *spline = gsl_spline2d_alloc(T, nx, ny);
  gsl_interp_accel *xacc = gsl_interp_accel_alloc();
  gsl_interp_accel *yacc = gsl_interp_accel_alloc();

  gsl_spline2d_init(spline, xa, ya, za, nx, ny);

  printf("-1.5 and 1.0 : %f\n", gsl_spline2d_eval(spline,-1.5,1.0,xacc, yacc));
  printf("1.0 and -1.5 : %f\n", gsl_spline2d_eval(spline,1.0,-1.5,xacc, yacc));

    gsl_spline2d_free(spline);
  gsl_interp_accel_free(xacc);
  gsl_interp_accel_free(yacc);
    // free(za);


  //   if(x.size() != y.size() || x.size()!= z.size() || y.size() != z.size()){
  //     std::cout << "Size does not match 2D interpolation Fail"<<std::endl;
  //   }
  //   // z <- f(x,y) mapping
  //   const gsl_interp2d_type *T = gsl_interp2d_bilinear;
  //   double resolution = 0.1;  

  // // const size_t N = 100;             /* number of points to interpolate */
  // const double xa[] = { x_min_max[0]-1, x_min_max[1]+1}; 
  // const double ya[] = { y_min_max[0]-1, y_min_max[1]+1}; 
  // const size_t nx = sizeof(xa) / sizeof(xa[0]); /* x grid points */
  // const size_t ny = sizeof(ya) / sizeof(ya[0]); /* y grid points */

  // const size_t N =int((*max_element(x.begin(), x.end())-*min_element(x.begin(), x.end()))*
  //                      (*max_element(y.begin(), y.end())-*min_element(y.begin(), y.end()))/
  //                      (resolution * resolution)); /* number of points to interpolate */
  // // const double xa[] = { 0.0, 1.0 }; /* define unit square */
  // // const double ya[] = { 0.0, 1.0 };
  // // const size_t nx = x.size(); /* x grid points */
  // // const size_t ny = y.size(); /* y grid points */
  // // double *za = malloc(nx * ny * sizeof(double));
  // double *za = static_cast<double*>(malloc(nx * ny * sizeof(double)));

  // gsl_spline2d *spline = gsl_spline2d_alloc(T, nx, ny);
  // gsl_interp_accel *xacc = gsl_interp_accel_alloc();
  // gsl_interp_accel *yacc = gsl_interp_accel_alloc();
  // size_t i, j;

  // /* set z grid values */
  // for (int i=0;i<x.size();i++){    
  //     gsl_spline2d_set(spline, za, x[i], y[i], z[i]);
  // }
  // /* initialize interpolation */
  // gsl_spline2d_init(spline, x.data(), y.data(), z.data(), nx, ny);

  // /* interpolate N values in x and y and print out grid for plotting */
  // for (int i=0; i < x.size(); ++i){
  //     double zij = gsl_spline2d_eval(spline, x[i], y[i], xacc, yacc);
  //     if(z[i]- zij < 1e-2){
  //       std::cout << "interpolation has big error or fail"<<std::endl;
  //     }
  // }
  // std::cout << "interpolation done without failure"<<std::endl;
  
  // gsl_spline2d_free(spline);
  // gsl_interp_accel_free(xacc);
  // gsl_interp_accel_free(yacc);
  // free(za);
}