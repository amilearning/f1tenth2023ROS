#include "vehicleState.h"

// Include other necessary headers...

void VehicleState::fromOdometry(const nav_msgs::Odometry& odom) {
    this->t = odom.header.stamp.toSec();
    this->header = odom.header;
    this->pose = odom.pose.pose;
    // Implementation of euler angle conversion and velocity calculations
    tf::Vector3 euler_angles = quaternionToEuler(odom.pose.pose.orientation);
    this->yaw = euler_angles.z();
    double vx_local = odom.twist.twist.linear.x * cos(euler_angles.z()) + odom.twist.twist.linear.y * sin(euler_angles.z());
    double vy_local = -odom.twist.twist.linear.x * sin(euler_angles.z()) + odom.twist.twist.linear.y * cos(euler_angles.z());
    this->vx = vx_local;
    this->vy = vy_local;
    this->wz = 0.0;
}

void VehicleState::updateFrenet(const Track& track){
    Pose tmp_pose = {this->pose.position.x, this->pose.position.y, this->pose.position.z};
    FrenPose fren_pose = track.globalToLocal(tmp_pose);
    this->p = fren_pose;
    // std::cout << "ego fren s: " << p.s << " ey : " << p.ey << std::endl;
}


