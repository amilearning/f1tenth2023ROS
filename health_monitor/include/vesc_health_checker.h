#ifndef VESC_HEALTH_CHECKER_H_
#define VESC_HEALTH_CHECKER_H_

#include <sensor_health_checker.h>
#include <nav_msgs/Odometry.h>

class VescHealthChecker : public SensorHealthChecker<nav_msgs::Odometry> {

public:
    VescHealthChecker(ros::NodeHandle& nh) : SensorHealthChecker<nav_msgs::Odometry>(nh, "/vesc/odom")
    {
       }
    void callback(const nav_msgs::Odometry::ConstPtr& msg) override {
        sensor_data_received_ = true;
        last_sensor_msg_ = msg;
    }
  void checkSensorHealth() override {
        double seq = last_sensor_msg_->header.seq;
        bool health_status;
        if (last_value_ != 0.0) {
            if (seq != last_value_) {
                health_status = true;
            } else {
                health_status = false;
            }
        } else {
            health_status = false;
        }
        last_value_ = seq;
        SensorHealthChecker<nav_msgs::Odometry>::publishHealth(health_status);
    }
    void timerCallback(const ros::TimerEvent& event) override {
        if (!sensor_data_received_) {
            SensorHealthChecker<nav_msgs::Odometry>::publishHealth(false);
        }
        else{
            this->checkSensorHealth();        }
        sensor_data_received_ = false;
    }


};

#endif  // VESC_HEALTH_CHECKER_H_
