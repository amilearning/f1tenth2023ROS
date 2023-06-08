#ifndef IMU_HEALTH_CHECKER_H_
#define IMU_HEALTH_CHECKER_H_

#include <sensor_health_checker.h>
#include <sensor_msgs/Imu.h>

class ImuHealthChecker : public SensorHealthChecker<sensor_msgs::Imu> {

public:
    ImuHealthChecker(ros::NodeHandle& nh) : SensorHealthChecker<sensor_msgs::Imu>(nh, "/imu/data")
    {
       }
    void callback(const sensor_msgs::Imu::ConstPtr& msg) override {
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
        SensorHealthChecker<sensor_msgs::Imu>::publishHealth(health_status);
    }

    void timerCallback(const ros::TimerEvent& event) override {
        if (!sensor_data_received_) {
            SensorHealthChecker<sensor_msgs::Imu>::publishHealth(false);
        }
        else{
            this->checkSensorHealth();        }
        sensor_data_received_ = false;
    }


};

#endif  // IMU_HEALTH_CHECKER_H_
