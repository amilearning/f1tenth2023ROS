#ifndef SENSOR_HEALTH_CHECKER_H_
#define SENSOR_HEALTH_CHECKER_H_

#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/Imu.h>
#include <boost/bind.hpp>
#include <atomic>

template <typename MessageType>
class SensorHealthChecker {

protected:
    ros::Publisher health_pub_;
    ros::Subscriber sensor_sub_;
    ros::Timer timer_;
    std::atomic<bool> sensor_data_received_;
    typename MessageType::ConstPtr last_sensor_msg_;
    double last_value_;

    void publishHealth(bool status) {
        std_msgs::Bool health_msg;
        health_msg.data = status;
        health_pub_.publish(health_msg);
    }

public:
    SensorHealthChecker(ros::NodeHandle& nh, const std::string& topic) : sensor_data_received_(false),last_value_(0.0) {
        health_pub_ = nh.advertise<std_msgs::Bool>(topic + "/health", 1);
        sensor_sub_ = nh.subscribe(topic, 10, &SensorHealthChecker::callback, this);
        timer_ = nh.createTimer(ros::Duration(1.0), boost::bind(&SensorHealthChecker::timerCallback, this, _1));
    }

    virtual ~SensorHealthChecker() {}

    virtual void callback(const typename MessageType::ConstPtr& msg) {
        sensor_data_received_ = true;
    }

    virtual void timerCallback(const ros::TimerEvent& event) {
        if (!sensor_data_received_) {
            publishHealth(false);
        } 
        else {
            checkSensorHealth();
        }
        sensor_data_received_ = false;
    }

    virtual void checkSensorHealth() {
    }
};

#endif  // SENSOR_HEALTH_CHECKER_H_