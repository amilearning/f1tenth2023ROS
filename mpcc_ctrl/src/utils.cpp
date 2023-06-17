
#include "main_ctrl.h"

void Ctrl::pred_eigen_to_markerArray(const Eigen::MatrixXd & eigen_mtx,  visualization_msgs::MarkerArray & markerArray){
    markerArray.markers.clear();
    for (int row = 0; row < eigen_mtx.rows(); row++) {
        // Create a marker for each row
        visualization_msgs::Marker marker;
        marker.header.frame_id = "map";  // Set the frame ID according to your needs
        marker.header.stamp = ros::Time::now();
        marker.ns = "arrows";
        marker.id = row;
        marker.type = visualization_msgs::Marker::ARROW;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = eigen_mtx(row, 3);
        marker.pose.position.y = eigen_mtx(row, 4);
        double yaw = eigen_mtx(row, 5);  // 5th column - yaw angle
        geometry_msgs::Quaternion orientation = tf2::toMsg(tf2::Quaternion(tf2::Vector3(0, 0, 1), yaw));
        marker.pose.orientation = orientation;
        marker.scale.x = 0.2;  // Arrow length
        marker.scale.y = 0.2;  // Arrow width
        marker.scale.z = 0.1;  // Arrow height
        marker.color.a = 1.0;  // Alpha channel
        marker.color.b = 1.0;  // 
        marker.color.g = 1.0;  // 
        
        markerArray.markers.push_back(marker);
    };


}



    