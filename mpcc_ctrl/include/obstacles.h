#ifndef OBSTACLE_H
#define OBSTACLE_H

#include <Eigen/Dense>
#include <tuple>
#include <cmath>

class RectangleObstacle {
public:
    RectangleObstacle(float xc = 0, float yc = 0, float s = 0, float x_tran = 0, float w = 0, float h = 0, float psi = 0);

    std::tuple<float, float, float, float, float> circumscribed_ellipse() const;

private:
    float xc, yc, s, x_tran, w, h, psi;
    float std_local_x, std_local_y;

    Eigen::Matrix<float, 4, 2> V;
    Eigen::Matrix<float, 4, 2> Ab;
    Eigen::Vector4f b;
    Eigen::Vector2f l, u;

    void calc_V();
    void calc_Ab_b();
    void calc_l_A_u();
    Eigen::Matrix2f R() const;
};

#endif // OBSTACLE_H