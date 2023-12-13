#include "obstacles.h"


RectangleObstacle::RectangleObstacle(float xc, float yc, float s, float x_tran, float w, float h, float psi)
    : xc(xc), yc(yc), s(s), x_tran(x_tran), w(w), h(h), psi(psi),
      std_local_x(0), std_local_y(0) {
    calc_V();
    calc_Ab_b();
    calc_l_A_u();
}

std::tuple<float, float, float, float, float> RectangleObstacle::circumscribed_ellipse() const {
    float a = w / 2;
    float b = h / 2;
    return std::make_tuple(xc, yc, a, b, psi);
}


Eigen::Matrix2f RectangleObstacle::R() const {
    Eigen::Matrix2f rotation;
    rotation << std::cos(psi), -std::sin(psi),
                std::sin(psi), std::cos(psi);
    return rotation;
}



void RectangleObstacle::calc_V() {
    Eigen::Matrix<float, 5, 2> xy;
    xy << -w/2, -h/2,
          -w/2,  h/2,
           w/2,  h/2,
           w/2, -h/2,
          -w/2, -h/2;
    xy = xy * R() + Eigen::Vector2f(xc, yc);
    
    V = xy.block<4, 2>(0, 0);
    std::cout << "V = " << V << std::endl;
}

void RectangleObstacle::calc_Ab_b() {
    Eigen::Matrix<float, 4, 2> Ab;
    Ab <<  1, 0,
           0, 1,
          -1, 0,
           0,-1;
    
    Ab = Ab * R();

    Eigen::Matrix2f A = Eigen::Matrix2f::Identity();
    Eigen::Vector2f l = (A.transpose().inverse() * Eigen::Vector2f(-xc, -yc)) + Eigen::Vector2f(w/2, h/2);
    Eigen::Vector2f u = (A.transpose().inverse() * Eigen::Vector2f(xc, yc)) + Eigen::Vector2f(w/2, h/2);
    b = Eigen::Vector4f(u(0), u(1), l(0), l(1));
}

void RectangleObstacle::calc_l_A_u() {
    Eigen::Matrix2f A = Eigen::Matrix2f::Identity() * R().transpose();
    l = (A.transpose().inverse() * Eigen::Vector2f(xc, yc)) - Eigen::Vector2f(w/2, h/2);
    u = (A.transpose().inverse() * Eigen::Vector2f(xc, yc)) + Eigen::Vector2f(w/2, h/2);
}
