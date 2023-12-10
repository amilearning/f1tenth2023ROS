

#include "track.h"

Track::Track(double initial_x, double initial_y, double initial_psi, double track_width, double slack)
: initial_x(initial_x), initial_y(initial_y), initial_psi(initial_psi),
  track_width(track_width), slack(slack) {
   
        // Add initial segment
        // addSegment(initial_x, initial_y, initial_psi, 0.0, 0.0);
        segments.push_back({0.0, 0.0});
        segments.push_back({0.3, 1});
        segments.push_back({-0.3, 2});
        segments.push_back({0.2, 3});        
        segments.push_back({0.1, 3});        
        getKeyPts();
        for (const auto& kp : key_pts) {
            std::cout << "Start X: " << kp.x 
                      << ", Start Y: " << kp.y
                      << ", Start Psi: " << kp.psi 
                      << ", Cumulative Length: " << kp.cum_length 
                      << ", Segment Length: " << kp.segment_length 
                      << ", Curvature: " << kp.curvature << std::endl;
        }
        
    }




void Track::test_local_and_global() {
        double x = 2.1;
        double y = -0.39; 
        double psi = -0.39; 
        Pose globalPose = {x,y,psi}; // Example global pose
        auto [s, ey, epsi] = globalToLocal(globalPose);
        FrenPose f = {s, ey, epsi};
        auto [nx, ny, npsi] = localToGlobal(f);
        // Define a tolerance for floating-point comparison
        const double tolerance = 1e-6;
        // Use the tolerance in the assertions
        assert(std::abs(x - nx) < tolerance);
        assert(std::abs(y - ny) < tolerance);
        assert(std::abs(psi - npsi) < tolerance);

    }



std::tuple<double, double, double> Track::nextPositionOrientation(double x, double y, double psi, double curvature, double segment_length) {
        double x_next, y_next, psi_next;

        if (curvature == 0) {  // Straight line
            x_next = x + segment_length * std::cos(psi);
            y_next = y + segment_length * std::sin(psi);
            psi_next = psi;
        } else {  // Circular path
            double radius = 1 / curvature;
            double angle_change = segment_length / radius;
            psi_next = psi + angle_change;
            double x_center = x - radius * std::sin(psi);
            double y_center = y + radius * std::cos(psi);
            x_next = x_center + radius * std::sin(psi_next);
            y_next = y_center - radius * std::cos(psi_next);
        }

        return std::make_tuple(x_next, y_next, psi_next);
    }




void Track::getKeyPts() {
        double x = initial_x;  // Using the member variable initial_x
        double y = initial_y;  // Using the member variable initial_y
        double psi = initial_psi; 
         
        for (const auto& segment : segments) {
            double curvature = segment.first;
            double segment_length = segment.second;
            
            cumulative_length += segment_length;
            auto [x_next, y_next, psi_next] = nextPositionOrientation(x, y, psi, curvature, segment_length);            
            KeyPoint kpt = {x_next, y_next, psi_next, cumulative_length, segment_length, curvature};
            key_pts.push_back(kpt);
            x = x_next;
            y = y_next;
            psi = psi_next;

            
        }
        track_length = cumulative_length;
        std::cout << "track_length= " << track_length << std::endl;        
        std::cout << "Key Points Generated" << std::endl;
    }



FrenPose Track::globalToLocal(const Pose& pos_cur) {
        if (key_pts.empty()) {
            throw std::runtime_error("Track key points have not been defined");
        }

        double x = pos_cur.x;
        double y = pos_cur.y;        
        double psi = pos_cur.psi;
        // Assuming psi (orientation) is part of pos_cur, you will need to adjust this based on your actual data structure

        FrenPose cl_coord;
         
        for (size_t i = 1; i < key_pts.size(); ++i) {
            const auto& kpt_start = key_pts[i - 1];
            const auto& kpt_end = key_pts[i];

            Pose pos_s = {kpt_start.x, kpt_start.y,kpt_start.psi};
            Pose pos_f  = {kpt_end.x, kpt_end.y, kpt_end.psi};
            

            // Check if at any of the segment start or end points
            // This uses Euclidean distance to check proximity to the start/end points
            if (std::hypot(pos_s.x - x, pos_s.y - y) < 1e-6) {
                // At start of segment
                double s = kpt_start.cum_length;
                double e_y = 0;
                double e_psi = std::remainder(psi - kpt_start.psi, 2 * M_PI);                                
                FrenPose cl_coord = {s, e_y, e_psi};
                
                break;
            }
            if (std::hypot(pos_f.x - x, pos_f.y - y) < 1e-6) {
                // At end of segment
                double s = kpt_end.cum_length;
                double e_y = 0;
                double e_psi = std::remainder(psi - kpt_end.psi, 2 * M_PI);                
                FrenPose cl_coord = {s, e_y, e_psi};
                
                break;
            }
            
            double curve_f = kpt_end.curvature;
            if (curve_f == 0) {
              
                // Straight segment
                if (std::abs(computeAnglePose(pos_s, pos_cur, pos_f)) <= M_PI / 2 &&
                    std::abs(computeAnglePose(pos_f, pos_cur, pos_s)) <= M_PI / 2) {

                    Pose v = {pos_cur.x - pos_s.x, pos_cur.y - pos_s.y, 0.0};
                    double ang = computeAnglePose(pos_s, pos_f, pos_cur);
                    double e_y = std::hypot(v.x, v.y) * std::sin(ang);                    
                    

                    if (std::abs(e_y) <= track_width / 2 + slack) {
                        double d = std::hypot(v.x, v.y) * std::cos(ang);
                        double s = kpt_start.cum_length + d;
                        double e_psi = std::remainder(psi - kpt_start.psi, 2 * M_PI);
                        
                        return FrenPose{s, e_y, e_psi};
                    }
                }
                
            }
             else {
                // Curved segment                
                  std::cout << " curve_f = " << curve_f << std::endl;
                double r = 1 / curve_f;
                int dir = (r > 0) ? 1 : -1;

                double x_c = kpt_start.x + std::abs(r) * std::cos(kpt_start.psi + dir * M_PI / 2);
                double y_c = kpt_start.y + std::abs(r) * std::sin(kpt_start.psi + dir * M_PI / 2);
                Pose curve_center = {x_c, y_c, 0};

                double span_ang = kpt_end.segment_length / r;
                double cur_ang = computeAnglePose(curve_center, pos_s, pos_cur);

                if ((span_ang > 0) == (cur_ang > 0) && std::abs(span_ang) >= std::abs(cur_ang)) {
                    Pose v = {pos_cur.x - curve_center.x, pos_cur.y - curve_center.y, 0.0};
                    double e_y = -dir * (std::hypot(v.x, v.y) - std::abs(r));
                        
                    if (std::abs(e_y) <= track_width / 2 + slack) {
                        double d = std::abs(cur_ang) * std::abs(r);
                        double s = kpt_start.cum_length + d;
                        double e_psi = std::remainder(psi - (kpt_start.psi + cur_ang), 2 * M_PI);
                        
                        return FrenPose{s, e_y, e_psi};
                    }
                }
                
            }
     
        }

        // Adjust based on the 'line' parameter
        // ... Additional logic ...

        return cl_coord;
    }

  
Pose Track::localToGlobal(const FrenPose& cl_coord) {
        if (key_pts.empty()) {
            throw std::runtime_error("Track key points have not been defined");
        }

        double s = cl_coord.s;
        // Modulo operation to ensure s is within track length
        while (s < 0) s += track_length;
        while (s >= track_length) s -= track_length;

        double e_y = cl_coord.ey;
        double e_psi = cl_coord.epsi;

        // Find key point indices corresponding to current segment
        size_t seg_idx = 0;
        for (; seg_idx < key_pts.size() && s >= key_pts[seg_idx].cum_length; ++seg_idx);
        if (seg_idx > 0) --seg_idx;
        std::cout << "seg_idx =  " << seg_idx <<  std::endl;   
        size_t key_pt_idx_s = seg_idx;
        size_t key_pt_idx_f = std::min(seg_idx + 1, key_pts.size() - 1);
        
        
        const auto& kpt_start = key_pts[key_pt_idx_s];
        const auto& kpt_end = key_pts[key_pt_idx_f];

        double x, y, psi;
        double d = s - kpt_start.cum_length;  // Distance along current segment
        
        if (kpt_start.curvature == 0) {
            // Segment is a straight line
            std::cout << "straight " << seg_idx <<  std::endl;
            x = kpt_start.x + (kpt_end.x - kpt_start.x) * d / kpt_end.segment_length + e_y * std::cos(kpt_end.psi + M_PI / 2);
            y = kpt_start.y + (kpt_end.y - kpt_start.y) * d / kpt_end.segment_length + e_y * std::sin(kpt_end.psi + M_PI / 2);
            psi = wrapAngle(kpt_end.psi + e_psi);
        } else {
            // Curved segment
            
            double r = 1 / (kpt_end.curvature+1e-8);
            int dir = (r > 0) ? 1 : -1;
            
            double x_c = kpt_start.x + std::abs(r) * std::cos(kpt_start.psi + (dir * M_PI / 2));
            double y_c = kpt_start.y + std::abs(r) * std::sin(kpt_start.psi + (dir * M_PI / 2));
            
            double span_ang = d / std::abs(r);
            
            
            double psi_d = wrapAngle(kpt_start.psi + dir * span_ang);
            

            double ang_norm = wrapAngle(kpt_start.psi + dir * M_PI / 2);

            int ang_norm_sign = (ang_norm > 0) ? 1 : -1;
            double ang = -ang_norm_sign * (M_PI - std::abs(ang_norm));
            
            x = x_c + (std::abs(r) - dir * e_y) * std::cos(ang + dir * span_ang);
            y = y_c + (std::abs(r) - dir * e_y) * std::sin(ang + dir * span_ang);
            psi = wrapAngle(psi_d + e_psi);
        }

        return Pose{x, y, psi};
    }




// int main() {
//     Track track;
//     track.addSegment(0, 0, M_PI/2, 10, 0.23);
//     // Add more segments as needed
//     track.plot();

//     return 0;
// }
