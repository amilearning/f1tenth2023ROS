
#pragma once


#include <fstream>
#include <mutex>
#include <vector> 
#include <cmath>

#include <tuple>
#include <algorithm>

#include <thread>
#include "trajectory.h"
#include "state.h"


class VehicleDynamics {
public:
    VehicleDynamics(double dt_) : dt(dt_) {
        mass = 2;
        width = 0.25;
        L = 0.33;
        Lr = 0.115;
        Lf = 0.115;        
        Caf = mass * Lf/L * 0.5 * 0.35 * 180/M_PI;
        Car = mass * Lr/L * 0.5 * 0.35 * 180/M_PI;
        Izz = Lf*Lr*mass;
        g = 9.814195;
        h = 0.15;
        Bp = 1.0;
        Cp = 1.25;
        mu = 0.8;
        Df = mu*mass*g*Lr/(Lr+Lf);
        Dr = mu*mass*g*Lf/(Lr+Lf);
        vx_low_bound = 0.1;
    }
     
    std::vector<VehicleState> dynamics_propogate(const VehicleState& state, int time_index){
        VehicleState init_state = state;
        std::vector<VehicleState> out_states; 
        for(int i=0;i<time_index; i++){
            out_states.push_back(init_state);
            init_state =dynamicsUpdate(init_state);
        }


        return out_states;
    }

    void compute_slip(double & alpha_f, double & alpha_r, const VehicleState & state){
        double clip_vx = std::max(state.vx,vx_low_bound);
        alpha_f = state.delta - std::atan2(state.vy+Lf*state.wz,clip_vx);
        alpha_r = -std::atan2(state.vy-Lr*state.wz,clip_vx);         
    }
    
    VehicleState dynamicsUpdate(const VehicleState & state){
        VehicleState tmp_state = state;
            double axb = tmp_state.accel;
            double delta = tmp_state.delta;   
            //  Pejekap 
            double alpha_f_p, alpha_r_p;  
            compute_slip(alpha_f_p, alpha_r_p, tmp_state);
            double Fyf = Df*sin(Cp*std::atan(Bp*alpha_f_p));
            double Fyr = Dr*sin(Cp*std::atan(Bp*alpha_r_p));
            // # x: s(0), ey(1), epsi(2), vx(3), vy(4), wz(5)        
            double curs = state.k;
          
            
            
        
            double s = state.s;
            double ey = state.ey;       
            double epsi = state.epsi;        
            double vx = state.vx;
            double vy = state.vy;
            double wz = state.wz;       
            
            tmp_state.s = s + dt * ( (vx * cos(epsi) - vy * sin(epsi)) / (1 - curs * ey) );
           
            
            tmp_state.ey  = ey + dt * (vx * sin(epsi) + vy * cos(epsi));
            
            tmp_state.epsi  = epsi + dt * ( wz - (vx * cos(epsi) - vy * sin(epsi)) / (1 - curs * ey) * curs );
            
            tmp_state.vx = std::max(  vx + dt * (axb - 1 / mass * Fyf * sin(delta) + wz*vy), vx_low_bound);     
            tmp_state.vy = vy + dt * (1 / mass * (Fyf * cos(delta) + Fyr) - wz * vx);
            tmp_state.wz = wz + dt * (1 / Izz *(Lf * Fyf * cos(delta) - Lr * Fyr) );
        
        return tmp_state;
    }
//    # state x : s(0), ey(1), epsi(2), vx(3), vy(4), wz(5)
//             self.state_dim = 6
//             # input u: accel_x(0), delta(1) 
    

    

    void updateTrajectory(const Trajectory& ref_){
        ref_traj = ref_;
    }

    double vx_low_bound;
    double dt;
    VehicleState cur_state;
    double mass;
    double width;
    double L;
    double Lf;
    double Lr; 
    double Caf; 
    double Car; 
    double Izz;
    double g;
    double h;
    double Bp;
    double Cp;
    double mu;
    double Df;
    double Dr;
    Trajectory ref_traj;


};