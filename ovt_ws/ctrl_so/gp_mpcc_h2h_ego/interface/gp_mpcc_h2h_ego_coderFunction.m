% gp_mpcc_h2h_ego : A fast customized optimization solver.
% 
% Copyright (C) 2013-2023 EMBOTECH AG [info@embotech.com]. All rights reserved.
% 
% 
% This software is intended for simulation and testing purposes only. 
% Use of this software for any commercial purpose is prohibited.
% 
% This program is distributed in the hope that it will be useful.
% EMBOTECH makes NO WARRANTIES with respect to the use of the software 
% without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
% PARTICULAR PURPOSE. 
% 
% EMBOTECH shall not have any liability for any damage arising from the use
% of the software.
% 
% This Agreement shall exclusively be governed by and interpreted in 
% accordance with the laws of Switzerland, excluding its principles
% of conflict of laws. The Courts of Zurich-City shall have exclusive 
% jurisdiction in case of any dispute.
% 
% [OUTPUTS] = gp_mpcc_h2h_ego(INPUTS) solves an optimization problem where:
% Inputs:
% - x0 - matrix of size [190x1]
% - xinit - matrix of size [14x1]
% - all_parameters - matrix of size [610x1]
% Outputs:
% - x01 - column vector of length 19
% - x02 - column vector of length 19
% - x03 - column vector of length 19
% - x04 - column vector of length 19
% - x05 - column vector of length 19
% - x06 - column vector of length 19
% - x07 - column vector of length 19
% - x08 - column vector of length 19
% - x09 - column vector of length 19
% - x10 - column vector of length 19
function [x01, x02, x03, x04, x05, x06, x07, x08, x09, x10] = gp_mpcc_h2h_ego(x0, xinit, all_parameters)
    
    [output, ~, ~] = gp_mpcc_h2h_egoBuildable.forcesCall(x0, xinit, all_parameters);
    x01 = output.x01;
    x02 = output.x02;
    x03 = output.x03;
    x04 = output.x04;
    x05 = output.x05;
    x06 = output.x06;
    x07 = output.x07;
    x08 = output.x08;
    x09 = output.x09;
    x10 = output.x10;
end
