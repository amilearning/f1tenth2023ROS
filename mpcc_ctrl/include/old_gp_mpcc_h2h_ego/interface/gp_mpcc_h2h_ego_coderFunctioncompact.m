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
% - outputs - column vector of length 190
function [outputs] = gp_mpcc_h2h_ego(x0, xinit, all_parameters)
    
    [output, ~, ~] = gp_mpcc_h2h_egoBuildable.forcesCall(x0, xinit, all_parameters);
    outputs = coder.nullcopy(zeros(190,1));
    outputs(1:19) = output.x01;
    outputs(20:38) = output.x02;
    outputs(39:57) = output.x03;
    outputs(58:76) = output.x04;
    outputs(77:95) = output.x05;
    outputs(96:114) = output.x06;
    outputs(115:133) = output.x07;
    outputs(134:152) = output.x08;
    outputs(153:171) = output.x09;
    outputs(172:190) = output.x10;
end
