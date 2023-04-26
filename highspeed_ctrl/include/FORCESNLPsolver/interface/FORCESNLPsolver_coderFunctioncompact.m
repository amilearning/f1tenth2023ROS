% FORCESNLPsolver : A fast customized optimization solver.
% 
% Copyright (C) 2013-2022 EMBOTECH AG [info@embotech.com]. All rights reserved.
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
% [OUTPUTS] = FORCESNLPsolver(INPUTS) solves an optimization problem where:
% Inputs:
% - xinit - matrix of size [5x1]
% - x0 - matrix of size [70x1]
% - all_parameters - matrix of size [20x1]
% - reinitialize - scalar
% Outputs:
% - outputs - column vector of length 70
function [outputs] = FORCESNLPsolver(xinit, x0, all_parameters, reinitialize)
    
    [output, ~, ~] = FORCESNLPsolverBuildable.forcesCall(xinit, x0, all_parameters, reinitialize);
    outputs = coder.nullcopy(zeros(70,1));
    outputs(1:7) = output.x01;
    outputs(8:14) = output.x02;
    outputs(15:21) = output.x03;
    outputs(22:28) = output.x04;
    outputs(29:35) = output.x05;
    outputs(36:42) = output.x06;
    outputs(43:49) = output.x07;
    outputs(50:56) = output.x08;
    outputs(57:63) = output.x09;
    outputs(64:70) = output.x10;
end
