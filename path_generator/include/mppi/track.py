#!/usr/bin/env python3

import numpy as np
import casadi as ca


class Track():
    def __init__(self):
        self.track_width = None
        self.track_s = None
        self.track_curv = None
        
    def initialize_dyn(self,  track_s = None):
        if track_s is not None:
            self.track_s = track_s
            self.track_length = self.track_s[-1]

        return

    def initialize(self, track_width=None, track_s = None, track_curv = None, slack=None, cl_segs=None):
        if track_width is not None:
            self.track_width = track_width

        if track_s is not None:
            self.track_s = track_s
            self.track_length = self.track_s[-1]
        if track_curv is not None:
            self.track_curv =track_curv

        if slack is not None:
            self.slack = slack
        if cl_segs is not None:
            self.cl_segs = cl_segs

        return


    def get_curvature_casadi_fn_dynamic(self):
        sym_s = ca.SX.sym('s', 1)
        # track_length = ca.SX.sym('track_length', 1)
        ## key_pts = [cumulative_lenth, signed_curvature]
        key_pts = ca.SX.sym('key_pts', 4, 2)
        
        # Makes sure s is within [0, track_length]
        sym_s_bar = ca.mod(ca.mod(sym_s, self.track_length) + self.track_length, self.track_length)
        # Piecewise constant function mapping s to track curvature
        pw_const_curvature = ca.pw_const(sym_s_bar, key_pts[:-1, 0], key_pts[:, 1])
        return ca.Function('track_curvature_dyn', [sym_s, key_pts], [pw_const_curvature])
    
    def get_track_width_casadi_fn_dynamic(self):
        sym_s = ca.SX.sym('s', 1)
        # Makes sure s is within [0, track_length]
        sym_s_bar = ca.mod(ca.mod(sym_s, self.track_length) + self.track_length, self.track_length)
        width_key_pts = ca.SX.sym('width_key_pts', 4, 2)
        pw_const_track_width = ca.pw_const(sym_s_bar,width_key_pts[:-1,0], width_key_pts[:,1])
        pw_const_function = ca.Function('track_curvature', [sym_s, width_key_pts], [pw_const_track_width])
        return pw_const_function 
    

    def get_track_width_casadi_fn(self):
        sym_s = ca.SX.sym('s', 1)
        # Makes sure s is within [0, track_length]
        sym_s_bar = ca.mod(ca.mod(sym_s, self.track_length) + self.track_length, self.track_length)
        pw_const_track_width = ca.pw_const(sym_s_bar,self.track_s[:-1], self.track_width)
        pw_const_function = ca.Function('track_curvature', [sym_s], [pw_const_track_width])
        return pw_const_function 

    def get_curvature_casadi_fn(self):
        sym_s = ca.SX.sym('s', 1)
        # Makes sure s is within [0, track_length]
        sym_s_bar = ca.mod(ca.mod(sym_s, self.track_length) + self.track_length, self.track_length)
        pw_const_curvature = ca.pw_const(sym_s_bar,self.track_s[:-1], self.track_curv)
        pw_const_function = ca.Function('track_curvature', [sym_s], [pw_const_curvature])
        return pw_const_function 
        
    def wrap_angle_ca(theta):
        return ca.if_else(theta < -ca.pi, 2 * ca.pi + theta, ca.if_else(theta > ca.pi, theta - 2 * ca.pi, theta))

def sign(a):
    if a >= 0:
        res = 1
    else:
        res = -1

    return res


"""
Helper function for computing the angle between the vectors point_1-point_0
and point_2-point_0. All points are defined in the inertial frame
Input:
    point_0: position of the intersection point (np.array of size 2)
    point_1, point_2: defines the intersecting lines (np.array of size 2)
Output:
    theta: angle in radians
"""


def compute_angle(point_0, point_1, point_2):
    v_1 = point_1 - point_0
    v_2 = point_2 - point_0

    dot = v_1.dot(v_2)
    det = v_1[0] * v_2[1] - v_1[1] * v_2[0]
    theta = np.arctan2(det, dot)

    return theta
