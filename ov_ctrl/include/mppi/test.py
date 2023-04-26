import casadi as ca
import numpy as np
from matplotlib import pyplot as plt
import time




sym_s = ca.SX.sym('s', 1)
# Makes sure s is within [0, track_length]
s_inits = np.array([[1,2,3,4]])
c_values = np.array([[2,-1,3,-2,-4]])
# Piecewise constant function mapping s to track curvature
pw_const_curvature = ca.pw_const(sym_s,s_inits, c_values)
pw_const_function = ca.Function('track_curvature', [sym_s], [pw_const_curvature])

s_lin_inits = np.array([[0.001,1,2,3,4,5]])
c_lin_values = np.array([[2,2,-1,3,-2,-4]])
pw_lin_curvature = ca.pw_lin(sym_s, s_lin_inits, c_lin_values)

pw_lin_function = ca.Function('track_curvature', [sym_s], [pw_lin_curvature])

s_set = np.linspace(0,5,50000)
# for s in s_set:
#   print("s = ",s, " : ", pw_lin_function(s))
start_time = time.time()
a_lin = [pw_lin_function(s) for s in s_set]
print("linear--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
a_cont = [pw_const_function(s) for s in s_set]
print("const--- %s seconds ---" % (time.time() - start_time))
# a_lin_np = np.array(a_lin).reshape(-1)
# a_cont_np = np.array(a_cont).reshape(-1)
# plt.plot(s_set,a_lin_np,'*')
# plt.plot(s_set,a_cont_np,'^')
# plt.show()