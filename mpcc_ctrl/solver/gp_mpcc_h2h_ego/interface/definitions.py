import numpy
import ctypes

name = "gp_mpcc_h2h_ego"
requires_callback = True
lib = "lib/libgp_mpcc_h2h_ego.so"
lib_static = "lib/libgp_mpcc_h2h_ego.a"
c_header = "include/gp_mpcc_h2h_ego.h"
nstages = 10

# Parameter             | Type    | Scalar type      | Ctypes type    | Numpy type   | Shape     | Len
params = \
[("x0"                  , "dense" , ""               , ctypes.c_double, numpy.float64, (190,   1),  190),
 ("xinit"               , "dense" , ""               , ctypes.c_double, numpy.float64, ( 14,   1),   14),
 ("all_parameters"      , "dense" , ""               , ctypes.c_double, numpy.float64, (610,   1),  610)]

# Output                | Type    | Ctypes type    | Numpy type   | Shape     | Len
outputs = \
[("x01"                 , ""               , ctypes.c_double, numpy.float64,     ( 19,),   19),
 ("x02"                 , ""               , ctypes.c_double, numpy.float64,     ( 19,),   19),
 ("x03"                 , ""               , ctypes.c_double, numpy.float64,     ( 19,),   19),
 ("x04"                 , ""               , ctypes.c_double, numpy.float64,     ( 19,),   19),
 ("x05"                 , ""               , ctypes.c_double, numpy.float64,     ( 19,),   19),
 ("x06"                 , ""               , ctypes.c_double, numpy.float64,     ( 19,),   19),
 ("x07"                 , ""               , ctypes.c_double, numpy.float64,     ( 19,),   19),
 ("x08"                 , ""               , ctypes.c_double, numpy.float64,     ( 19,),   19),
 ("x09"                 , ""               , ctypes.c_double, numpy.float64,     ( 19,),   19),
 ("x10"                 , ""               , ctypes.c_double, numpy.float64,     ( 19,),   19)]

# Info Struct Fields
info = \
[('it', ctypes.c_int),
 ('it2opt', ctypes.c_int),
 ('res_eq', ctypes.c_double),
 ('res_ineq', ctypes.c_double),
 ('rsnorm', ctypes.c_double),
 ('rcompnorm', ctypes.c_double),
 ('pobj', ctypes.c_double),
 ('dobj', ctypes.c_double),
 ('dgap', ctypes.c_double),
 ('rdgap', ctypes.c_double),
 ('mu', ctypes.c_double),
 ('mu_aff', ctypes.c_double),
 ('sigma', ctypes.c_double),
 ('lsit_aff', ctypes.c_int),
 ('lsit_cc', ctypes.c_int),
 ('step_aff', ctypes.c_double),
 ('step_cc', ctypes.c_double),
 ('solvetime', ctypes.c_double),
 ('fevalstime', ctypes.c_double),
 ('solver_id', ctypes.c_int * 8)
]

# Dynamics dimensions
#   nvar    |   neq   |   dimh    |   dimp    |   diml    |   dimu    |   dimhl   |   dimhu    
dynamics_dims = [
	(19, 14, 9, 61, 5, 5, 7, 3), 
	(19, 14, 9, 61, 9, 9, 7, 3), 
	(19, 14, 9, 61, 9, 9, 7, 3), 
	(19, 14, 9, 61, 9, 9, 7, 3), 
	(19, 14, 9, 61, 9, 9, 7, 3), 
	(19, 14, 9, 61, 9, 9, 7, 3), 
	(19, 14, 9, 61, 9, 9, 7, 3), 
	(19, 14, 9, 61, 9, 9, 7, 3), 
	(19, 14, 9, 61, 9, 9, 7, 3), 
	(19, 0, 9, 61, 9, 9, 7, 3)
]