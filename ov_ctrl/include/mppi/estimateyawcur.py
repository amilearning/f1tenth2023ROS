import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize, LinearConstraint
import matplotlib.pyplot as plt
import time


def piecewise_constant_approximation(x, n, min_segment_length):
    # Define the objective function to minimize the mean squared error
    def objective(params):
        seg_len = np.round(np.cumsum(params)*len(x)).astype(int)
        x_segments = np.split(x, seg_len)
        x_approx = []
        for i in range(n):
            x_approx += [np.mean(x_segments[i])] * len(x_segments[i])
        # x_approx.append(x[-1])
        return np.mean((x - x_approx)**2)

    # Set the initial guess for the segment lengths
    init_guess = np.ones(n-1) / n

    # Define the linear constraint to ensure the segment lengths sum up to 1
    A = np.ones((1, n-1))
    b = [1.0]
    linear_constraint = LinearConstraint(A, lb=b, ub=b)

    # Define the nonlinear constraint to ensure the minimum length of each segment is greater than the threshold
    def nonlin_constraint(params):
        seg_len = np.round(np.cumsum(params)*len(x)).astype(int)
        return min_segment_length - np.min(np.diff(np.concatenate([[0], seg_len])))

    # Minimize the objective function subject to the linear and nonlinear constraints
    res = minimize(objective, init_guess, method='SLSQP', constraints=[linear_constraint, {'type': 'ineq', 'fun': nonlin_constraint}])

    # Compute the segment endpoints and piecewise constant approximation for each segment
    seg_len = np.round(np.cumsum(res.x)*len(x)).astype(int)
    seg_len = np.append(seg_len,len(x)-1)
    
    x_segments = np.split(x, seg_len)
    x_approx = []
    for i in range(n):
        x_approx += [np.mean(x_segments[i])] * len(x_segments[i])
    x_approx.append(x[-1])
    seg_len = np.insert(seg_len,0,0)
    return np.array(x_approx), seg_len, x_segments
    

def estimate_yaw_and_curvature(x, y, s=0.1):
    # Smooth x and y using a UnivariateSpline with smoothing factor s
    t = np.arange(len(x))
    spl_x = UnivariateSpline(t, x, k=4, s=s)
    spl_y = UnivariateSpline(t, y, k=4, s=s)
    x_smooth = spl_x(t)
    y_smooth = spl_y(t)

    # Compute the derivatives of x and y
    dx_dt = spl_x.derivative(n=1)(t)
    dy_dt = spl_y.derivative(n=1)(t)

    # Compute the yaw
    yaw = np.arctan2(dy_dt, dx_dt)

    # Compute the curvature
    ddx_dt = spl_x.derivative(n=2)(t)
    ddy_dt = spl_y.derivative(n=2)(t)
    curvature = (dx_dt * ddy_dt - dy_dt * ddx_dt) / (dx_dt**2 + dy_dt**2)**1.5

    # Smooth the yaw and curvature using a UnivariateSpline with smoothing factor s
    spl_yaw = UnivariateSpline(t, yaw, k=4, s=s)
    spl_curvature = UnivariateSpline(t, curvature, k=4, s=s)
    yaw_smooth = spl_yaw(t)
    curvature_smooth = spl_curvature(t)

    dx = np.gradient(x)
    dy = np.gradient(y)
    ds = np.sqrt(dx**2 + dy**2)
    s = np.cumsum(ds)

    return (x,y,yaw_smooth,s,curvature_smooth)


def get_keypts(x,y,n_segments):
    
    (x_,y_,psi_,cum_s,curvature_smooth) = estimate_yaw_and_curvature(x, y, s=0.1)
    # animate_path_yaw_curvature(x, y, yaw_smooth, curvature_smooth)
    cur_pw_const,  seg_len, x_segments= piecewise_constant_approximation(curvature_smooth, n_segments,2)
    track_key_pts = np.zeros((n_segments+1, 6))
    track_key_pts[0, 0] = x_[0]
    track_key_pts[0, 1] = y_[0]
    track_key_pts[0, 2] = psi_[0]
    for i in range(1,n_segments+1):
        idx = seg_len[i]
        prev_idx = seg_len[i-1]
        track_key_pts[i] = np.array([x_[idx], y_[idx], psi_[idx], cum_s[idx], cum_s[idx]-cum_s[prev_idx], cur_pw_const[prev_idx]])
    return track_key_pts


# Example usage
# x = np.linspace(0, 10, 1000)
# y = np.sin(x)
# n_segments = 20
# key_pts = get_keypts(x,y,n_segments)

# def find_keypts(x,y,key_pts):

# import matplotlib.pyplot as plt
# plt.plot(cur_pw_const)
# plt.plot(curvature_smooth)
# plt.plot(seg_len,track_key_pts[:,5],'*')
