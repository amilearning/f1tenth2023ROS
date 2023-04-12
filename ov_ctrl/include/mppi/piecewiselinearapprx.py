import numpy as np
from scipy.optimize import minimize, LinearConstraint
import matplotlib.pyplot as plt

import time


import numpy as np
from scipy.optimize import minimize, LinearConstraint

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
    x_segments = np.split(x, seg_len)
    x_approx = []
    for i in range(n):
        x_approx += [np.mean(x_segments[i])] * len(x_segments[i])
    x_approx.append(x[-1])
    
    return np.array(x_approx)





def piecewise_linear_approximation(x, n, min_segment_length):
    # Define the objective function to minimize the mean squared error
    def objective(params):
        seg_len = np.round(np.cumsum(params)*len(x)).astype(int)
        x_segments = np.split(x, seg_len)
        x_approx = []
        for i in range(n):
            slope, intercept = np.polyfit(range(len(x_segments[i])), x_segments[i], 1)
            x_approx += list(slope*np.arange(len(x_segments[i])) + intercept)
        # x_approx.append(x[-1])
        return np.mean((x - x_approx)**2)

    # Set the initial guess for the segment lengths
    init_guess = np.ones(n-1) / n

    # Define the linear constraint to ensure the segment lengths sum up to 1
    A = np.ones((1, n-1))
    b = [1.0]
    linear_constraint = LinearConstraint(A, lb=b, ub=b)

    # Define the nonlinear constraint to ensure the segment lengths are greater than the minimum threshold
    def nonlin_constraint(params):
        seg_len = np.round(np.cumsum(params)*len(x)).astype(int)
        return min_segment_length - np.min(np.diff(np.concatenate([[0], seg_len])))


    # Minimize the objective function subject to the linear and nonlinear constraints
    res = minimize(objective, init_guess, method='SLSQP', constraints=[linear_constraint, {'type': 'ineq', 'fun': nonlin_constraint}])

    # Compute the segment endpoints and fit linear functions to each segment
    seg_len = np.round(np.cumsum(res.x)*len(x)).astype(int)
    x_segments = np.split(x, seg_len)
    x_approx = []
    for i in range(n):
        slope, intercept = np.polyfit(range(len(x_segments[i])), x_segments[i], 1)
        x_approx += list(slope*np.arange(len(x_segments[i])) + intercept)
    x_approx.append(x[-1])
    
    return np.array(x_approx)

# Generate a test array
x = np.linspace(0, 5, 50)

y = np.sin(x) + 0.1*np.random.randn(len(x))
y[int(len(x)/2):] = 5

# Approximate the array with piecewise linear segments
n_segments = 5

start = time.process_time()
y_approx = piecewise_constant_approximation(y, n_segments,2)
print(time.process_time() - start)

# Plot the original array and the piecewise linear approximation
plt.plot(x, y,'*', label='Original')
plt.plot(x, y_approx[:-1],'^', label='Approximation')
plt.legend()
plt.show()