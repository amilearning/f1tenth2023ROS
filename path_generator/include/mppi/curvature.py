from scipy.interpolate import UnivariateSpline
import numpy as np


def compute_curvature(x, y):
    # Resample input data to ensure evenly spaced points
    t = np.arange(0, 1, 0.001)
    fx = UnivariateSpline(t, x)(t)
    fy = UnivariateSpline(t, y)(t)

    # Compute first and second derivatives
    dx = np.gradient(fx, t)
    ddx = np.gradient(dx, t)
    dy = np.gradient(fy, t)
    ddy = np.gradient(dy, t)

    # Compute curvature
    curvature = (dx * ddy - dy * ddx) / ((dx**2 + dy**2)**(3/2))

    return curvature, fx, fy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define a path (x, y)
t = np.linspace(0, 2*np.pi, 1000)
x = np.sin(t)
y = np.cos(2*t)

# Compute curvature
curvature, fx, fy = compute_curvature(x, y)

# Create figure and axis
fig, ax = plt.subplots()
ax.set_xlim((-1.5, 1.5))
ax.set_ylim((-1.5, 1.5))

# Create line object for path
line, = ax.plot([], [], lw=2)

# Create scatter object for curvature
scatter = ax.scatter([], [], c=[], cmap='coolwarm')

# Define update function for animation
def update(frame):
    # Get curvature values up to current frame
    curv = curvature[:frame+1]
    
    # Set data for path line
    line.set_data(fx[:frame+1], fy[:frame+1])
    
    # Set data for curvature scatter plot
    scatter.set_offsets(np.column_stack((fx[:frame+1], fy[:frame+1])))
    scatter.set_array(curv)

    return line, scatter

# Create animation object
ani = animation.FuncAnimation(fig, update, frames=len(curvature), interval=50, blit=True)

# Show animation
plt.show()
