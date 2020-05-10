import numpy as np

# quad model parameters
m = 1  # mass of quadrotor (kg)
L = 0.25  # length from center of mass to point of thrust (meters)
J = 8.1e-3  # moments of inertia in (kg*m^2)


gr = 9.81  # gravity (m/s^2)
states = 6  # number of states
num_controllers = 2
total_time = 10  # total time duration (s)
dt = 0.01  # discretization timestep

timesteps = int(total_time / dt)  # total timesteps

xf = np.zeros([states, 1])

xf[0, 0] = 1.5 # xpos
xf[1, 0] = 2 # ypos

max_speed = 100


# ddp parameters
num_iter = 50  # optimization iterations
Q_r_ddp = np.diag([0.1, 0.1, 0.1, 0.1, 10, 0.1])
Q_f_ddp = np.diag([100, 100, 10, 10, 50, 5])
R_ddp = 0.1 * np.eye(num_controllers)
gamma = 0.5  # how much we account for du in updating the control during optimization
