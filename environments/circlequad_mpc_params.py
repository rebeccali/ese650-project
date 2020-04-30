import numpy as np
import pdb

# quad model parameters
m = 1  # mass of quadrotor (kg)
L = 0.25  # length from center of mass to point of thrust (meters)
J = np.zeros((3, 3))  # moments of inertia in (kg*m^2)
J[0, 0] = 8.1e-3
J[1, 1] = 8.1e-3
J[2, 2] = 14.2e-3

gr = 9.81  # gravity (m/s^2)
states = 12  # number of states
num_controllers = 4
total_time = 4  # total time duration (s)
dt = 0.01  # discretization timestep

total_timesteps = int(total_time / dt)
timesteps = int(total_timesteps / 10)  # total timesteps

# setting goal trajectory to follow:
r = 1
th1 = np.arange(0, 2 * np.pi, (2 * np.pi / total_timesteps))
th2 = th1

th = np.array([[th1], [th2]])

pdb.set_trace()

xf = np.zeros([states, total_timesteps])

xf[0, 0] = 1.5  # xpos
xf[1, 0] = 2  # ypos
xf[2, 0] = 3  # zpos

# ddp parameters
num_iter = 50  # optimization iterations
Q_r_ddp = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 10, 10, 10, 0.1, 0.1, 0.1])
Q_f_ddp = np.diag([100, 100, 100, 10, 10, 10, 50, 50, 5, 5, 5, 5])
R_ddp = 0.1 * np.eye(num_controllers)
gamma = 0.5  # how much we account for du in updating the control during optimization
