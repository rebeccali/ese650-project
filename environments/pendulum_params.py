import numpy as np
# TODO: convert these to params files

# params used for the inverted pendulum system
m = 1.4  # mass of quadrotor (kg)
L = 0.3  # length from center of mass to point of thrust (meters)
gr = 9.81  # gravity (m/s^2)
I = m * L ** 2
b = 1
max_torque = 1.0
max_speed = 8
states = 2  # theta and thetadot
num_controllers = 1
total_time = 1  # total time duration (s)
dt = 0.01  # discretization timestep
timesteps = int(total_time / dt)  # total timesteps

# goal state
xf = np.zeros([states, 1])
xf[0, 0] = np.pi
xf[1, 0] = 0

# ddp parameters
num_iter = 50  # optimization iterations
# TODO: fix this so learned_pendulum doesn't have to use this I guess
Q_f_ddp = np.diag([100, 1])
Q_r_ddp = np.zeros([states, states])


R_ddp = 0.1 * np.eye(num_controllers)
gamma = 0.5  # how much we account for du in updating the control during optimization

