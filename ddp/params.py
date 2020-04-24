import numpy as np

# params used for the inverted pendulum system

m = 1.4  # mass of quadrotor (kg)
L = 0.3             # length from center of mass to point of thrust (meters)
gr = 9.81  # gravity (m/s^2)
I = m * L ** 2
b = 1

states = 2  # theta and thetadot
total_time = 1  # total time duration (s)
dt = 0.01  # discretization timestep

timesteps = int(total_time / dt)  # total timesteps

# ddp parameters
num_iter = 200  # optimization iterations
num_controllers = 1

Q_f_ddp = np.diag([100, 50])
Q_r_ddp = Q_f_ddp

R_ddp = 0.0001 * np.eye(num_controllers)
gamma = 0.5         # how much we account for du in updating the control during optimization

max_u=2 # max torque
max_v = 8 # max angular velocity

# goal state
xf = np.zeros([states,1])
xf[0, 0] = np.pi
xf[1, 0] = 0
