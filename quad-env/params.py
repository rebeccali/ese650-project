import numpy as np

m           = 1              # mass of quadrotor (kg)
L           = 0.25           # length from center of mass to point of thrust (meters)
J           = np.zeros((3,3))# moments of inertia in (kg*m^2)
J[0,0]      = 8.1e-3
J[1,1]      = 8.1e-3
J[2,2]      = 14.2e-3
gr          = 9.81           # gravity (m/s^2)
states      = 12             # number of states
total_time  = 1.5            # total time duration (s)
dt          = 0.01           # discretization timestep
timesteps   = total_time/dt  # total timesteps 
