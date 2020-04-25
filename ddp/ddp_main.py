""" main file for quadrotor tracking with ddp controller """

import params
from ddp_functions import *
import matplotlib.pyplot as plt
from pendulum import PendulumEnv

import pdb

if __name__ == "__main__":

    ################################ system specific stuff ###################################

    sys = PendulumEnv()

    ################################################################################################

    num_iter = params.num_iter

    u = np.zeros([params.num_controllers, params.timesteps-1])
    du = np.zeros([params.num_controllers, params.timesteps-1])
    x = np.zeros([params.states, params.timesteps])

    costvec = []

    for i in range(num_iter):

        u_opt = ddp(sys, x, u)

        # simulate the real system over the prediction time horizon, this uses the step function from the system
        x_new, cost = apply_control(sys, u_opt)

        # update state and control trajectories
        x = x_new
        u = u_opt

        # reset the system so that the next optimization step starts from the correct initial state
        sys.reset()
        costvec.append(-cost)

        # reset the system so that the next optimization step starts from the correct initial state
        sys.reset()
        
        print('iteration: ', i, "cost: ", -cost)

    xf = sys.goal
    x = np.asarray(x)
    u = np.asarray(u)
    costvec = np.asarray(costvec)

    plt.figure(1)

    plt.subplot(211)
    plt.plot(x[0, :])
    plt.plot(xf[0]*np.ones([params.timesteps, ]), 'r')
    plt.title('theta')

    plt.subplot(212)
    plt.plot(x[1, :])
    plt.plot(xf[1] * np.ones([params.timesteps, ]), 'r')
    plt.title('thetadot')

    plt.figure(3)
    plt.plot(costvec[:, 0, 0])
    plt.title('cost over iterations')

    plt.figure(4)
    plt.plot(u[0, :].T)
    plt.title('u opt output')


    plt.show()

    pdb.set_trace()

