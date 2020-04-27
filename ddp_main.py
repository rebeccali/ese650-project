""" main file for quadrotor tracking with ddp controller """

from ddp.ddp_functions import *
from ddp import ddp_params
import matplotlib.pyplot as plt
import gym
import environments
import argparse
import numpy as np

import pdb

if __name__ == "__main__":
    # Parse Command Line Arguments
    # e.g. python ddp_main.py --test
    parser = argparse.ArgumentParser(description='Run DDP')
    parser.add_argument('--test', action='store_true',
                        help='Run as test')
    args = parser.parse_args()

    ################################ system specific stuff ###################################

    sys = gym.make('DDP-Pendulum-v0')
    ################################################################################################

    num_iter = ddp_params.num_iter
    if args.test:
        num_iter = 3
    x = np.zeros([ddp_params.states, ddp_params.timesteps])
    u = np.zeros([ddp_params.num_controllers, ddp_params.timesteps - 1])

    costvec = []

    for i in range(num_iter):
        u_opt = ddp(sys, x, u)
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
    plt.plot(xf[0] * np.ones([ddp_params.timesteps, ]), 'r')
    plt.title('theta')

    plt.subplot(212)
    plt.plot(x[1, :])
    plt.plot(xf[1] * np.ones([ddp_params.timesteps, ]), 'r')
    plt.title('thetadot')

    plt.figure(3)
    plt.plot(costvec[:, 0, 0])
    plt.title('cost over iterations')

    plt.figure(4)
    plt.plot(u[0, :].T)
    plt.title('u opt output')

    plt.show()
    if not args.test:
        plt.show()
