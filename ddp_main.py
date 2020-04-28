""" main file for quadrotor tracking with ddp controller """

from ddp.ddp_functions import *
import matplotlib.pyplot as plt
import gym
import environments
import argparse
import numpy as np

import pdb


def main(args):
    sys = gym.make(args.env)
    num_iter = sys.num_iter
    if args.test:
        num_iter = 3
        print('Running in test mode.')
    print('Using %s environment for %d iterations' % (args.env, num_iter))
    x = np.zeros([sys.states, sys.timesteps])
    u = np.zeros([sys.num_controllers, sys.timesteps - 1])
    costvec = []
    for i in range(num_iter):
        u_opt = ddp(sys, x, u)
        x_new, cost = apply_control(sys, u_opt)

        # update state and control trajectories
        x = x_new
        u = u_opt

        # reset the system so that the next optimization step starts from the correct initial state
        costvec.append(-cost)

        # reset the system so that the next optimization step starts from the correct initial state
        sys.reset()

        print('iteration: ', i, "cost: ", -cost)
    xf = sys.goal
    x = np.asarray(x)
    u = np.asarray(u)
    costvec = np.asarray(costvec)
    sys.plot(xf, x, u, costvec)
    if not args.test:
        plt.show()


if __name__ == "__main__":
    envs = ['DDP-Pendulum-v0', 'Simple-Quad-v0']
    # Parse Command Line Arguments
    # e.g. python ddp_main.py --test
    parser = argparse.ArgumentParser(description='Run DDP')
    parser.add_argument('--test', action='store_true', help='Run as test')
    parser.add_argument('--env', help='Which environment to use.', default=envs[0], choices=envs)
    args = parser.parse_args()

    main(args)
