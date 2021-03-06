""" main file for quadrotor tracking with ddp controller """

from ddp.ddp_functions import *
import gym
import environments
from environments import quadrotor_params
import matplotlib.pyplot as plt
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

    # quad = SimpleQuadEnv()
    quad = gym.make('Simple-Quad-v0')

    # set desired goal state
    xf = np.zeros([quadrotor_params.states, 1])
    xf[0, 0] = -1
    xf[1, 0] = 1
    xf[2, 0] = 0.5

    quad.set_goal(xf)

    num_iter = quadrotor_params.num_iter
    if args.test:
        num_iter = 3
    u = np.zeros([quadrotor_params.num_controllers, quadrotor_params.timesteps - 1])
    du = np.zeros([quadrotor_params.num_controllers, quadrotor_params.timesteps - 1])
    x = np.zeros([quadrotor_params.states, quadrotor_params.timesteps])

    costvec = []

    for i in range(num_iter):
        u_opt = ddp(quad, x, u)

        # simulate the real system over the prediction time horizon, this uses the step function from the system
        x_new, cost = apply_control(quad, u_opt)

        # update state and control trajectories
        x = x_new
        u = u_opt

        # reset the system so that the next optimization step starts from the correct initial state
        quad.reset()
        costvec.append(-cost)

        # reset the system so that the next optimization step starts from the correct initial state
        quad.reset()

        print('iteration: ', i, "cost: ", -cost)

    # translational states
    plt.figure(1)

    plt.subplot(231)
    plt.plot(x[0, :])
    plt.plot(xf[0] * np.ones([quadrotor_params.timesteps, ]), 'r')
    plt.title('x')

    plt.subplot(232)
    plt.plot(x[1, :])
    plt.plot(xf[1] * np.ones([quadrotor_params.timesteps, ]), 'r')
    plt.title('y')

    plt.subplot(233)
    plt.plot(x[2, :])
    plt.plot(xf[2] * np.ones([quadrotor_params.timesteps, ]), 'r')
    plt.title('z')

    plt.subplot(234)
    plt.plot(x[3, :])
    plt.plot(xf[3] * np.ones([quadrotor_params.timesteps, ]), 'r')
    plt.title('x dot')

    plt.subplot(235)
    plt.plot(x[4, :])
    plt.plot(xf[4] * np.ones([quadrotor_params.timesteps, ]), 'r')
    plt.title('y dot')

    plt.subplot(236)
    plt.plot(x[5, :])
    plt.plot(xf[5] * np.ones([quadrotor_params.timesteps, ]), 'r')
    plt.title('z dot')

    # rotational states
    plt.figure(2)

    plt.subplot(231)
    plt.plot(x[6, :])
    plt.plot(xf[6] * np.ones([quadrotor_params.timesteps, ]), 'r')
    plt.title('phi')

    plt.subplot(232)
    plt.plot(x[7, :])
    plt.plot(xf[7] * np.ones([quadrotor_params.timesteps, ]), 'r')
    plt.title('theta')

    plt.subplot(233)
    plt.plot(x[8, :])
    plt.plot(xf[8] * np.ones([quadrotor_params.timesteps, ]), 'r')
    plt.title('psi')

    plt.subplot(234)
    plt.plot(x[9, :])
    plt.plot(xf[9] * np.ones([quadrotor_params.timesteps, ]), 'r')
    plt.title('phi dot')

    plt.subplot(235)
    plt.plot(x[10, :])
    plt.plot(xf[10] * np.ones([quadrotor_params.timesteps, ]), 'r')
    plt.title('theta dot')

    plt.subplot(236)
    plt.plot(x[11, :])
    plt.plot(xf[11] * np.ones([quadrotor_params.timesteps, ]), 'r')
    plt.title('psi dot')

    # cost over iterations
    plt.figure(3)
    plt.plot(costvec)

    plt.figure(4)
    plt.subplot(411)
    plt.plot(u[0, :])

    plt.subplot(412)
    plt.plot(u[2, :])

    plt.subplot(413)
    plt.plot(u[2, :])

    plt.subplot(414)
    plt.plot(u[3, :])

    if not args.test:
        plt.show()
    # pdb.set_trace()
