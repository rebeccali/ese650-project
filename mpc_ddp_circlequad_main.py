""" main file for quadrotor tracking with mpc ddp controller """

from ddp.mpc_ddp_functions_circlequad import *
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

    sys = gym.make('MPC-DDP-Circle-Quad-v0')

    ################################################################################################

    ################################ MPC stufff ####################################################

    current_time = 0
    index = 0

    ################################################################################################
    num_iter = sys.num_iter

    if args.test:
        num_iter = 3

    costvec = []
    x = []
    x.append(sys.state)
    u = []

    x1 = sys.state # set intial state

    while current_time < (sys.total_time-sys.dt):

        x_ddp = np.zeros([sys.states, int(sys.timesteps)])
        x_ddp[:, 0] = x1
        u_ddp = np.zeros([sys.num_controllers, int(sys.timesteps)])

        for i in range(num_iter):
            u_opt = ddp(sys, x_ddp, u_ddp)

            x_new, cost = apply_control(sys, u_opt)

            # update state and control trajectories
            x_ddp = x_new
            u_ddp = u_opt

            sys.reset(reset_state=x1)

        # apply first control from the sequence and step one timestep
        x1, c1 = sys.step(u_ddp[:, 0])

        # save the current state and control and reset intial state for the system
        # x[:, index + 1] = x1
        # u[:, index] = u_ddp[:, 0]

        x.append(x1)
        u.append(u_ddp[:, 0])
        costvec.append(-c1)

        sys.reset(reset_state=x1)

        # update the goal trajectory for MPC
        sys.set_goal()

        pdb.set_trace()

        current_time += sys.dt
        index += 1

        print('iteration: ', index, "cost: ", -c1)

    xf = sys.total_goal
    x = np.asarray(x)
    u = np.asarray(u)
    costvec = np.asarray(costvec)

    # sys.plot(xf, x.T, u.T, costvec)

    if not args.test:
        plt.show()

