""" main file for quadrotor tracking with mpc ddp controller """

from ddp.ddp_functions import run_ddp, run_mpc_ddp
import matplotlib.pyplot as plt
import gym
import environments
import argparse

import numpy as np
import pdb

def main(args):

    if args.env in environments.learned_models:
        # TODO(rebecca) make this less manual
        if args.env == 'LearnedPendulum-v0':
            from environments.learned_pendulum import LearnedPendulumEnv
            env = LearnedPendulumEnv(model_type='structure')
        elif args.env == 'LearnedQuad-v0':
            from environments.learned_quad import LearnedQuadEnv
            env = LearnedQuadEnv(model_type='structure')
        else:
            raise RuntimeError('Do not recognized learned model %s ' % args.env)
    else:
        env = gym.make(args.env)

    num_iter = env.num_iter
    opt_time = 0.1  # how to split the mpc total time horizon

    if args.test:
        num_iter = 3
        print('Running in test mode.')

    # print('Using %s environment for %d iterations' % (args.env, num_iter))
    costvec, u, x, xf = run_mpc_ddp(env, num_iter, opt_time)
    env.plot(xf, x.T, u.T, costvec)

    if not args.test:
        plt.show()

if __name__ == "__main__":

    # envs = ['MPC-DDP-Simple-Quad-v0']
    # envs = ['MPC-DDP-Pendulum-v0']
    envs = ['DDP-Pendulum-v0'] # slower than MPC-DDP-Pendulum-v0

    # Parse Command Line Arguments
    # e.g. python ddp_main.py --test
    parser = argparse.ArgumentParser(description='Run DDP')
    parser.add_argument('--test', action='store_true', help='Run as test')
    parser.add_argument('--env', help='Which environment to use.', default=envs[0], choices=envs)

    args = parser.parse_args()

    main(args)

