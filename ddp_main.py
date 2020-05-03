""" main file for quadrotor tracking with ddp controller """

from ddp.ddp_functions import apply_control, ddp
import matplotlib.pyplot as plt
import gym
import environments
import argparse
import numpy as np

import pdb


def main(args):

    if args.env in environments.learned_models:
        # TODO make this less manual
        if args.env == 'LearnedPendulum-v0':
            from environments.learned_pendulum import LearnedPendulumEnv
            env = LearnedPendulumEnv(model_type='structure')
        else:
            raise RuntimeError('Do not recognized learned model %s ' % args.env)
    else:
        env = gym.make(args.env)

    num_iter = env.num_iter
    if args.test:
        num_iter = 3
        print('Running in test mode.')
    print('Using %s environment for %d iterations' % (args.env, num_iter))
    costvec, u, x, xf = run_ddp(env, num_iter)
    env.plot(xf, x, u, costvec)
    if not args.test:
        plt.show()


def run_ddp(env, num_iter):
    """Run DDP on environment for num_iter timesteps
        Returns:
            costvec(nparray): costs at each timestep
            u(nparray): the control trajectory
            x(nparray): the state trajectory
            xf(nparray): the system goal
    """
    x = np.zeros([env.states, env.timesteps])
    u = np.zeros([env.num_controllers, env.timesteps - 1])
    costvec = []
    for i in range(num_iter):
        u_opt = ddp(env, x, u)
        x_new, cost = apply_control(env, u_opt)

        # update state and control trajectories
        x = x_new
        u = u_opt

        # reset the system so that the next optimization step starts from the correct initial state
        costvec.append(-cost)

        # reset the system so that the next optimization step starts from the correct initial state
        env.reset()

        print('iteration: ', i, "cost: ", -cost)
    xf = env.goal
    x = np.asarray(x)
    u = np.asarray(u)
    costvec = np.asarray(costvec)
    return costvec, u, x, xf


if __name__ == "__main__":
    envs = ['DDP-Pendulum-v0', 'Simple-Quad-v0', 'LearnedPendulum-v0']
    # Parse Command Line Arguments
    # e.g. python ddp_main.py --test
    parser = argparse.ArgumentParser(description='Run DDP')
    parser.add_argument('--test', action='store_true', help='Run as test')
    parser.add_argument('--env', help='Which environment to use.', default=envs[0], choices=envs)
    args = parser.parse_args()

    main(args)
