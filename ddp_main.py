""" main file for quadrotor tracking with ddp controller """
import argparse
import matplotlib.pyplot as plt

from ddp.ddp_functions import run_ddp
from environments.utils import construct_env


def main(args):
    env = construct_env(args.env)

    num_iter = env.num_iter
    if args.test:
        num_iter = 3
        print('Running in test mode.')
    print('Using %s environment for %d iterations' % (args.env, num_iter))
    costvec, u, x, xf = run_ddp(env, num_iter)
    env.plot(xf, x, u, costvec)
    if not args.test:
        plt.show()


if __name__ == "__main__":
    envs = ['DDP-Pendulum-v0', 'Simple-Quad-v0', 'LearnedPendulum-v0', 'LearnedQuad-v0']
    # Parse Command Line Arguments
    # e.g. python ddp_main.py --test
    parser = argparse.ArgumentParser(description='Run DDP')
    parser.add_argument('--test', action='store_true', help='Run as test')
    parser.add_argument('--env', help='Which environment to use.', default=envs[0], choices=envs)
    args = parser.parse_args()

    main(args)
