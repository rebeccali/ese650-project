""" main file for quadrotor tracking with ddp controller """

from ddp.ddp_functions import run_ddp
import matplotlib.pyplot as plt
import gym
import environments
import argparse


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
