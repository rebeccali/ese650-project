""" Compares trajectories between learned and ground truth environment """

import argparse
import gym
import matplotlib.pyplot as plt
import sys

sys.path.append('./') # TODO Hacky, remove by reorganizing the python folder structure.

from environments.learned_pendulum import LearnedPendulumEnv
from utils.plot_utils import plot_trajectories, rollout_models


def main(args):
    learned_env = LearnedPendulumEnv(model_type='structure')
    naive_env = LearnedPendulumEnv(model_type='naive')
    ground_truth_env = gym.make('DDP-Pendulum-v0')
    horizon = args.horizon
    traj_true, traj_learned, traj_naive, ts = rollout_models(ground_truth_env, learned_env, naive_env, horizon)
    plot_trajectories(ts, traj_true, traj_learned, traj_naive, ('Pendulum'))
    if not args.test:
        plt.show()


if __name__ == "__main__":
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser(description='Test Dynamics')
    parser.add_argument('--horizon', help='Simulation Horizon in seconds', type=float, default=1.0)
    parser.add_argument('--test', action='store_true', help='Run as test')
    args = parser.parse_args()

    main(args)
