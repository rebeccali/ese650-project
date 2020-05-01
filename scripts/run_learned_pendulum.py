from environments import learned_pendulum
import gym
import numpy as np
import sys
import torch

import environments
from environments.learned_pendulum import LearnedPendulumEnv
from symplectic.analysis import simulate_control, simulate_models, get_all_models, get_prediction_error


EXPERIMENT_DIR = '../experiment_single_embed/'
sys.path.append(EXPERIMENT_DIR)

# %%
DPI = 100
FORMAT = 'pdf'


def get_args():
    """ Arguments to fetch a model. Must match existing trained models exactly."""
    return {'num_angle': 1, # Number of generalized coordinates
            'nonlinearity': 'tanh',  # NN nonlinearity
            'name': 'pend',  # name of environment
            'seed': 0,
            'save_dir': './{}'.format(EXPERIMENT_DIR),
            'fig_dir': './figures',
            'num_points': 2,  # number of evaluation points by ode solver, including initial point
            'gpu': 0,
            'solver': 'rk4',
            'env': 'MyPendulum-v0' # Name of the gym environment
            }

class ObjectView(object):
    def __init__(self, d): self.__dict__ = d


def main(args):
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    base_ode_model, naive_ode_model, symoden_ode_model, symoden_ode_struct_model = get_all_models(args, device, verbose=False)
    env = LearnedPendulumEnv(model=symoden_ode_struct_model, device=device)
    x0 = torch.tensor([1,2])
    u0 = torch.tensor([1])
    A,B = env.state_control_transition(x0, u0)
    print('A', A)
    print('B', B)

if __name__ == "__main__":
    args = ObjectView(get_args())

    main(args)