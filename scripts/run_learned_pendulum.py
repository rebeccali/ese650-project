
import gym
import numpy as np
import sys
import torch
import os
sys.path.append('./')
from environments.learned_pendulum import LearnedPendulumEnv
from symplectic.analysis import simulate_control, simulate_models, get_all_models, get_prediction_error


EXPERIMENT_DIR = 'experiment_single_embed/'
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
    env.dt = 0.00001

    x0 = torch.tensor([0.5, 0.])
    env.state = x0
    u0 = torch.tensor([0.])
    A,B = env.state_control_transition(x0, u0)
    y0_u = torch.tensor([np.cos(x0[0]), np.sin(x0[0]), x0[1], u0], requires_grad=True, dtype=torch.float32).view(1, 4).to(
        device)
    y0_un = y0_u.detach().cpu().numpy()

    j = env.get_full_jacobian(y0_u)
    jn = j.detach().cpu().numpy()
    print('A', A)
    print('B', B)
    s, u = env.step(u0)
    print('s', s)
    print('u', u)
    print('c, s', (np.sin(s[0]), np.cos(s[0])))
    xs = np.matmul(A.detach().cpu().numpy(), [np.cos(x0[0]), np.sin(x0[0]), x0[1]])
    us = np.matmul(B.detach().cpu().numpy(), u0.detach().cpu().numpy())
    qs = xs + us
    qs = np.matmul(jn, y0_un.T)
    print('qs', qs)
    As = [np.arctan2(qs[1], qs[0]), qs[2]]
    print('As', As)

if __name__ == "__main__":
    args = ObjectView(get_args())

    main(args)