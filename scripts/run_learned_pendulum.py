import gym
import numpy as np
import sys
import torch
import os

sys.path.append('./')
from environments.learned_pendulum import LearnedPendulumEnv
from symplectic.analysis import simulate_control, simulate_models, get_all_models, get_prediction_error
#
# EXPERIMENT_DIR = 'experiment_single_embed/'
# sys.path.append(EXPERIMENT_DIR)

# %%
DPI = 100
FORMAT = 'pdf'

#
# def get_args():
#     """ Arguments to fetch a model. Must match existing trained models exactly."""
#     return {'num_angle': 1,  # Number of generalized coordinates
#             'nonlinearity': 'tanh',  # NN nonlinearity
#             'name': 'pend',  # name of environment
#             'seed': 0,
#             'save_dir': './{}'.format(EXPERIMENT_DIR),
#             'fig_dir': './figures',
#             'num_points': 2,  # number of evaluation points by ode solver, including initial point
#             'gpu': 0,
#             'solver': 'rk4',
#             'env': 'MyPendulum-v0'  # Name of the gym environment
#             }
#
#
# class ObjectView(object):
#     def __init__(self, d): self.__dict__ = d


def main():
    #
    #
    # device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    # base_ode_model, naive_ode_model, symoden_ode_model, symoden_ode_struct_model = get_all_models(args, device,
    #                                                                                               verbose=False)
    env = LearnedPendulumEnv(model_type='structure')
    dt = 0.05
    env.dt = dt

    x0 = np.array([0.5, 0.5, 0.])
    env.state = x0
    u0 = np.array([0.])
    A, B = env.state_control_transition(x0, u0)
    y0_un = np.hstack([x0, u0])
    jn = env.get_full_jacobian(y0_un)
    print('A', A)
    print('B', B)
    new_s, _ = env.step(u0)

    csx0 = x0
    new_ds = np.array(new_s) - x0

    print('new_ds', new_ds)
    # check the jacobian and A B match

    xs = np.matmul(A, csx0)
    us = np.matmul(B, u0)
    qs_AB = (xs + us)
    print('qs_AB', qs_AB)
    qs_jac = y0_un.T + np.matmul(jn, y0_un.T) * dt
    print('qs_jac', qs_jac)
    # check the system matches s, u
    ABs = [np.arctan2(qs_AB[1], qs_AB[0]), qs_AB[2]]
    print('ABs', ABs)
    jacs = [np.arctan2(qs_jac[1], qs_jac[0]), qs_jac[2]]
    print('jacs', jacs)


if __name__ == "__main__":
    # args = get_args

    main()
