""" Learned params idk why"""
import pathlib

proj_dir = pathlib.Path(__file__).parents[1].absolute()


def get_device(gpu):
    import torch
    torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')


def get_quad_args(EXPERIMENT_DIR, env):
    return {'num_angle': 1,  # Number of generalized coordinates # TODO
            'nonlinearity': 'tanh',  # NN nonlinearity
            'name': 'quad',  # name of environment # TODO
            'seed': 0,
            'save_dir': '{}'.format(str(proj_dir) + EXPERIMENT_DIR),
            'fig_dir': '{}/figures'.format(proj_dir),
            'num_points': 2,  # number of evaluation points by ode solver, including initial point
            'gpu': 0,
            'solver': 'rk4',
            'env': env,  # Name of the gym environment
            }


def get_pendulum_args(EXPERIMENT_DIR, env):
    return {'num_angle': 1,  # Number of generalized coordinates
            'nonlinearity': 'tanh',  # NN nonlinearity
            'name': 'pend',  # name of environment
            'seed': 0,
            'save_dir': '{}'.format(str(proj_dir) + EXPERIMENT_DIR),
            'fig_dir': '{}/figures'.format(proj_dir),
            'num_points': 2,  # number of evaluation points by ode solver, including initial point
            'gpu': 0,
            'solver': 'rk4',
            'env': env,  # Name of the gym environment
            }
