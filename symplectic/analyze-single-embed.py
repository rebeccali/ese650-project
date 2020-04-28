# Symplectic ODE-Net | 2019
# Yaofeng Desmond Zhong, Biswadip Dey, Amit Chakraborty

# code structure follows the style of HNN by Sam Greydanus
# https://github.com/greydanus/hamiltonian-nn

# Cells are seperated by the vscode convention '#%%'

# %%
import sys
import torch
import matplotlib.pyplot as plt


from symplectic.analysis import simulate_control, simulate_models, get_all_models, get_prediction_error
from symplectic.plot_single_embed import plot_control, plot_energy_variation, plot_learned_functions, \
    plot_sin_cos_sanity_check, plot_model_vs_true_ivp

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
    # %%
    base_ode_model, naive_ode_model, symoden_ode_model, symoden_ode_struct_model = get_all_models(args, device)
    get_prediction_error(args, base_ode_model, device, naive_ode_model, symoden_ode_model, symoden_ode_struct_model)
    base_ivp, naive_ivp, symoden_ivp, symoden_struct_ivp, t_linspace_model, t_linspace_true, true_ivp_y = simulate_models(
        base_ode_model, naive_ode_model, symoden_ode_model, symoden_ode_struct_model, device, args)

    # Plot Models
    plot_sin_cos_sanity_check(base_ivp, naive_ivp, symoden_ivp, symoden_struct_ivp, t_linspace_model)
    plot_learned_functions(symoden_ode_struct_model, device, DPI=DPI)
    plot_energy_variation(base_ivp, symoden_ivp, symoden_struct_ivp, t_linspace_model, t_linspace_true, true_ivp_y)

    # Plot control
    t_eval, y_traj = simulate_control(device, symoden_ode_struct_model, args)
    plot_control(t_eval, y_traj, args, DPI=DPI, FORMAT=FORMAT)

    # Plot trajectory
    plot_model_vs_true_ivp(base_ivp, naive_ivp, symoden_ivp, symoden_struct_ivp, true_ivp_y, DPI=DPI)
    plt.show()


if __name__ == "__main__":
    args = ObjectView(get_args())

    main(args)

# %%
