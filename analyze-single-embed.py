# Symplectic ODE-Net | 2019
# Yaofeng Desmond Zhong, Biswadip Dey, Amit Chakraborty

# code structure follows the style of HNN by Sam Greydanus
# https://github.com/greydanus/hamiltonian-nn

# Cells are seperated by the vscode convention '#%%'

# %%
import argparse
import sys
import torch
import matplotlib.pyplot as plt

from environments import learned_params
from symplectic.analysis import simulate_control, simulate_models, get_all_models_and_stats, get_prediction_error
from symplectic.plot_single_embed import plot_control, plot_energy_variation, plot_learned_functions, \
    plot_model_vs_true_ivp, plot_learning
from symplectic.utils import ObjectView

EXPERIMENT_DIR = '/experiment_single_embed'
sys.path.append(EXPERIMENT_DIR)

# %%
DPI = 100
FORMAT = 'pdf'

# TODO(Rebecca): put this with learned_params
# _env_name = 'DDP-Pendulum-v0'
_env_name = 'MyPendulum-v0'

def get_args():
    """ Arguments to fetch a model. Must match existing trained models exactly."""
    return learned_params.get_pendulum_args(EXPERIMENT_DIR, _env_name)


def main(args):
    model_args = ObjectView(get_args())
    device = torch.device('cuda:' + str(model_args.gpu) if torch.cuda.is_available() else 'cpu')
    # %%
    base_ode_model, naive_ode_model, symoden_ode_model, symoden_ode_struct_model, base_ode_stats, naive_ode_stats, symoden_ode_stats, symoden_ode_struct_stats = get_all_models_and_stats(model_args, device)
    get_prediction_error(model_args, base_ode_model, device, naive_ode_model, symoden_ode_model, symoden_ode_struct_model)
    base_ivp, naive_ivp, symoden_ivp, symoden_struct_ivp, t_linspace_model, t_linspace_true, true_ivp_y = simulate_models(
        base_ode_model, naive_ode_model, symoden_ode_model, symoden_ode_struct_model, device, model_args)

    #custom plots
    plot_learning(base_ode_stats, naive_ode_stats, symoden_ode_stats, symoden_ode_struct_stats, DPI=DPI)


    # Plot Models
    #plot_sin_cos_sanity_check(base_ivp, naive_ivp, symoden_ivp, symoden_struct_ivp, t_linspace_model)
    plot_learned_functions(symoden_ode_struct_model, device, DPI=DPI)
    plot_energy_variation(base_ivp, symoden_ivp, symoden_struct_ivp, t_linspace_model, t_linspace_true, true_ivp_y)
    #
    # # Plot control
    t_eval, y_traj = simulate_control(device, symoden_ode_struct_model, model_args, render=False)
    plot_control(t_eval, y_traj, model_args, DPI=DPI, FORMAT=FORMAT)

    # Plot trajectory
    plot_model_vs_true_ivp(base_ivp, naive_ivp, symoden_ivp, symoden_struct_ivp, true_ivp_y, DPI=DPI)

    if not args.test:
        plt.show()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run analysis on single embed test')
    parser.add_argument('--test', action='store_true', help='Run as test')
    args = parser.parse_args()

    main(args)

# %%
