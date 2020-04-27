# Symplectic ODE-Net | 2019
# Yaofeng Desmond Zhong, Biswadip Dey, Amit Chakraborty

# code structure follows the style of HNN by Sam Greydanus
# https://github.com/greydanus/hamiltonian-nn

# This file is a script version of 'analyze-single-embed.ipynb'
# Cells are seperated by the vscode convention '#%%'

# %%
import torch, sys
import matplotlib.pyplot as plt
import scipy.integrate

from symplectic.analysis import simulate_control, simulate_models, get_all_models, get_prediction_error
from symplectic.plot_single_embed import plot_control, plot_energy_variation, plot_learned_functions, \
    plot_sin_cos_sanity_check

solve_ivp = scipy.integrate.solve_ivp

EXPERIMENT_DIR = '../experiment_single_embed/'
sys.path.append(EXPERIMENT_DIR)

# %%
DPI = 100
FORMAT = 'pdf'


def get_args():
    return {'num_angle': 1,
            'nonlinearity': 'tanh',
            'name': 'pend',
            'seed': 0,
            'save_dir': './{}'.format(EXPERIMENT_DIR),
            'fig_dir': './figures',
            'num_points': 2,
            'gpu': 0,
            'solver': 'rk4'}


class ObjectView(object):
    def __init__(self, d): self.__dict__ = d


def main():
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    # %%
    base_ode_model, naive_ode_model, symoden_ode_model, symoden_ode_struct_model = get_all_models(device)
    get_prediction_error(base_ode_model, device, naive_ode_model, symoden_ode_model, symoden_ode_struct_model)
    base_ivp, naive_ivp, symoden_ivp, symoden_struct_ivp, t_linspace_model, t_linspace_true, true_ivp = simulate_models(
        base_ode_model, naive_ode_model, symoden_ode_model, symoden_ode_struct_model, device)
    # true_qp = get_qp(true_ivp.T)
    # naive_qp = get_qp(naive_ivp.y.T)
    # base_qp = get_qp(base_ivp.y.T)
    # symoden_qp = get_qp(symoden_ivp.y.T)
    # symoden_struct_qp = get_qp(symoden_struct_ivp.y.T)
    # %%
    # comparing true trajectory and the estimated trajectory
    # plt.plot(t_linspace_model, symoden_struct_ivp.y[1,:], 'r')
    # plt.plot(t_linspace_true, true_ivp[1,:], 'g')
    plot_sin_cos_sanity_check(base_ivp, naive_ivp, symoden_ivp, symoden_struct_ivp, t_linspace_model)
    plot_learned_functions(symoden_ode_struct_model, device)
    plot_energy_variation(base_ivp, symoden_ivp, symoden_struct_ivp, t_linspace_model, t_linspace_true, true_ivp)
    t_eval, y_traj = simulate_control(device, symoden_ode_struct_model)
    plot_control(t_eval, y_traj)
    plt.show()


if __name__ == "__main__":
    args = ObjectView(get_args())

    main()

# %%
