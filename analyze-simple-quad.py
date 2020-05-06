# Analyze learned simple quad

# %%
import argparse
import sys
import torch
import matplotlib.pyplot as plt

from environments import learned_params
from symplectic.analysis import simulate_control, simulate_models, get_all_models_and_stats, get_prediction_error
from symplectic.plot_single_embed import plot_control, plot_energy_variation, plot_learned_functions, \
    plot_sin_cos_sanity_check, plot_model_vs_true_ivp
from symplectic.utils import ObjectView

EXPERIMENT_DIR = '/experiment_simple_quad'
sys.path.append(EXPERIMENT_DIR)

# %%
DPI = 100
FORMAT = 'pdf'


def get_args():
    """ Arguments to fetch a model. Must match existing trained models exactly."""
    return learned_params.get_quad_args(EXPERIMENT_DIR, 'Simple-Quad-v0')

def plot_learning(base_ode_stats, naive_ode_stats, symoden_ode_stats, symoden_ode_struct_stats, DPI):
    base_test_loss = base_ode_stats['test_loss']
    naive_test_loss = naive_ode_stats['test_loss']
    unstruct_test_loss = symoden_ode_stats['test_loss']
    struct_test_loss = symoden_ode_struct_stats['test_loss']
    print(base_test_loss)
    print(len(base_test_loss), len(naive_test_loss), len(unstruct_test_loss), len(struct_test_loss))

    print(len(struct_test_loss),len(symoden_ode_struct_stats['forward_time']))

    fig = plt.figure(figsize=[5, 5], dpi=DPI)
    plt.plot(range(len(base_test_loss)),base_test_loss,label='Baseline')
    plt.plot(range(len(naive_test_loss)),naive_test_loss,label='Naive')
    plt.plot(range(len(unstruct_test_loss)),unstruct_test_loss,label='Untructured sympODEN')
    plt.plot(range(len(struct_test_loss)),struct_test_loss,label='Structured sympODEN')
    #plt.plot(range(len(symoden_ode_struct_stats['forward_time'])),symoden_ode_struct_stats['train_loss'])
    plt.legend(fontsize=10)
    plt.title('Loss vs training step')



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

    """
    # Plot Models
    #plot_sin_cos_sanity_check(base_ivp, naive_ivp, symoden_ivp, symoden_struct_ivp, t_linspace_model)
    plot_learned_functions(symoden_ode_struct_model, device, DPI=DPI)
    plot_energy_variation(base_ivp, symoden_ivp, symoden_struct_ivp, t_linspace_model, t_linspace_true, true_ivp_y)

    # Plot control
    t_eval, y_traj = simulate_control(device, symoden_ode_struct_model, args)
    plot_control(t_eval, y_traj, args, DPI=DPI, FORMAT=FORMAT)

    # Plot trajectory
    plot_model_vs_true_ivp(base_ivp, naive_ivp, symoden_ivp, symoden_struct_ivp, true_ivp_y, DPI=DPI)
    """
    if not args.test:
        plt.show()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run analysis on simple quad test')
    parser.add_argument('--test', action='store_true', help='Run as test')
    args = parser.parse_args()

    main(args)

# %%
