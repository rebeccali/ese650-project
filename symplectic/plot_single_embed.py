import numpy as np
import torch
from matplotlib import pyplot as plt


def plot_control(t_eval, y_traj, args, DPI=100, FORMAT='pdf'):
    # %% [markdown]
    # ## Plot control results
    # %%
    fig = plt.figure(figsize=[10, 2.2], dpi=DPI)
    plt.subplot(1, 3, 1)
    plt.plot(t_eval.numpy(), 1 * np.ones_like(t_eval.numpy()), 'k--', linewidth=0.5)
    plt.plot(t_eval.numpy(), 0 * np.ones_like(t_eval.numpy()), 'k-', linewidth=0.5)
    plt.plot(t_eval.numpy(), y_traj[:, 1], 'b', label=r'$\sin(q)$', linewidth=2)
    plt.plot(t_eval.numpy(), y_traj[:, 0], 'b--', label=r'$\cos(q)$', linewidth=2)
    plt.title('$q$', fontsize=14)
    plt.xlabel('$t$', fontsize=14)
    plt.ylim([-1.1, 1.1])
    plt.legend(fontsize=10)
    plt.subplot(1, 3, 2)
    plt.plot(t_eval.numpy(), 0 * np.ones_like(t_eval.numpy()), 'k-', linewidth=0.5)
    plt.plot(t_eval.numpy(), y_traj[:, 2], 'b', linewidth=2)
    plt.title('$\dot{q}$', fontsize=14)
    plt.xlabel('$t$', fontsize=14)
    plt.ylim([-4.1, 4.1])
    plt.subplot(1, 3, 3)
    plt.plot(t_eval.numpy(), 0 * np.ones_like(t_eval.numpy()), 'k-', linewidth=0.5)
    plt.plot(t_eval.numpy(), y_traj[:, 3], 'b', linewidth=2)
    plt.title('$u$', fontsize=14)
    plt.xlabel('$t$', fontsize=14)
    plt.ylim([-10.1, 10.1])
    plt.tight_layout()
    fig.savefig('{}/fig-embed-ctrl.{}'.format(args.fig_dir, FORMAT))


def plot_energy_variation(base_ivp, symoden_ivp, symoden_struct_ivp, t_linspace_model, t_linspace_true, true_ivp, DPI=100):
    # %% [markdown]
    # ## Energy variation along each trajectory
    # %%
    fig = plt.figure(figsize=[5, 5], dpi=DPI)
    E_true = true_ivp.T[:, 2] ** 2 / 6 + 5 * (1 + true_ivp.T[:, 0])
    E_base = base_ivp.y.T[:, 2] ** 2 / 6 + 5 * (1 + base_ivp.y.T[:, 0])
    E_symoden = symoden_ivp.y.T[:, 2] ** 2 / 6 + 5 * (1 + symoden_ivp.y.T[:, 0])
    E_symoden_struct = symoden_struct_ivp.y.T[:, 2] ** 2 / 6 + 5 * (1 + symoden_struct_ivp.y.T[:, 0])
    plt.plot(t_linspace_true, E_true, 'k', label='Ground Truth')
    plt.plot(t_linspace_model, E_base, 'r', label='Baseline')
    plt.plot(t_linspace_model, E_symoden, 'g', label='SympODEN')
    plt.plot(t_linspace_model, E_symoden_struct, 'b', label='Structured SympODEN')
    plt.legend(fontsize=10)
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Energy Variation along each trajectory')


def plot_learned_functions(symoden_ode_struct_model, device, DPI=100):
    # ## Plot learnt functions

    q = np.linspace(-5.0, 5.0, 40)
    q_tensor = torch.tensor(q, dtype=torch.float32).view(40, 1).to(device)
    cos_q_sin_q = torch.cat((-torch.cos(q_tensor), -torch.sin(q_tensor)), dim=1)
    beta = 0.357

    fig = plt.figure(figsize=(9.6, 2.5), dpi=DPI)

    plt.subplot(1, 3, 1)
    g_q = symoden_ode_struct_model.g_net(cos_q_sin_q)
    plt.plot(q, np.ones_like(q), label='Ground Truth', color='k', linewidth=2)
    plt.plot(q, beta * g_q.detach().cpu().numpy(), 'b--', linewidth=3, label=r'SymODEN $\beta g_{\theta_3}(q)$')
    plt.xlabel("$q$", fontsize=14)
    # plt.ylabel("$g(q)$", rotation=0, fontsize=14)
    plt.title("$g(q)$", pad=10, fontsize=14)
    plt.xlim(-5, 5)
    plt.ylim(0, 4)
    plt.legend(fontsize=10)
    M_q_inv = symoden_ode_struct_model.M_net(cos_q_sin_q)

    plt.subplot(1, 3, 2)
    plt.plot(q, 3 * np.ones_like(q), label='Ground Truth', color='k', linewidth=2)
    plt.plot(q, M_q_inv.detach().cpu().numpy() / beta, 'b--', linewidth=3,
             label=r'SymODEN $M^{-1}_{\theta_1}(q)/\beta$')
    plt.xlabel("$q$", fontsize=14)
    # plt.ylabel("$M^{-1}(q)$", rotation=0, fontsize=14)
    plt.title("$M^{-1}(q)$", pad=10, fontsize=14)
    plt.xlim(-5, 5)
    plt.ylim(0, 4)
    plt.legend(fontsize=10)
    V_q = symoden_ode_struct_model.V_net(cos_q_sin_q)

    plt.subplot(1, 3, 3)
    plt.plot(q, 5. - 5. * np.cos(q), label='Ground Truth', color='k', linewidth=2)
    plt.plot(q, beta * V_q.detach().cpu().numpy(), 'b--', label=r'SymODEN $\beta V_{\theta_2}(q)$', linewidth=3)
    plt.xlabel("$q$", fontsize=14)
    # plt.ylabel("$V(q)$", rotation=0, fontsize=14)
    plt.title("$V(q)$", pad=10, fontsize=14)
    plt.xlim(-5, 5)
    plt.ylim(-7, 22)
    plt.legend(fontsize=10)
    plt.tight_layout()
    # fig.savefig('{}/fig-single-embed.{}'.format(args.fig_dir, FORMAT))


def plot_sin_cos_sanity_check(base_ivp, naive_ivp, symoden_ivp, symoden_struct_ivp, t_linspace_model):
    # ## Sanity Check of the trajectories. The first two state variables $\cos q$ and $\sin q$ should lie on $\mathbb{S}^1$.
    naive_1 = naive_ivp.y[0, :] ** 2 + naive_ivp.y[1, :] ** 2
    base_1 = base_ivp.y[0, :] ** 2 + base_ivp.y[1, :] ** 2
    symoden_1 = symoden_ivp.y[0, :] ** 2 + symoden_ivp.y[1, :] ** 2
    symoden_struct_1 = symoden_struct_ivp.y[0, :] ** 2 + symoden_struct_ivp.y[1, :] ** 2
    plt.plot(t_linspace_model, naive_1, 'y', label='Naive Baseline')
    plt.plot(t_linspace_model, base_1, 'r', label='Geometric Baseline')
    plt.plot(t_linspace_model, symoden_1, 'g', label='unstructured SymODEN')
    plt.plot(t_linspace_model, symoden_struct_1, 'b', label='SymODEN')
    plt.title(r'Sanity check of $\sin^2 q + \cos^2 q$')
    plt.legend(fontsize=10)
    # %%
    plt.plot(t_linspace_model, base_1, 'r', label='Geometric Baseline')
    plt.plot(t_linspace_model, symoden_1, 'g', label='unstructured SymODEN')
    plt.plot(t_linspace_model, symoden_struct_1, 'b', label='SymODEN')
    plt.title(r'Sanity check of $\sin^2 q + \cos^2 q$')
    plt.legend(fontsize=10)