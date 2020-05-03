import gym
import gym.wrappers
import numpy as np
from scipy.integrate import solve_ivp
import torch
from torchdiffeq import odeint

from experiment_single_embed.data import get_dataset
from symplectic.nn_models import PSD, MLP
from symplectic.symoden import SymODEN_T
from symplectic.utils import from_pickle


def get_model(args, baseline, structure, naive, device):
    """ Gets a model from save. Must be trained first, so these must match the existing model.
        arguements:
        args(object): should match args in analyze-single-embed
        baseline(bool): Whether it is the baseline model or not
        structure(bool): Whether it is the structure model or not
        naive(bool): Whether it is the naive model or not
        device(): cuda device to use
        Return:
            returns the neural network model, as well as the stats of the trained model.
    """

    M_input_dim = 2 * args.num_angle
    M_output_dim = args.num_angle
    M_hidden_dim = 300
    M_net = PSD(M_input_dim, M_hidden_dim, M_output_dim).to(device)

    g_input_dim = 2 * args.num_angle
    g_output_dim = args.num_angle
    g_hidden_dim = 200
    g_net = MLP(g_input_dim, g_hidden_dim, g_output_dim).to(device)
    if not structure:
        if naive and baseline:
            raise RuntimeError('argument *baseline* and *naive* cannot both be true')
        elif naive:
            input_dim = 4 * args.num_angle
            output_dim = 3 * args.num_angle
            hidden_dim = 800
            nn_model = MLP(input_dim, hidden_dim, output_dim, args.nonlinearity).to(device)
            model = SymODEN_T(args.num_angle, H_net=nn_model, device=device, baseline=baseline, naive=naive)
        elif baseline:
            input_dim = 4 * args.num_angle
            output_dim = 2 * args.num_angle
            hidden_dim = 600
            nn_model = MLP(input_dim, hidden_dim, output_dim, args.nonlinearity).to(device)
            model = SymODEN_T(args.num_angle, H_net=nn_model, M_net=M_net, device=device, baseline=baseline,
                              naive=naive)
        else:
            input_dim = 3 * args.num_angle
            output_dim = 1
            hidden_dim = 500
            nn_model = MLP(input_dim, hidden_dim, output_dim, args.nonlinearity).to(device)
            model = SymODEN_T(args.num_angle, H_net=nn_model, M_net=M_net, g_net=g_net, device=device,
                              baseline=baseline, naive=naive)
    elif structure and not baseline and not naive:
        input_dim = 2 * args.num_angle
        output_dim = 1
        hidden_dim = 50
        V_net = MLP(input_dim, hidden_dim, output_dim).to(device)
        model = SymODEN_T(args.num_angle, M_net=M_net, V_net=V_net, g_net=g_net, device=device, baseline=baseline,
                          structure=True).to(device)
    else:
        raise RuntimeError('argument *structure* is set to true, no *baseline* or *naive*!')

    if naive:
        label = '-naive_ode'
    elif baseline:
        label = '-baseline_ode'
    else:
        label = '-hnn_ode'
    struct = '-struct' if structure else ''
    path = '{}/{}{}{}-{}-p{}.tar'.format(args.save_dir, args.name, label, struct, args.solver, args.num_points)
    print(path)
    model.load_state_dict(torch.load(path, map_location=device))
    path = '{}/{}{}{}-{}-p{}-stats.pkl'.format(args.save_dir, args.name, label, struct, args.solver, args.num_points)
    stats = from_pickle(path)
    return model, stats

def np_to_integratable_type_1D(x, device):
    """Converts 1D np array to integratable type for use in this library"""
    shape = x.shape
    assert len(shape) == 1, ("Expected 1D array, got ", x)
    return torch.tensor(x, requires_grad=True, dtype=torch.float32).view(1,shape[0]).to(device)

def get_one_step_prediction(model, x0, dt, device):
    """ Given a model, and an initial condition (1D numpy array), predict for some dt (scalar) in the future.
        returns x_hats
    """
    assert type(x0) == np.ndarray
    x0 = np_to_integratable_type_1D(x0, device)
    ts = torch.tensor([0., dt], requires_grad=True, dtype=torch.float32).to(device)
    x_hats = odeint(model, x0, ts, method='rk4')
    x_hat = x_hats[1]
    assert x_hat.shape == x0.shape
    return x_hat[0].detach().cpu().numpy()


def get_pred_loss(pred_x, pred_t_eval, model, device):
    pred_x = torch.tensor(pred_x, requires_grad=True, dtype=torch.float32).to(device)
    pred_t_eval = torch.tensor(pred_t_eval, requires_grad=True, dtype=torch.float32).to(device)
    # print('pred_x shape', pred_x.shape)
    # pred_x is something like [traj index??, time, ??, ??]
    pred_loss = []
    for i in range(pred_x.shape[0]):
        pred_x0 = pred_x[i, 0, :, :]  # this is the initial condition
        # print('Pred_x0_shape', pred_x0.shape)
        pred_x_hat = odeint(model, pred_x0, pred_t_eval, method='rk4')
        # print('Pred_x_hat_shape', pred_x_hat.shape)
        pred_loss.append((pred_x[i, :, :, :] - pred_x_hat) ** 2)

    pred_loss = torch.cat(pred_loss, dim=1)
    pred_loss_per_traj = torch.sum(pred_loss, dim=(0, 2))

    return pred_loss_per_traj.detach().cpu().numpy()


def integrate_model(model, t_span, y0, device, **kwargs):
    def fun(t, np_x):
        x = torch.tensor(np_x, requires_grad=True, dtype=torch.float32).view(1, 4).to(device)
        dx = model(0, x).detach().cpu().numpy().reshape(-1)
        return dx

    return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)


def get_qp(x):
    q = np.arctan2(-x[:, 1], -x[:, 0])
    p = x[:, 2] / 3
    return np.stack((q, p), axis=1)


def simulate_control(device, symoden_ode_struct_model, args):
    # %% [markdown]
    # ## Energy-based control
    # The following code saves the rendering as a mp4 video and as a GIF at the same time
    # %%
    # time info for simulation
    time_step = 200
    n_eval = 200
    t_span = [0, time_step * 0.05]
    t_eval = torch.linspace(t_span[0], t_span[1], n_eval)
    # angle info for simulation
    init_angle = 3.14
    u0 = 0.0
    env = gym.make(args.env)
    # record video
    env = gym.wrappers.Monitor(env, './videos/' + 'single-embed' + '/',
                               force=True)  # , video_callable=lambda x: True, force=True
    env.reset()
    env.env.state = np.array([init_angle, u0], dtype=np.float32)
    obs = env.env._get_obs()
    y = torch.tensor([obs[0], obs[1], obs[2], u0], requires_grad=True, device=device, dtype=torch.float32).view(1, 4)
    k_d = 3
    y_traj = []
    y_traj.append(y)
    frames = []
    for i in range(len(t_eval) - 1):
        frames.append(env.render(mode='rgb_array'))

        cos_q_sin_q, q_dot, _ = torch.split(y, [2, 1, 1], dim=1)
        cos_q, sin_q = torch.chunk(cos_q_sin_q, 2, dim=1)
        V_q = symoden_ode_struct_model.V_net(cos_q_sin_q)
        g_q = symoden_ode_struct_model.g_net(cos_q_sin_q)
        dV = torch.autograd.grad(V_q, cos_q_sin_q)[0]
        dVdcos_q, dVdsin_q = torch.chunk(dV, 2, dim=1)
        dV_q = - dVdcos_q * sin_q + dVdsin_q * cos_q
        M_inv = symoden_ode_struct_model.M_net(cos_q_sin_q)
        q = torch.atan2(sin_q, cos_q)

        u = 1 / g_q * (2 * dV_q - k_d * q_dot)

        # use openai simulator
        u = u.detach().cpu().numpy()
        obs, _, _, _ = env.step(u)
        y = torch.tensor([obs[0], obs[1], obs[2], u], requires_grad=True, device=device, dtype=torch.float32).view(1, 4)
        # use learnt model
        # y0_u = torch.cat((cos_q_sin_q, q_dot, u), dim = 1)
        # y_step = odeint(symoden_ode_struct_model, y0_u, t_eval[i:i+2], method='rk4')
        # y = y_step[-1,:,:]

        y_traj.append(y)
    env.close()
    # imageio.mimsave('./videos/single-embed/single-embed.gif', frames, duration=0.05)
    y_traj = torch.stack(y_traj).view(-1, 4).detach().cpu().numpy()
    return t_eval, y_traj


def simulate_models(base_ode_model, naive_ode_model, symoden_ode_model, symoden_ode_struct_model, device, args):
    # time info for simualtion
    time_step = 100
    n_eval = 100
    t_span = [0, time_step * 0.05]
    t_linspace_true = np.linspace(t_span[0], time_step, time_step) * 0.05
    t_linspace_model = np.linspace(t_span[0], t_span[1], n_eval)
    # angle info for simuation
    init_angle = 0.5
    y0 = np.asarray([init_angle, 0])
    u0 = 0.0
    y0_u = np.asarray([np.cos(init_angle), np.sin(init_angle), 0, u0])
    # simulate
    kwargs = {'t_eval': t_linspace_model, 'rtol': 1e-12, 'method': 'RK45'}
    naive_ivp = integrate_model(naive_ode_model, t_span, y0_u, device=device, **kwargs)
    base_ivp = integrate_model(base_ode_model, t_span, y0_u, device=device, **kwargs)
    symoden_ivp = integrate_model(symoden_ode_model, t_span, y0_u, device=device, **kwargs)
    symoden_struct_ivp = integrate_model(symoden_ode_struct_model, t_span, y0_u, device=device, **kwargs)
    env = gym.make(args.env)
    env.reset()
    env.state = np.array([init_angle, 0.0], dtype=np.float32)
    obs = env._get_obs()
    obs_list = []
    for _ in range(time_step):
        obs_list.append(obs)
        obs, _, _, _ = env.step([u0])
    true_ivp_y = np.stack(obs_list, 1)
    true_ivp_y = np.concatenate((true_ivp_y, np.zeros((1, time_step))), axis=0)
    return base_ivp, naive_ivp, symoden_ivp, symoden_struct_ivp, t_linspace_model, t_linspace_true, true_ivp_y


def get_all_models(args, device, verbose=True):
    naive_ode_model, naive_ode_stats = get_model(args, baseline=False, structure=False, naive=True, device=device)
    base_ode_model, base_ode_stats = get_model(args, baseline=True, structure=False, naive=False, device=device)
    symoden_ode_model, symoden_ode_stats = get_model(args, baseline=False, structure=False, naive=False, device=device)
    symoden_ode_struct_model, symoden_ode_struct_stats = get_model(args, baseline=False, structure=True, naive=False,
                                                                device=device)
    if verbose:
        print('Naive Baseline contains {} parameters'.format(get_model_parm_nums(naive_ode_model)))
        print('Final trajectory train loss {:.4e} +/- {:.4e}\nFinal trajectory test loss {:.4e} +/- {:.4e}'
              .format(np.mean(naive_ode_stats['traj_train_loss']), np.std(naive_ode_stats['traj_train_loss']),
                      np.mean(naive_ode_stats['traj_test_loss']), np.std(naive_ode_stats['traj_test_loss'])))
        print('')
        print('Geometric Baseline contains {} parameters'.format(get_model_parm_nums(base_ode_model)))
        print('Final trajectory train loss {:.4e} +/- {:.4e}\nFinal trajectory test loss {:.4e} +/- {:.4e}'
              .format(np.mean(base_ode_stats['traj_train_loss']), np.std(base_ode_stats['traj_train_loss']),
                      np.mean(base_ode_stats['traj_test_loss']), np.std(base_ode_stats['traj_test_loss'])))
        print('')
        print('Unstructured SymODEN contains {} parameters'.format(get_model_parm_nums(symoden_ode_model)))
        print('Final trajectory train loss {:.4e} +/- {:.4e}\nFinal trajectory test loss {:.4e} +/- {:.4e}'
              .format(np.mean(symoden_ode_stats['traj_train_loss']), np.std(symoden_ode_stats['traj_train_loss']),
                      np.mean(symoden_ode_stats['traj_test_loss']), np.std(symoden_ode_stats['traj_test_loss'])))
        print('')
        print('SymODEN contains {} parameters'.format(get_model_parm_nums(symoden_ode_struct_model)))
        print('Final trajectory train loss {:.4e} +/- {:.4e}\nFinal trajectory test loss {:.4e} +/- {:.4e}'
              .format(np.mean(symoden_ode_struct_stats['traj_train_loss']),
                      np.std(symoden_ode_struct_stats['traj_train_loss']),
                      np.mean(symoden_ode_struct_stats['traj_test_loss']),
                      np.std(symoden_ode_struct_stats['traj_test_loss'])))
    return base_ode_model, naive_ode_model, symoden_ode_model, symoden_ode_struct_model

def get_all_models_and_stats(args, device):
    naive_ode_model, naive_ode_stats = get_model(args, baseline=False, structure=False, naive=True, device=device)
    base_ode_model, base_ode_stats = get_model(args, baseline=True, structure=False, naive=False, device=device)
    symoden_ode_model, symoden_ode_stats = get_model(args, baseline=False, structure=False, naive=False, device=device)
    symoden_ode_struct_model, symoden_ode_struct_stats = get_model(args, baseline=False, structure=True, naive=False,
                                                                   device=device)
    print('Naive Baseline contains {} parameters'.format(get_model_parm_nums(naive_ode_model)))
    print('Final trajectory train loss {:.4e} +/- {:.4e}\nFinal trajectory test loss {:.4e} +/- {:.4e}'
          .format(np.mean(naive_ode_stats['traj_train_loss']), np.std(naive_ode_stats['traj_train_loss']),
                  np.mean(naive_ode_stats['traj_test_loss']), np.std(naive_ode_stats['traj_test_loss'])))
    print('')
    print('Geometric Baseline contains {} parameters'.format(get_model_parm_nums(base_ode_model)))
    print('Final trajectory train loss {:.4e} +/- {:.4e}\nFinal trajectory test loss {:.4e} +/- {:.4e}'
          .format(np.mean(base_ode_stats['traj_train_loss']), np.std(base_ode_stats['traj_train_loss']),
                  np.mean(base_ode_stats['traj_test_loss']), np.std(base_ode_stats['traj_test_loss'])))
    print('')
    print('Unstructured SymODEN contains {} parameters'.format(get_model_parm_nums(symoden_ode_model)))
    print('Final trajectory train loss {:.4e} +/- {:.4e}\nFinal trajectory test loss {:.4e} +/- {:.4e}'
          .format(np.mean(symoden_ode_stats['traj_train_loss']), np.std(symoden_ode_stats['traj_train_loss']),
                  np.mean(symoden_ode_stats['traj_test_loss']), np.std(symoden_ode_stats['traj_test_loss'])))
    print('')
    print('SymODEN contains {} parameters'.format(get_model_parm_nums(symoden_ode_struct_model)))
    print('Final trajectory train loss {:.4e} +/- {:.4e}\nFinal trajectory test loss {:.4e} +/- {:.4e}'
          .format(np.mean(symoden_ode_struct_stats['traj_train_loss']),
                  np.std(symoden_ode_struct_stats['traj_train_loss']),
                  np.mean(symoden_ode_struct_stats['traj_test_loss']),
                  np.std(symoden_ode_struct_stats['traj_test_loss'])))
    return base_ode_model, naive_ode_model, symoden_ode_model, symoden_ode_struct_model, base_ode_stats, naive_ode_stats, symoden_ode_stats, symoden_ode_struct_stats

def get_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    return total


def get_prediction_error(args, base_ode_model, device, naive_ode_model, symoden_ode_model, symoden_ode_struct_model):
    # %% [markdown]
    # ## Dataset to get prediction error
    # %%
    us = [0.0]
    data = get_dataset(seed=args.seed, timesteps=40,
                       save_dir=args.save_dir, us=us, samples=128)
    pred_x, pred_t_eval = data['x'], data['t']
    naive_pred_loss = get_pred_loss(pred_x, pred_t_eval, naive_ode_model, device=device)
    base_pred_loss = get_pred_loss(pred_x, pred_t_eval, base_ode_model, device=device)
    symoden_pred_loss = get_pred_loss(pred_x, pred_t_eval, symoden_ode_model, device=device)
    symoden_struct_pred_loss = get_pred_loss(pred_x, pred_t_eval, symoden_ode_struct_model, device=device)
    # %%
    print('Naive Baseline')
    print('Prediction error {:.4e} +/- {:.4e}'
          .format(np.mean(naive_pred_loss), np.std(naive_pred_loss)))
    print('')
    print('Geometric Baseline')
    print('Prediction error {:.4e} +/- {:.4e}'
          .format(np.mean(base_pred_loss), np.std(base_pred_loss)))
    print('')
    print('Unstructured SymODEN')
    print('Prediction error {:.4e} +/- {:.4e}'
          .format(np.mean(symoden_pred_loss), np.std(symoden_pred_loss)))
    print('')
    print('SymODEN')
    print('Prediction error {:.4e} +/- {:.4e}'
          .format(np.mean(symoden_struct_pred_loss), np.std(symoden_struct_pred_loss)))
