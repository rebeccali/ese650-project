import numpy as np
from matplotlib import pyplot as plt


def plot_trajectories(ts, traj_true, traj_learned, traj_naive, env_name):
    """ Plots a comparison of each of the states of the learned and ground truth trajectories
    """
    num_states = traj_true.shape[1]

    f, ax = plt.subplots(num_states)
    for i in range(num_states):
        ax[i].plot(ts, traj_true[:, i], 'r--', label='Ground Truth')
        ax[i].plot(ts, traj_learned[:, i], 'g-', label='Structured Learned')
        ax[i].plot(ts, traj_naive[:, i], 'b-', label='Naive Baseline Learned')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel('x_%d' % i)
        ax[i].legend()
    ax[0].set_title('Learned Dynamics for %s system' % env_name)


def rollout_models(ground_truth_env, learned_env, naive_env, horizon):
    """ Gets a trajectory for a horizon number of seconds of the ground truth and learned models
    """
    assert learned_env.dt == ground_truth_env.dt
    assert learned_env.dt == naive_env.dt
    dt = learned_env.dt
    ts = np.arange(0, horizon, dt)
    N = len(ts)
    us = np.zeros((N, learned_env.num_controllers))
    # Set initial state
    x0 = np.ones(ground_truth_env.states)
    ground_truth_env.state = x0
    x0_embed = ground_truth_env._get_obs()
    learned_env.state = x0_embed
    naive_env.state = x0_embed
    ground_truth_env.set_training_mode()
    traj_true = [x0_embed]
    traj_learned = [x0_embed]
    traj_naive = [x0_embed]
    print('Comparing systems with dt=%2.3f for %2.1f seconds' % (dt, horizon))
    for i in range(N-1):
        u = us[i]
        x_true, _, _, _ = ground_truth_env.step(u)
        x_learned, _ = learned_env.step(u)
        x_naive, _ = naive_env.step(u)
        traj_true.append(x_true)
        traj_learned.append(x_learned)
        traj_naive.append(x_naive)
    return np.array(traj_true), np.array(traj_learned), np.array(traj_naive), ts
