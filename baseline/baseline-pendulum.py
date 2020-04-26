import torch
import autograd
import autograd.numpy as np

import scipy.integrate
import pdb

solve_ivp = scipy.integrate.solve_ivp

# pendulum parameters
states = 2  # number of states of system (theta, thetadot)
m = 0.5  # mass
g = 9.81  # gravity
L = 1  # length of the pendulum

# network params
input_dim = states
hidden_dim = 200
learning_rate = 1e-3
total_steps = 2000
noise = 0.5


def pend_dynamics(t, x0):
    theta, thetadot = np.split(x0, 2)

    dtheta = thetadot
    dthetadot = -g / L * np.sin(theta)

    dxdt = np.concatenate([dtheta, dthetadot], axis=-1)

    return dxdt


def get_trajectory(t_span, timescale, noise_std, radius=None, x0=None, **kwargs):
    """ takes in the batch size and outputs the state trajectory from a randomly sampled intitial condition """
    # t_span = [0, 3]
    # timescale = 15  # this is the discretization per second, size of the trajectory is 45 in this case
    # noise_std = 1

    t_eval = np.linspace(t_span[0], t_span[1], int(timescale * (t_span[1] - t_span[0])))

    # get initial state randomly from the distribution
    if x0 is None:
        x0 = np.random.rand(states) * 2. - 1

    if radius is None:
        radius = np.random.rand() + 1.3

    # radius is scaled by the norm of the state vector
    x0 = x0 / np.sqrt((x0 ** 2).sum()) * radius

    # solve ivp with scipy ivp solver
    y = solve_ivp(fun=pend_dynamics, t_span=t_span, y0=x0, t_eval=t_eval, rtol=1e-10)

    x = y['y']
    x += np.random.randn(*x.shape) * noise_std

    return x


def get_dataset():
    """ gets the dataset """

    num_samples = 50  # number of batches
    split = 0.5
    seed = 0

    # params for each specific trajectory
    t_span = [0, 3]
    timescale = 15  # this is the discretization per second, size of the trajectory is 45 in this case
    noise_std = 1

    # randomly sample inputs
    np.random.seed(seed)

    x = np.zeros([num_samples, states, timescale * (t_span[1] - t_span[0])])

    for s in range(num_samples):
        # get batch of samples
        x_sample = get_trajectory(t_span, timescale, noise_std)
        x[s, :, :] = x_sample

    test_size = int(np.ceil(num_samples / 2))

    test_data = x[:test_size, :, :]
    train_data = x[test_size:, :, :]

    return test_data, train_data


if __name__ == "__main__":
    data = get_dataset()
    #
