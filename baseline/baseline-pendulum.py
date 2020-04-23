import torch
import autograd
import autograd.numpy as np

import scipy.integrate

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
    theta = x0[0]
    thetadot = x0[1]

    dxdt = np.zeros([states, 1])
    dxdt[0] = thetadot
    dxdt[1] = -g / L * np.sin(theta)

    return dxdt


def get_trajectory(radius=None, x0=None, **kwargs):
    """ takes in the batch size and outputs the state trajectory from a randomly sampled intitial condition """
    t_span = [0, 3]
    timescale = 15  # size of trajectory ~~ is this same as batch size???
    noise_std = 1

    t_eval = np.linspace(t_span[0], t_span[1], int(timescale * (t_span[1] - t_span[0])))

    # get initial state randomly from the distribution
    if x0 is None:
        x0 = np.random.rand(states) * 2. - 1

    if radius is None:
        radius = np.random.rand() + 1.3

    # radius is scaled by the norm of the state vector
    x0 = x0 / (np.sqrt(x0 ** 2).sum()) * radius

    # solve ivp with scipy ivp solver

    x = solve_ivp(fun=pend_dynamics, t_span=t_span, y0=x0, t_eval=t_eval, rtol=1e-10)

    # add noise to states
    x += np.random.randn(states) * noise_std

    return x, x0


def get_dataset():
    """ gets the dataset """

    samples = 50  # number of batches
    split = 0.5
    seed = 0

    # randomly sample inputs
    np.random.seed(seed)

    x = []
    dx = []

    for s in range(samples):
        # get batch of samples
        x, x0 = get_trajectory()

    return data


if __name__ == "__main__":
    data = get_dataset()
