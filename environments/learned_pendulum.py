""" A Pendulum Learned using the SymplectivODENet thing"""

import torch
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from environments import pendulum_params
from environments import pendulum
from symplectic.analysis import get_one_step_prediction
import pdb



class LearnedPendulumEnv(pendulum.PendulumEnv):

    def __init__(self, symp_oden_model, device):
        """ TODO: make compatible with other models"""
        pendulum.PendulumEnv.__init__(self)
        self.model = symp_oden_model
        self.device = device
        # Paramters for kwargs of integrate_model
        self.rtol = 1e-12
        self.method = 'RK45'
        # Let solve_ivp choose the points to evaluate the system at
        self.t_eval = None
        kwargs = {'t_eval': self.t_eval, 'rtol': self.rtol, 'method': self.method}

    def step(self, u):
        """ Do one step of simulation given an input u
            TODO: implement
        """

        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        b = self.b
        dt = self.dt
        I = self.I

        u = u[0]
        acceleration = -b / I * thdot - m * g * l / I * np.sin(th) + u / I
        newth = th + thdot * dt
        newthdot = thdot + acceleration * dt

        self.state = np.array([newth, newthdot])

        reward = self.get_ddp_reward(u)

        return self.state, reward

    def state_control_transition(self, x, u):
        """ takes in state and control trajectories and outputs the Jacobians for the linearized system
        edit function to use with autograd when linearizing the neural network output REBECCA
            TODO:Rebecca
        """
        assert x.shape[0] == 2, "Expected x = [q, qdot], got something else."
        # First, find the model as evaluated at x, u
        y0_u = np.asarray([np.cos(x[0]), np.sin(x[0]), x[1], u])
        kwargs = {'t_eval': self.t_eval, 'rtol': self.rtol, 'method': self.method}

        t_span = [0, self.dt]
        x_hats = get_one_step_prediction(self.model, y0_u, t_span)
        # TODO: rebecca figure out how to get the jacobian


        return A, B
