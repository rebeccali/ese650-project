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
from symplectic.au_functional import jacobian
import pdb


class LearnedPendulumEnv(pendulum.PendulumEnv):

    def __init__(self, model, device):
        """ TODO: make compatible with other models"""
        pendulum.PendulumEnv.__init__(self)
        self.model = model
        self.device = device
        # Parameters for kwargs of integrate_model
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
        cos_q = np.cos(th)
        sin_q = np.sin(th)

        y0_u = torch.tensor([cos_q, sin_q, thdot, u[0]], requires_grad=True, dtype=torch.float32).view(1,4).to(self.device)

        y1_u = get_one_step_prediction(self.model, y0_u, self.dt, device=self.device)
        y1_u = y1_u[0].detach().cpu().numpy()
        new_cos_q = y1_u[0]
        new_sin_q = y1_u[1]
        new_thdot = y1_u[2]
        new_th = np.arctan2(new_sin_q, new_cos_q)

        self.state = np.array([new_th, new_thdot])

        reward = self.get_ddp_reward(u[0])

        return self.state, reward

    def state_control_transition(self, x, u):
        """ takes in state and control trajectories and outputs the Jacobians for the linearized system
        """
        assert x.shape[0] == self.states, "Expected x = [q, qdot], got something else."
        assert u.shape[0] == self.num_controllers, "Expected u = [torque], got something else."
        # First, find the model as evaluated at x, u
        # y0_u should be (n, 4) shape
        y0_u = torch.tensor([np.cos(x[0]), np.sin(x[0]), x[1], u[0]], requires_grad=True, dtype=torch.float32).view(1, 4).to(self.device)

        dfdy0 = self.get_full_jacobian(y0_u)
        # Note the jacobian is 4x4 instead of 4x3:
        # this is because the network returns [dcos(q), dsin(q), ddq, 0]
        # so for n controls, the n final rows should be ignored. They should
        # also be zero.
        dfdy0 = dfdy0[:-self.num_controllers, :]

        # first columns are A
        A = dfdy0[:, :-self.num_controllers]
        # last columns are B
        B = dfdy0[:, -self.num_controllers:]

        return A, B

    def get_full_jacobian(self, y0_u):
        # Jacobian evaluated at y0
        t0 = 0.
        # Dynamics function to take jacobian of
        f = lambda y: self.model(t0, y)
        dfdy0 = jacobian(f, y0_u)  # Jacobian evaluated at y0_u

        dfdy0 = dfdy0[0, :, 0, :]  # Reshape to get rid of extra dimensions
        return dfdy0

    def render(self, mode='human'):
        print('UNIMPLEMENTED RENDER')
        pass
