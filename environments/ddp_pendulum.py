import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
import matplotlib.pyplot as plt
from environments import pendulum_params as params
import pdb


# from open ai gym source:
class PendulumEnv(gym.Env):

    def __init__(self):
        self.dt = params.dt
        self.g = params.gr
        self.m = params.m
        self.l = params.L
        self.b = params.b
        self.I = params.I

        self.Q_r_ddp = params.Q_f_ddp
        self.Q_f_ddp = params.Q_f_ddp
        self.R_ddp = params.R_ddp
        self.states = params.states
        self.num_controllers = params.num_controllers

        # set initial and final states
        self.state = np.zeros((params.states,))
        self.goal = params.xf

        self.max_speed = params.max_speed
        self.max_torque = params.max_torque

        high = np.array([1., 1., self.max_speed])
        self.min_state = -high
        self.max_state = high
        self.min_action = [-self.max_torque]
        self.max_action = [self.max_torque]

        self.observation_space = spaces.Box(np.array([-1, -1, -1]), np.array([1, 1, 1]))
        self.action_space = spaces.Box(np.array([-self.max_torque]), np.array([self.max_torque]))

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):

        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        b = self.b
        dt = self.dt
        I = self.I

        u = u[0]
        acceleration = -b/I * thdot - m * g * l/I * np.sin(th) + u/I
        newth = th + thdot * dt
        newthdot = thdot + acceleration * dt

        self.state = np.array([newth, newthdot])

        reward = self.get_ddp_reward(u)

        return self.state, reward

    def reset(self, reset_state=None):
        # TODO: make this choose random values centered around hover
        if reset_state is None:
            self.state = np.zeros((params.states,))
        else:
            self.state = reset_state
        return self.state

    # def _get_obs(self):
    #     theta, thetadot = self.state
    #     return np.array([np.cos(theta), np.sin(theta), thetadot])

    def get_ddp_reward(self, u):

        Q = self.Q_r_ddp
        R = self.R_ddp

        delta_x = self.state - np.squeeze(self.goal)

        cost = 0.5 * delta_x.T.dot(Q).dot(delta_x) + 0.5 * u * R * u

        # cost = 0.5 * np.matmul(delta_x.T, np.matmul(Q, delta_x)) + 0.5 * np.matmul(u.T, np.matmul(R, u))

        return cost


    def state_control_transition(self, x, u):
        """ takes in state and control trajectories and outputs the Jacobians for the linearized system
        edit function to use with autograd when linearizing the neural network output REBECCA """

        m = params.m
        L = params.L
        g = params.gr
        I = params.I
        b = params.b
        states = params.states
        controllers = params.num_controllers

        th = x[0]

        A = np.zeros([states, states])
        B = np.zeros([states, controllers])

        A[0, 1] = 1
        A[1, 0] = -m * g * L / I * np.cos(th)
        A[1, 1] = -b / I

        B[0, 0] = 0
        B[1, 0] = 1 / I

        return A, B