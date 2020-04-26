import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
import matplotlib.pyplot as plt
import params
import pdb


# from open ai gym source:
class PendulumEnv(gym.Env):

    def __init__(self, g=10.0):
        self.max_speed = params.max_v
        self.max_torque = params.max_u
        self.dt = params.dt
        self.g = params.gr
        self.m = params.m
        self.l = params.L

        self.Q_r_ddp = params.Q_r_ddp
        self.R_ddp = params.R_ddp

        self.state_limits = np.ones((params.states,), dtype=np.float32) * 10000  # initialize really large (no limits) for now

        self.observation_space = spaces.Box(self.state_limits * -1, self.state_limits)
        self.action_space = spaces.Box(-10000, 10000, (4,))  # all 4 motors can be actuated 0 to 1

        # set initial and final states
        self.state = np.zeros((params.states,))
        self.goal = params.xf

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)  # pylint: disable=E1111

        self.state = np.array([newth, newthdot])

        reward = self.get_ddp_reward(u)

        return self.state, reward

        # return self._get_obs(), -costs, False, {}

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

        cost = 0.5 * delta_x.T.dot(Q).dot(delta_x) + 0.5 * u.T.dot(R).dot(u)

        # cost = 0.5 * np.matmul(delta_x.T, np.matmul(Q, delta_x)) + 0.5 * np.matmul(u.T, np.matmul(R, u))

        return -cost

# def angle_normalize(x):
#     return (((x + np.pi) % (2 * np.pi)) - np.pi)
