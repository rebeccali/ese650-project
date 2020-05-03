import gym
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
import numpy as np
from os import path

from environments import pendulum_params
import pdb



class PendulumEnv(gym.Env):

    def __init__(self):
        self.dt = pendulum_params.dt
        self.g = pendulum_params.gr
        self.m = pendulum_params.m
        self.l = pendulum_params.L
        self.b = pendulum_params.b
        self.I = pendulum_params.I

        self.timesteps = pendulum_params.timesteps

        self.num_iter = pendulum_params.num_iter

        self.Q_r_ddp = pendulum_params.Q_r_ddp
        self.Q_f_ddp = pendulum_params.Q_f_ddp
        self.R_ddp = pendulum_params.R_ddp
        self.gamma = pendulum_params.gamma
        self.states = pendulum_params.states
        self.num_controllers = pendulum_params.num_controllers

        # set initial and final states
        self.state = np.zeros((pendulum_params.states,))
        self.goal = pendulum_params.xf

        self.max_speed = pendulum_params.max_speed
        self.max_torque = pendulum_params.max_torque

        high = np.array([1., 1., self.max_speed])
        self.min_state = -high
        self.max_state = high
        self.min_action = [-self.max_torque]
        self.max_action = [self.max_torque]

        self.observation_space = spaces.Box(np.array([-1, -1, -self.max_speed]), np.array([1, 1, self.max_speed]))
        self.action_space = spaces.Box(np.array([-self.max_torque]), np.array([self.max_torque]))
        self.training_mode = False  # Lets us know if we're in training mode
        self.seed()

    def training_mode(self):
        """ Converts model to training mode """
        self.training_mode = True

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
        acceleration = -b / I * thdot - m * g * l / I * np.sin(th) + u / I
        newth = th + thdot * dt
        newthdot = thdot + acceleration * dt

        self.state = np.array([newth, newthdot])

        reward = self.get_ddp_reward(u)
        if self.training_mode:
            return self._get_obs(), reward, False, {}
        return self.state, reward

    def reset(self, reset_state=None):
        # TODO: make this choose random values centered around hover
        if self.training_mode:
            high = np.array([np.pi, 1])
            self.state = self.np_random.uniform(low=-high, high=high)
            self.last_u = None
        else:
            if reset_state is None:
                self.state = np.zeros((pendulum_params.states,))
            else:
                self.state = reset_state
        return self.state

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

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

        m = pendulum_params.m
        L = pendulum_params.L
        g = pendulum_params.gr
        I = pendulum_params.I
        b = pendulum_params.b
        states = pendulum_params.states
        controllers = pendulum_params.num_controllers

        th = x[0]

        A = np.zeros([states, states])
        B = np.zeros([states, controllers])

        A[0, 1] = 1
        A[1, 0] = -m * g * L / I * np.cos(th)
        A[1, 1] = -b / I

        B[0, 0] = 0
        B[1, 0] = 1 / I

        return A, B

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def plot(self, xf, x, u, costvec):

        plt.figure(1)
        plt.subplot(211)
        plt.plot(x[0, :])
        plt.plot(xf[0] * np.ones([self.timesteps, ]), 'r')
        plt.title('theta')

        plt.subplot(212)
        plt.plot(x[1, :])
        plt.plot(xf[1] * np.ones([self.timesteps, ]), 'r')
        plt.title('thetadot')

        plt.figure(2)
        plt.plot(costvec[:, 0, 0])
        plt.title('cost over iterations')

        plt.figure(3)
        plt.plot(u[0, :].T)
        plt.title('u opt output')