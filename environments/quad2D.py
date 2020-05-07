import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from environments import quad2D_params
import pdb


class Quad2DEnv(gym.Env):

    def __init__(self):
        self.dt = quad2D_params.dt
        self.timesteps = quad2D_params.timesteps

        self.g = quad2D_params.gr  # gravity
        self.J = quad2D_params.J  # moment of inertia
        self.L = quad2D_params.L  # length (m) from COM to thrust point of action
        self.m = quad2D_params.m

        self.states = quad2D_params.states
        self.num_controllers = quad2D_params.num_controllers

        # ddp parameters
        self.Q_r_ddp = quad2D_params.Q_r_ddp
        self.Q_f_ddp = quad2D_params.Q_f_ddp
        self.R_ddp = quad2D_params.R_ddp
        self.gamma = quad2D_params.gamma
        self.num_iter = quad2D_params.num_iter

        # state and control spaces
        self.state_limits = np.ones((quad2D_params.states,), dtype=np.float32) * 10000

        self.observation_space = spaces.Box(self.state_limits * -1, self.state_limits)
        self.action_space = spaces.Box(-10000, 10000, (2,))  # all 4 motors can be actuated 0 to 1

        # initialize state to zero
        self.state = np.zeros((quad2D_params.states,))
        # state: x,y,z, dx,dy,dz, r,p,y, dr,dp,dy

        self.goal = quad2D_params.xf

    def reset(self, reset_state=None):
        # TODO: make this choose random values centered around hover
        if reset_state is None:
            self.state = np.zeros((quad2D_params.states,))
        else:
            self.state = reset_state
        return self.state

    def step(self, u):

        assert self.action_space.contains(u)

        state = self.state
        th = state[4]
        R = np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
        #print(R)
        
        state_dot = np.zeros((quad2D_params.states,))

        state_dot[0:2] = state[2:4]
        state_dot[4] = state[5]

        pos_dd = (1/self.m) *R @ np.array([[0],[u[0]+u[1]]]) - np.array([[0],[self.g]])

        #print(pos_dd.reshape((2,)))
        state_dot[2:4] = np.squeeze(pos_dd)
        
        state_dot[5] = 1/self.J*(u[1]-u[0])*self.L
        
        #print(state_dot)
        state_dot[2:4] = np.squeeze(pos_dd)
        
        state_dot[5] = 1/self.J*(u[1]-u[0])*self.L
        
        # propogate state forward using state_dot
        new_state = self.state + self.dt * state_dot

        self.state = new_state

        reward = self.get_ddp_reward(u)

        return self.state, reward

    def get_ddp_reward(self, u):
        Q = self.Q_r_ddp
        R = self.R_ddp

        delta_x = self.state - np.squeeze(self.goal)

        cost = 0.5 * delta_x.T.dot(Q).dot(delta_x) + 0.5 * u.T.dot(R).dot(u)

        return cost
    
    def get_H(self):
        dx = self.state[2:4]
        dth = self.state[5]
        y = self.state[1]
        
        trans = (1/2)*self.m*np.dot(dx,dx)
        rot = (1/2)*(dth**2) * self.J
        
        return trans + rot + self.m*self.g*y
    
    def get_pdot(self):
        pdot = np.zeros((3,))
        pdot[1] = -self.m*self.g
        return pdot

    def get_qdot(self):
        return np.concatenate((self.state[2:4],self.state[5]))

    def set_goal(self, goal):
        self.goal = goal

    def state_control_transition(self, x, u):
        """ takes in state and control trajectories and outputs the Jacobians for the linearized system
        edit function to use with autograd when linearizing the neural network output REBECCA """

        m = quad2D_params.m
        L = quad2D_params.L
        J = quad2D_params.J
        
        th = x[4]
        u1 = u[0]
        u2 = u[1]
        
        A = np.zeros((6,6))
        
        A[3:6,3:6] = np.eye(3)
        A[3,5] = -np.cos(th)*(u1+u2)
        A[4,5] = -np.sin(th)*(u1+u2)
        B = np.zeros((6,2))
        B[3:6,:] = np.array([[-np.sin(th), -np.sin(th)],[np.cos(th),np.cos(th)],[-L/J, L/J]])

        return A, B

    def plot(self, xf, x, u, costvec):

        # translational states
        plt.figure(1)
        plt.subplot(231)
        plt.plot(x[0, :])
        plt.plot(xf[0] * np.ones([self.timesteps, ]), 'r')
        plt.title('x')

        plt.subplot(232)
        plt.plot(x[1, :])
        plt.plot(xf[1] * np.ones([self.timesteps, ]), 'r')
        plt.title('y')

        plt.subplot(233)
        plt.plot(x[2, :])
        plt.plot(xf[2] * np.ones([self.timesteps, ]), 'r')
        plt.title('x dot')

        plt.subplot(234)
        plt.plot(x[3, :])
        plt.plot(xf[3] * np.ones([self.timesteps, ]), 'r')
        plt.title('y dot')

        plt.subplot(235)
        plt.plot(x[4, :])
        plt.plot(xf[4] * np.ones([self.timesteps, ]), 'r')
        plt.title('theta')

        plt.subplot(236)
        plt.plot(x[5, :])
        plt.plot(xf[5] * np.ones([self.timesteps, ]), 'r')
        plt.title('theta dot')

        # cost
        plt.figure(3)
        plt.plot(costvec[:])
        plt.title('cost over iterations')

        # control
        plt.figure(4)
        plt.subplot(211)
        plt.plot(u[0, :].T)
        plt.title('u opt output')

        plt.subplot(212)
        plt.plot(u[1, :].T)

