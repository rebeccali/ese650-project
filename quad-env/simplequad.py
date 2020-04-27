import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import params
import pdb


class SimpleQuadEnv(gym.Env):

    def __init__(self):
        self.dt = params.dt
        self.g = params.gr  # gravity
        self.J = params.J  # moment of inertia
        self.L = params.L  # length (m) from COM to thrust point of action
        self.m = params.m
        self.Q_r_ddp = params.Q_r_ddp
        self.R_ddp = params.R_ddp

        self.state_limits = np.ones((params.states,), dtype=np.float32) * 10000  # initialize really large (no limits) for now

        self.observation_space = spaces.Box(self.state_limits * -1, self.state_limits)
        self.action_space = spaces.Box(-10000, 10000, (4,))  # all 4 motors can be actuated "infinitely"

        # initialize state to zero
        self.state = np.zeros((params.states,))
        # state: x,y,z, dx,dy,dz, r,p,y, dr,dp,dy

        self.goal = np.zeros((params.states,))  # TODO need to initialize this - maybe with
        # initialization of the environment itself? Or maybe a separate function?

    def reset(self, reset_state=None):
        # TODO: make this choose random values centered around hover
        if reset_state is None:
            self.state = np.zeros((params.states,))
        else:
            self.state = reset_state
        return self.state

    def step(self, u):

        # assert self.action_space.contains(u)

        state = self.state
        f1 = u[0]
        f2 = u[1]
        f3 = u[2]
        f4 = u[3]

        u1 = f1 + f2 + f3 + f4  # thrust force
        u2 = f4 - f2  # roll force
        u3 = f1 - f3  # pitch force
        u4 = 0.05 * (f2 + f4 - f1 - f3)  # yaw moment 
        # TODO : WHERE DOES 0.05 come from??)

        Jx = self.J[0, 0]
        Jy = self.J[1, 1]
        Jz = self.J[2, 2]

        x_dot = state[3]
        y_dot = state[4]
        z_dot = state[5]
        phi = state[6]
        theta = state[7]
        psi = state[8]
        phi_dot = state[9]
        theta_dot = state[10]
        psi_dot = state[11]

        state_dot = np.zeros((params.states,))

        # translational ------------------------------------------------------
        state_dot[0:3] = [x_dot, y_dot, z_dot]
        # position second derivative
        state_dot[3] = 1 / self.m * (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)) * u1
        state_dot[4] = 1 / self.m * (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)) * u1
        state_dot[5] = self.g - 1 / self.m * (np.cos(phi) * np.cos(theta)) * u1

        # rotational    ------------------------------------------------------
        state_dot[6:9] = [phi_dot, theta_dot, psi_dot]
        # angle second derivative
        state_dot[9] = 1 / Jx * (theta_dot * psi_dot * (Jy - Jz) + self.L * u2)
        state_dot[10] = 1 / Jy * (phi_dot * psi_dot * (Jz - Jx) + self.L * u3)
        state_dot[11] = 1 / Jz * (phi_dot * theta_dot * (Jx - Jy) + u4)

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

        # cost = 0.5 * np.matmul(delta_x.T, np.matmul(Q, delta_x)) + 0.5 * np.matmul(u.T, np.matmul(R, u))

        return -cost
    
    def get_H(self):
        dx = self.state[3:6]
        dth = self.state[9:12]
        z = self.state[2]
        
        trans = (1/2)*self.m*np.dot(dx,dx)
        rot = (1/2)*(dth.T @ self.J) @ dth
        
        return trans + rot - self.m*self.g*z
    
    def get_pdot(self):
        pdot = np.zeros((6,))
        pdot[2] = self.m*self.g
        return pdot

    def get_qdot(self):
        return np.concatenate((self.state[3:6],self.state[9:12]))

    def set_goal(self, goal):
        # takes a size (12,) numpy array representing the goal state against which rewards should be measured
        # stores the goal state in the environment class variables
        self.goal = goal

    def plot(self, state_history, control_history):
        # this should definitely be replaced by an actual render function in line with the "viewer" stuff that
        # gym provides. For now, just take in state history and control history, graph x,y,z,u. State history should be
        # 12xtsteps numpy array, this just plots the first 3 rows

        dt = self.dt
        t = np.arange(0, control_history.size * dt, dt)

        ax1 = plt.subplot(411)
        ax1.plot(t, state_history[0, :])

        ax2 = plt.subplot(412)
        ax2.plot(t, state_history[1, :])

        ax3 = plt.subplot(413)
        ax3.plot(t, state_history[2, :])

        ax4 = plt.subplot(414)
        ax4.plot(t, control_history)

        plt.show()
