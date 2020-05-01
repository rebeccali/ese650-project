import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from environments import circlequad_mpc_params
import pdb


class SimpleQuadEnv(gym.Env):

    def __init__(self):
        self.dt = circlequad_mpc_params.dt
        self.timesteps = circlequad_mpc_params.timesteps
        self.total_time = circlequad_mpc_params.total_time

        self.g = circlequad_mpc_params.gr  # gravity
        self.J = circlequad_mpc_params.J  # moment of inertia
        self.L = circlequad_mpc_params.L  # length (m) from COM to thrust point of action
        self.m = circlequad_mpc_params.m

        self.states = circlequad_mpc_params.states
        self.num_controllers = circlequad_mpc_params.num_controllers

        # ddp parameters
        self.Q_r_ddp = circlequad_mpc_params.Q_r_ddp
        self.Q_f_ddp = circlequad_mpc_params.Q_f_ddp
        self.R_ddp = circlequad_mpc_params.R_ddp
        self.gamma = circlequad_mpc_params.gamma
        self.num_iter = circlequad_mpc_params.num_iter

        # state and control spaces
        self.state_limits = np.ones((circlequad_mpc_params.states,), dtype=np.float32) * 10000

        self.observation_space = spaces.Box(self.state_limits * -1, self.state_limits)
        self.action_space = spaces.Box(-10000, 10000, (4,))  # all 4 motors can be actuated 0 to 1

        # initialize state to zero
        self.state = np.zeros((circlequad_mpc_params.states,))
        # state: x,y,z, dx,dy,dz, r,p,y, dr,dp,dy

        self.goal_index = 0
        self.total_goal = circlequad_mpc_params.xf
        self.goal = circlequad_mpc_params.xf[self.goal_index:self.timesteps]

        pdb.set_trace()

    def reset(self, reset_state=None):
        # TODO: make this choose random values centered around hover
        if reset_state is None:
            self.state = np.zeros((circlequad_mpc_params.states,))
        else:
            # update the reset state to the correct initial state after forward MPC step
            self.state = reset_state

        return self.state

    def step(self, u):

        assert self.action_space.contains(u)

        state = self.state
        f1 = u[0]
        f2 = u[1]
        f3 = u[2]
        f4 = u[3]

        u1 = f1 + f2 + f3 + f4  # thrust force
        u2 = f4 - f2  # roll force
        u3 = f1 - f3  # pitch force
        u4 = 0.05 * (f2 + f4 - f1 - f3)  # yaw moment (TODO: WHERE DOES 0.05 come from??)

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

        state_dot = np.zeros((circlequad_mpc_params.states,))

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

        return cost

    def get_H(self):
        dx = self.state[3:6]
        dth = self.state[9:12]
        z = self.state[2]

        trans = (1 / 2) * self.m * np.dot(dx, dx)
        rot = (1 / 2) * (dth.T @ self.J) @ dth

        return trans + rot - self.m * self.g * z

    def get_pdot(self):
        pdot = np.zeros((6,))
        pdot[2] = self.m * self.g
        return pdot

    def get_qdot(self):
        return np.concatenate((self.state[3:6], self.state[9:12]))

    def set_goal(self):

        # reset goal trajectory
        self.goal_index += 1
        self.goal = circlequad_mpc_params.xf[self.goal_index:(self.goal_index + self.timesteps)]

    def state_control_transition(self, x, u):
        """ takes in state and control trajectories and outputs the Jacobians for the linearized system
        edit function to use with autograd when linearizing the neural network output REBECCA """

        m = circlequad_mpc_params.m
        L = circlequad_mpc_params.L
        J = circlequad_mpc_params.J
        Jx = J[0, 0]
        Jy = J[1, 1]
        Jz = J[2, 2]

        states = circlequad_mpc_params.states
        controllers = circlequad_mpc_params.num_controllers

        A = np.zeros([states, states])
        B = np.zeros([states, controllers])

        phi = x[6]
        theta = x[7]
        psi = x[8]
        phi_dot = x[9]
        theta_dot = x[10]
        psi_dot = x[11]

        f1 = u[0]
        f2 = u[1]
        f3 = u[2]
        f4 = u[3]

        u1 = f1 + f2 + f3 + f4  # total force
        u2 = f4 - f2  # roll actuation
        u3 = f1 - f3  # pitch actuation
        u4 = 0.05 * (f2 + f4 - f1 - f3)  # yaw moment

        A[0, 3] = 1
        A[1, 4] = 1
        A[2, 5] = 1
        A[6, 9] = 1
        A[7, 10] = 1
        A[8, 11] = 1

        A[3, 6] = (u1 * (np.cos(phi) * np.sin(psi) - np.cos(psi) * np.sin(phi) * np.sin(theta))) / m
        A[3, 7] = (u1 * np.cos(phi) * np.cos(psi) * np.cos(theta)) / m
        A[3, 8] = (u1 * (np.cos(psi) * np.sin(phi) - np.cos(phi) * np.sin(psi) * np.sin(theta))) / m

        A[4, 6] = -(u1 * (np.cos(phi) * np.cos(psi) + np.sin(phi) * np.sin(psi) * np.sin(theta))) / m
        A[4, 7] = (u1 * np.cos(phi) * np.cos(theta) * np.sin(psi)) / m
        A[4, 8] = (u1 * (np.sin(phi) * np.sin(psi) + np.cos(phi) * np.cos(psi) * np.sin(theta))) / m

        A[5, 6] = (u1 * np.cos(theta) * np.sin(phi)) / m

        A[9, 10] = (psi_dot * (Jy - Jz)) / Jx
        A[9, 11] = (theta_dot * (Jy - Jz)) / Jx

        A[10, 9] = -(psi_dot * (Jx - Jz)) / Jy
        A[10, 11] = -(phi_dot * (Jx - Jz)) / Jy

        A[11, 9] = (theta_dot * (Jx - Jy)) / Jz
        A[11, 10] = (phi_dot * (Jx - Jy)) / Jz

        B[3, 0] = (np.sin(phi) * np.sin(psi) + np.cos(phi) * np.cos(psi) * np.sin(theta)) / m
        B[3, 1] = (np.sin(phi) * np.sin(psi) + np.cos(phi) * np.cos(psi) * np.sin(theta)) / m
        B[3, 2] = (np.sin(phi) * np.sin(psi) + np.cos(phi) * np.cos(psi) * np.sin(theta)) / m
        B[3, 3] = (np.sin(phi) * np.sin(psi) + np.cos(phi) * np.cos(psi) * np.sin(theta)) / m

        B[4, 0] = -(np.cos(psi) * np.sin(phi) - np.cos(phi) * np.sin(psi) * np.sin(theta)) / m
        B[4, 1] = -(np.cos(psi) * np.sin(phi) - np.cos(phi) * np.sin(psi) * np.sin(theta)) / m
        B[4, 2] = -(np.cos(psi) * np.sin(phi) - np.cos(phi) * np.sin(psi) * np.sin(theta)) / m
        B[4, 3] = -(np.cos(psi) * np.sin(phi) - np.cos(phi) * np.sin(psi) * np.sin(theta)) / m

        B[5, 0] = -(np.cos(phi) * np.cos(theta)) / m
        B[5, 1] = -(np.cos(phi) * np.cos(theta)) / m
        B[5, 2] = -(np.cos(phi) * np.cos(theta)) / m
        B[5, 3] = -(np.cos(phi) * np.cos(theta)) / m

        B[9, 1] = -L / Jx
        B[9, 3] = L / Jx

        B[10, 0] = L / Jy
        B[10, 2] = -L / Jy

        B[11, 0] = -1 / Jz
        B[11, 1] = 1 / Jz
        B[11, 2] = -1 / Jz
        B[11, 3] = 1 / Jz

        # B[11, 0] = -1 / (20 * Jz)
        # B[11, 1] = 1 / (20 * Jz)
        # B[11, 2] = -1 / (20 * Jz)
        # B[11, 3] = 1 / (20 * Jz)

        return A, B

    def plot(self, xf, x, u, costvec):

        # translational states
        plt.figure(1)
        plt.subplot(231)
        plt.plot(x[0, :])
        plt.plot(xf[0] * np.ones([x.shape[1], ]), 'r')
        plt.title('x')

        plt.subplot(232)
        plt.plot(x[1, :])
        plt.plot(xf[1] * np.ones([x.shape[1], ]), 'r')
        plt.title('y')

        plt.subplot(233)
        plt.plot(x[2, :])
        plt.plot(xf[2] * np.ones([x.shape[1], ]), 'r')
        plt.title('z')

        plt.subplot(234)
        plt.plot(x[3, :])
        plt.plot(xf[3] * np.ones([x.shape[1], ]), 'r')
        plt.title('x dot')

        plt.subplot(235)
        plt.plot(x[4, :])
        plt.plot(xf[4] * np.ones([x.shape[1], ]), 'r')
        plt.title('y dot')

        plt.subplot(236)
        plt.plot(x[5, :])
        plt.plot(xf[5] * np.ones([x.shape[1], ]), 'r')
        plt.title('z dot')

        # rotational states

        plt.figure(2)
        plt.subplot(231)
        plt.plot(x[6, :])
        plt.plot(xf[6] * np.ones([x.shape[1], ]), 'r')
        plt.title('theta')

        plt.subplot(232)
        plt.plot(x[7, :])
        plt.plot(xf[7] * np.ones([x.shape[1], ]), 'r')
        plt.title('phi')

        plt.subplot(233)
        plt.plot(x[8, :])
        plt.plot(xf[8] * np.ones([x.shape[1], ]), 'r')
        plt.title('psi')

        plt.subplot(234)
        plt.plot(x[9, :])
        plt.plot(xf[9] * np.ones([x.shape[1], ]), 'r')
        plt.title('theta dot')

        plt.subplot(235)
        plt.plot(x[10, :])
        plt.plot(xf[10] * np.ones([x.shape[1], ]), 'r')
        plt.title('phi dot')

        plt.subplot(236)
        plt.plot(x[11, :])
        plt.plot(xf[11] * np.ones([x.shape[1], ]), 'r')
        plt.title('psi dot')

        # cost
        plt.figure(3)
        plt.plot(costvec[:])
        plt.title('cost over iterations')

        # control
        plt.figure(4)
        plt.subplot(411)
        plt.plot(u[0, :].T)
        plt.title('u opt output')

        plt.subplot(412)
        plt.plot(u[1, :].T)

        plt.subplot(413)
        plt.plot(u[2, :].T)

        plt.subplot(414)
        plt.plot(u[3, :].T)