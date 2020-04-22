""" main file for quadrotor tracking with ddp controller """

import params
from simplequad import SimpleQuadEnv
from ddp_functions import *
import matplotlib.pyplot as plt

import pdb

if __name__ == "__main__":

    quad = SimpleQuadEnv()

    # set desired goal state
    xf = np.zeros([params.states, 1])
    xf[0, 0] = -1
    xf[1, 0] = 1
    xf[2, 0] = 0.5

    quad.set_goal(xf)

    num_iter = params.num_iter

    u = np.zeros([params.num_controllers, params.timesteps-1])
    du = np.zeros([params.num_controllers, params.timesteps-1])
    x = np.zeros([params.states, params.timesteps])

    costvec = []

    for i in range(num_iter):

        u_opt = ddp(quad, x, u)

        # simulate the real system over the prediction time horizon, this uses the step function from the system
        x_new, cost = apply_control(quad, u_opt)

        # update state and control trajectories
        x = x_new
        u = u_opt

        # reset the system so that the next optimization step starts from the correct initial state
        quad.reset()
        costvec.append(-cost)

        # reset the system so that the next optimization step starts from the correct initial state
        quad.reset()
        
        print('iteration: ', i, "cost: ", -cost)


    # translational states
    plt.figure(1)

    plt.subplot(231)
    plt.plot(x[0, :])
    plt.plot(xf[0]*np.ones([params.timesteps, ]), 'r')
    plt.title('x')

    plt.subplot(232)
    plt.plot(x[1, :])
    plt.plot(xf[1] * np.ones([params.timesteps, ]), 'r')
    plt.title('y')

    plt.subplot(233)
    plt.plot(x[2, :])
    plt.plot(xf[2] * np.ones([params.timesteps, ]), 'r')
    plt.title('z')

    plt.subplot(234)
    plt.plot(x[3, :])
    plt.plot(xf[3] * np.ones([params.timesteps, ]), 'r')
    plt.title('x dot')

    plt.subplot(235)
    plt.plot(x[4, :])
    plt.plot(xf[4] * np.ones([params.timesteps, ]), 'r')
    plt.title('y dot')

    plt.subplot(236)
    plt.plot(x[5, :])
    plt.plot(xf[5] * np.ones([params.timesteps, ]), 'r')
    plt.title('z dot')

    # rotational states
    plt.figure(2)

    plt.subplot(231)
    plt.plot(x[6, :])
    plt.plot(xf[6]*np.ones([params.timesteps, ]), 'r')
    plt.title('phi')

    plt.subplot(232)
    plt.plot(x[7, :])
    plt.plot(xf[7] * np.ones([params.timesteps, ]), 'r')
    plt.title('theta')

    plt.subplot(233)
    plt.plot(x[8, :])
    plt.plot(xf[8] * np.ones([params.timesteps, ]), 'r')
    plt.title('psi')

    plt.subplot(234)
    plt.plot(x[9, :])
    plt.plot(xf[9] * np.ones([params.timesteps, ]), 'r')
    plt.title('phi dot')

    plt.subplot(235)
    plt.plot(x[10, :])
    plt.plot(xf[10] * np.ones([params.timesteps, ]), 'r')
    plt.title('theta dot')

    plt.subplot(236)
    plt.plot(x[11, :])
    plt.plot(xf[11] * np.ones([params.timesteps, ]), 'r')
    plt.title('psi dot')

    # cost over iterations
    plt.figure(3)
    plt.plot(costvec)

    plt.show()


    # rotation states

    pdb.set_trace()

