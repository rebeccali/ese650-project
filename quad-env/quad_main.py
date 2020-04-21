""" main file for quadrotor tracking with ddp controller """

import params
from simplequad import SimpleQuadEnv
from ddp_functions import *

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



    # loop for ddp optimization
    for i in range(num_iter):
        u_opt = ddp(quad, x, u)

        # simulate the real system over the prediction time horizon, this uses the step function from the system
        x_new, cost = apply_control(quad, u_opt)

        # update trajectories
        x = x_new
        u = u_opt

        u_max = np.max(u)


        costvec.append(cost)

        # reset the system so that the next optimization step starts from the correct initial state
        quad.reset()

        costvec.append(cost)

        # reset the system so that the next optimization step starts from the correct initial state
        quad.reset()
        
        print('iteration: ', i, "cost: ", cost, "u max: ", u_max)

    pdb.set_trace()

