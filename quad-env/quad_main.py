""" main file for quadrotor tracking with ddp controller """

import params
from simplequad import SimpleQuadEnv
from ddp_functions import *

import pdb

if __name__ == "__main__":

    quad = SimpleQuadEnv()

    num_iter = params.num_iter
    u = np.zeros([params.num_controllers, params.timesteps-1])
    du = np.zeros([params.num_controllers, params.timesteps-1])
    x = np.zeros([params.states, params.timesteps])


    # loop for ddp optimization
    for i in range(num_iter):
        u_opt = ddp(quad, x, u)




    pdb.set_trace()

