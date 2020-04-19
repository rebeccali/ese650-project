import numpy as np
import params

import pdb

class DDP():

    def __init__(self):
        self.num_iterations = params.num_iter
        self.controllers = params.num_controllers
        self.Qr = params.Q_r_ddp
        self.Qf = params.Q_f_ddp
        self.R = params.R_ddp


    # main ddp loop
    def ddp(self):
        # initialize the control and state trajectories



        for i in range(self.num_iterations):












        return u_opt, cost


        pdb.set_trace()

