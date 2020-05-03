""" A Pendulum Learned using the SymplectivODENet thing"""
import numpy as np
import pathlib

from environments import learned_params, pendulum
from symplectic.analysis import get_one_step_prediction, np_to_integratable_type_1D, get_model
from symplectic.au_functional import jacobian
from symplectic.utils import ObjectView


class LearnedPendulumEnv(pendulum.PendulumEnv):

    def __init__(self, model_type='structure', verbose=True):
        """ Inputs: model type. Either structure or naive
            structure is the structured Sympode
            naive is the naive baseline model.
        """
        pendulum.PendulumEnv.__init__(self)

        parent_dir = pathlib.Path(__file__).parents[1].absolute()
        print("Current script directory: ", parent_dir)
        EXPERIMENT_DIR = str(parent_dir) + '/experiment_single_embed'
        print('Expeirment de %s' % EXPERIMENT_DIR)
        self.args = ObjectView({'num_angle': 1,  # Number of generalized coordinates
                'nonlinearity': 'tanh',  # NN nonlinearity
                'name': 'pend',  # name of environment
                'seed': 0,
                'save_dir': '{}'.format(EXPERIMENT_DIR),
                'fig_dir': '{}/figures'.format(parent_dir),
                'num_points': 2,  # number of evaluation points by ode solver, including initial point
                'gpu': 0,
                'solver': 'rk4',
                'env': 'LearnedPendulum-v0',  # Name of the gym environment
                })
        self.device = learned_params.get_device(self.args.gpu)

        # Fetch the model
        self.model_type = model_type
        if model_type == 'structure':
            self.model, self.stats = get_model(self.args, baseline=False, structure=True, naive=False,
                                   device=self.device, verbose=verbose)
        elif model_type == 'naive':
            self.model, self.stats = get_model(self.args, baseline=False, structure=False, naive=True,
                                                         device=self.device, verbose=verbose)
        else:
            raise RuntimeError('Model type %s not accepted' % model_type)

    def step(self, u):
        """ Do one step of simulation given an input u
        """
        th, thdot = self.state  # th := theta
        cos_q = np.cos(th)
        sin_q = np.sin(th)
        y0_u = np.array([cos_q, sin_q, thdot, u[0]])

        y1_u = get_one_step_prediction(self.model, y0_u, self.dt, device=self.device)

        new_cos_q = y1_u[0]
        new_sin_q = y1_u[1]
        new_thdot = y1_u[2]
        new_th = np.arctan2(new_sin_q, new_cos_q)

        self.state = np.array([new_th, new_thdot])

        reward = self.get_ddp_reward(u[0])

        return self.state, reward

    def state_control_transition(self, x, u):
        """ takes in state and control trajectories and outputs the Jacobians for the linearized system
        """
        assert x.shape[0] == self.states, "Expected x = [q, qdot], got something else."
        assert u.shape[0] == self.num_controllers, "Expected u = [torque], got something else."
        # First, find the model as evaluated at x, u
        # y0_u should be (4,) shape
        y0_u = np.array([np.cos(x[0]), np.sin(x[0]), x[1], u[0]])

        A, B = self.get_linearized_A_B(y0_u)

        return A, B

    def get_linearized_A_B(self, y0_u):
        """ Returns A,B linearized about y0_u
            y0_u is a numpy array of the current state x concatenated with the current input u
        """
        dfdy0 = self.get_full_jacobian(y0_u)
        # Note the jacobian is 4x4 instead of 4x3:
        # this is because the network returns [dcos(q), dsin(q), ddq, 0]
        # so for n controls, the n final rows should be ignored. They should
        # also be zero.
        dfdy0 = dfdy0[:-self.num_controllers, :]
        # first columns are A
        A = dfdy0[:, :-self.num_controllers]
        # last columns are B
        B = dfdy0[:, -self.num_controllers:]
        return A, B

    def get_full_jacobian(self, y0_u):
        """ Returns full jacobian (np.ndarray) of model with respect to y0_u (np array)"""
        y0_u = np_to_integratable_type_1D(y0_u, self.device)
        # Jacobian evaluated at y0
        t0 = 0.
        # Dynamics function to take jacobian of
        f = lambda y: self.model(t0, y)
        dfdy0 = jacobian(f, y0_u)  # Jacobian evaluated at y0_u
        dfdy0 = dfdy0[0, :, 0, :]  # Reshape to get rid of extra dimensions
        return dfdy0.detach().cpu().numpy()

    def render(self, mode='human'):
        print('UNIMPLEMENTED RENDER')
        pass
