""" A Quadrotor Learned using the SymplectivODENet thing"""
import numpy as np

from environments import learned_params, simplequad
from symplectic.analysis import get_one_step_prediction, np_to_integratable_type_1D, get_model
from symplectic.au_functional import jacobian
from symplectic.utils import ObjectView


class LearnedQuadEnv(simplequad.SimpleQuadEnv):

    def __init__(self, model_type='structure', verbose=False):
        """ Inputs: model type. Either structure or naive
            structure is the structured Sympode
            naive is the naive baseline model.
        """
        simplequad.SimpleQuadEnv.__init__(self)
        # TODO(rebecca): rewrite these correctly with a dictionary or something instead of pulling from params
        # convert goal to embedded states
        self.goal = to_embed_state(self.goal)
        def error(x): raise Exception(x) # TODO: remove lol
        self.states = error('Unimplemented')  # TODO: fix, shold be number of embedded states
        self.state = np.zeros((self.states))
        self.Q_r_ddp = np.zeros([self.states, self.states])
        self.Q_f_ddp = error('Unimplemented')
        self.Q_f_ddp = np.diag([100, 100, 1])  # TODO: fix to be the right number of states

        # Set up the model arguments
        EXPERIMENT_DIR = '/experiment_simple_quad'
        # TODO: fill out args correctly
        self.args = error('Need to go filll out args correctly in learned_params')
        #self.args =  ObjectView(learned_params.get_quad_args(EXPERIMENT_DIR, 'LearnedQuad-v0'))
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
        # TODO(rebecca): make this a super class and use inheiritance
        """
        # Assemble combined state-input vector
        y0_u = np.hstack([self.state, u])
        # Predict the next step
        y1_u = get_one_step_prediction(self.model, y0_u, self.dt, device=self.device)

        self.state = y1_u[0:self.states]

        reward = self.get_ddp_reward(u)
        return self.state, reward

    def state_control_transition(self, x, u):
        """ takes in state and control trajectories and outputs the Jacobians for the linearized system
        """
        raise Exception('Unimplemented asserts for x, u size instate control trans')
        assert x.shape[0] == self.states, ("Expected x = [cosq, sinq, qdot], got ", x)  # TODO
        assert u.shape[0] == self.num_controllers, ("Expected u = [torque], got ", u)  # TODO
        # First, find the model as evaluated at x, u
        # y0_u should be (4,) shape
        y0_u = np.hstack([x, u])
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

    def reset(self, *args, **kwargs):
        # Pass through arguments but then convert the state
        self.state = to_embed_state(super(LearnedQuadEnv, self).reset(*args, **kwargs))
        return self.state

    def _get_obs(self):
        """ Returns the observation. It is different from the superclass since it is the embedded state."""
        return self.state

    def render(self, mode='human'):
        print('UNIMPLEMENTED RENDER')
        pass


def to_embed_state(x):
    """ converts from state to embedded angle state"""
    # TODO
    raise Exception('Unimplemented')
    return np.array([np.cos(x[0]), np.sin(x[0]), x[1]])  # TODO
