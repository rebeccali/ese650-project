import numpy as np
import params
import pdb


def running_cost(sys, x, u):

    """
    :param sys: system from gym environment this stores the
    :param x: state trajectory
    :param u: control trajectory
    :return: gradients and Hessians of the loss function with respect to states and controls
    """

    xf = sys.goal

    states = params.states
    controllers = params.num_controllers
    R = params.R_ddp
    Qr = params.Q_r_ddp


    err = x - xf
    l0 = 0.5 * err.T.dot(Qr).dot(err) + 0.5 * u.T.dot(R).dot(u)
    lx = Qr.dot(err)
    lxx = Qr
    lu = R.dot(u)
    luu = R
    lux = np.zeros([controllers, states])

    return l0, lx, lxx, lu, luu, lux

def state_control_transition(sys, x, u):

    m = params.m
    L = params.L
    J = params.J
    Jx = J[0, 0]
    Jy = J[1, 1]
    Jz = J[2, 2]

    states = params.states
    controllers = params.num_controllers

    A = np.zeros([states, states])
    B = np.zeros([controllers, controllers])




    return dfx, dfu






def ddp(sys, x, u):
    """ takes in the current state and control trajectories and outputs optimal control trajectory """

    states = params.states
    controllers = params.num_controllers
    timesteps = params.timesteps
    dt = params.dt

    xf = sys.goal

    Qf = params.Q_f_ddp
    gamma = params.gamma

    q0 = np.zeros([1, timesteps-1])
    qk = np.zeros([states, timesteps-1])
    Qk = np.zeros([states, states, timesteps-1])

    rk = np.zeros([controllers, timesteps-1])
    Rk = np.zeros([controllers, controllers, timesteps-1])
    Pk = np.zeros([controllers, states, timesteps-1])

    A = np.zeros([states, states, timesteps-1])
    B = np.zeros([states, controllers, timesteps-1])

    V = np.zeros([1, timesteps])
    Vx = np.zeros([states, timesteps])
    Vxx = np.zeros([states, states, timesteps])

    u_new = np.zeros([controllers, timesteps-1])

    for t in range(timesteps-1):

        # get running cost gradients and hessians
        l0, lx, lxx, lu, luu, lux = running_cost(sys, x[:, t], u[:, t])

        q0[t] = dt * l0
        qk[:, t] = dt * lx
        Qk[:, :, t] = dt * lxx
        rk[:, t] = dt * lu
        Rk[:, :, t] = dt * luu
        Pk[:, :, t] = dt * lux

        # linearize dynamics
        dfx, dfu = state_control_transition(sys, x[:, t], u[:, t])




        pdb.set_trace()





    return u_opt


def apply_control(u_opt):




    return x_new




