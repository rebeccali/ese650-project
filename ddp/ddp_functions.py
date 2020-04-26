import numpy as np
from ddp import ddp_params
import pdb


def running_cost(sys, x, u):
    """
    :param sys: system from gym environment this stores the
    :param x: state trajectory
    :param u: control trajectory
    :return: gradients and Hessians of the loss function with respect to states and controls

    """
    xf = np.squeeze(sys.goal)

    states = sys.states
    controllers = sys.num_controllers
    R = sys.R_ddp
    Qr = sys.Q_r_ddp

    err = x - xf

    l0 = 0.5 * err.T.dot(Qr).dot(err) + 0.5 * u.T.dot(R).dot(u)
    lx = Qr.dot(err)
    lxx = Qr
    lu = R.dot(u)
    luu = R
    lux = np.zeros([controllers, states])
    lxu = lux.T

    return l0, lx, lxx, lu, luu, lux, lxu


def state_action(L, Lx, Lu, Lxx, Luu, Lxu, V, Vx, Vxx, phi, B):
    """ takes in the value function, loss and the gradients and Hessians and evaluates
    the state action value function """

    Q = L + V
    Qx = Lx + phi.T.dot(Vx)
    Qu = Lu + B.T.dot(Vx)
    Qxx = Lxx + phi.T.dot(Vxx).dot(phi)
    Quu = Luu + B.T.dot(Vxx).dot(B)
    Qxu = Lxu + phi.T.dot(Vxx).dot(B)

    return Q, Qx, Qu, Qxx, Quu, Qxu


def ddp(sys, x, u):
    """ takes in the current state and control trajectories and outputs optimal control trajectory """

    states = sys.states
    controllers = sys.num_controllers
    timesteps = ddp_params.timesteps
    dt = ddp_params.dt

    xf = np.squeeze(sys.goal)

    Qf = sys.Q_f_ddp

    q0 = np.zeros([1, timesteps - 1])
    qk = np.zeros([states, timesteps - 1])
    Qk = np.zeros([states, states, timesteps - 1])
    rk = np.zeros([controllers, timesteps - 1])
    Rk = np.zeros([controllers, controllers, timesteps - 1])
    Pk = np.zeros([states, controllers, timesteps - 1])

    A = np.zeros([states, states, timesteps - 1])
    B = np.zeros([states, controllers, timesteps - 1])

    V = np.zeros([1, timesteps])
    Vx = np.zeros([states, timesteps])
    Vxx = np.zeros([states, states, timesteps])

    u_new = np.zeros([controllers, timesteps - 1])

    for t in range(timesteps - 1):
        l0, lx, lxx, lu, luu, lux, lxu = running_cost(sys, x[:, t], u[:, t])

        q0[:, t] = dt * l0
        qk[:, t] = dt * lx
        Qk[:, :, t] = dt * lxx
        rk[:, t] = dt * lu
        Rk[:, :, t] = dt * luu
        Pk[:, :, t] = dt * lxu

        dfx, dfu = sys.state_control_transition(x[:, t], u[:, t])

        A[:, :, t] = np.eye(states) + dfx * dt
        B[:, :, t] = dfu * dt

    # back prop for value function
    last_index = int(V.shape[1] - 1)
    err = x[:, last_index] - xf

    V[:, last_index] = err.T.dot(Qf).dot(err)
    Vx[:, last_index] = Qf.dot(err)
    Vxx[:, :, last_index] = Qf

    Lk = np.zeros([controllers, states, timesteps])
    lk = np.zeros([controllers, timesteps])

    for t in range((timesteps - 2), -1, -1):
        # get state action value function to evaluate the linearized bellman equation

        Q, Qx, Qu, Qxx, Quu, Qxu = state_action(q0[:, t], qk[:, t], rk[:, t], Qk[:, :, t], Rk[:, :, t],
                                                Pk[:, :, t], V[:, t + 1], Vx[:, t + 1], Vxx[:, :, t + 1],
                                                A[:, :, t], B[:, :, t])

        # Lk[:, :, t] = -1 * np.linalg.solve(Quu, Qxu.T)
        # lk[:, t] = -1 * np.linalg.solve(Quu, Qu)

        Lk[:, :, t] = -1 * np.linalg.inv(Quu).dot(Qxu.T)
        lk[:, t] = -1 * np.linalg.inv(Quu).dot(Qu)

        V[:, t] = Q + Qu.T.dot(lk[:, t]) + 1 / 2 * lk[:, t].dot(Quu).dot(lk[:, t])
        Vx[:, t] = Qx + Lk[:, :, t].T.dot(Qu) + Qxu.dot(lk[:, t]) + Lk[:, :, t].T.dot(Quu).dot(lk[:, t])
        Vxx[:, :, t] = Qxx + 2 * Lk[:, :, t].T.dot(Qxu.T) + Lk[:, :, t].T.dot(Quu).dot(Lk[:, :, t])

    dx = np.zeros([states, 1])

    for t in range(timesteps - 1):
        gamma = ddp_params.gamma

        du = lk[:, t] + np.squeeze(Lk[:, :, t].dot(dx))
        dx = np.squeeze(A[:, :, t].dot(dx)) + B[:, :, t].dot(du)

        u_new[:, t] = u[:, t] + gamma * du

    u_opt = u_new

    return u_opt


def apply_control(sys, u_opt):
    """ evaluates the controlled system trajectory """

    timesteps = ddp_params.timesteps
    states = sys.states

    x_new = np.zeros([states, timesteps])
    x_new[:, 0] = sys.state
    cost = 0

    for t in range(timesteps - 1):
        u = u_opt[:, t]

        # returns next state and the reward of that state
        x1, c1 = sys.step(u)

        x_new[:, t + 1] = x1
        cost += c1

    return x_new, -cost
