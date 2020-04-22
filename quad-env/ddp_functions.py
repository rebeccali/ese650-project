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
    xf = np.squeeze(sys.goal)

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
    """ takes in state and control trajectories and outputs the Jacobians for the linearized system
    edit function to use with autograd when linearizing the neural network output REBECCA """

    m = params.m
    L = params.L
    J = params.J
    Jx = J[0, 0]
    Jy = J[1, 1]
    Jz = J[2, 2]

    states = params.states
    controllers = params.num_controllers

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

    B[11, 0] = -1 / (20 * Jz)
    B[11, 1] = 1 / (20 * Jz)
    B[11, 2] = -1 / (20 * Jz)
    B[11, 3] = 1 / (20 * Jz)

    return A, B


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

    states = params.states
    controllers = params.num_controllers
    timesteps = params.timesteps
    dt = params.dt

    xf = np.squeeze(sys.goal)

    print(xf)
    pdb.set_trace()
    

    Qf = params.Q_f_ddp

    q0 = np.zeros([1, timesteps - 1])
    qk = np.zeros([states, timesteps - 1])
    Qk = np.zeros([states, states, timesteps - 1])
    rk = np.zeros([controllers, timesteps - 1])
    Rk = np.zeros([controllers, controllers, timesteps - 1])
    Pk = np.zeros([controllers, states, timesteps - 1])
    A = np.zeros([states, states, timesteps - 1])
    B = np.zeros([states, controllers, timesteps - 1])

    V = np.zeros([1, timesteps])
    Vx = np.zeros([states, timesteps])
    Vxx = np.zeros([states, states, timesteps])

    u_new = np.zeros([controllers, timesteps - 1])

    for t in range(timesteps - 1):
        l0, lx, lxx, lu, luu, lux = running_cost(sys, x[:, t], u[:, t])

        q0[:, t] = dt * l0
        qk[:, t] = dt * lx
        Qk[:, :, t] = dt * lxx
        rk[:, t] = dt * lu
        Rk[:, :, t] = dt * luu
        Pk[:, :, t] = dt * lux

        dfx, dfu = state_control_transition(sys, x[:, t], u[:, t])

        A[:, :, t] = np.eye(states, states) + dfx * dt
        B[:, :, t] = dfu * dt

    # back prop for value function
    last_index = int(V.shape[1] - 1)

    V[:, last_index] = 0.5 * (x[:, last_index] - xf).T.dot(Qf).dot(x[:, last_index] - xf)
    Vx[:, last_index] = Qf.dot(x[:, last_index] - xf)
    Vxx[:, :, last_index] = Qf

    Lk = np.zeros([controllers, states, timesteps - 1])
    lk = np.zeros([controllers, timesteps - 1])

    for t in range((timesteps - 2), -1, -1):
        # get state action value function to evaluate the linearized bellman equation

        Q, Qx, Qu, Qxx, Quu, Qxu = state_action(q0[:, t], qk[:, t], rk[:, t], Qk[:, :, t], Rk[:, :, t],
                                                Pk[:, :, t].T, V[:, t + 1], Vx[:, t + 1], Vxx[:, :, t + 1], A[:, :, t],
                                                B[:, :, t])

        Lk[:, :, t] = np.linalg.solve(-Quu, Qxu.T)
        lk[:, t] = np.linalg.solve(-Quu, Qu)

        V[:, t] = Q + Qu.T.dot(lk[:, t]) + 1 / 2 * lk[:, t].T.dot(Quu).dot(lk[:, t])
        Vx[:, t] = Qx + Lk[:, :, t].T.dot(Qu) + Qxu.dot(lk[:, t]) + Lk[:, :, t].T.dot(Quu).dot(lk[:, t])
        Vxx[:, :, t] = Qxx + 2 * Lk[:, :, t].T.dot(Qxu.T) + Lk[:, :, t].T.dot(Quu).dot(Lk[:, :, t])

    dx = np.zeros([states, 1])

    for t in range(timesteps - 1):

        gamma = params.gamma

        du = lk[:, t] + np.squeeze(Lk[:, :, t].dot(dx))
        dx = np.squeeze(A[:, :, t].dot(dx)) + B[:, :, t].dot(du)

        u_new[:, t] = u[:, t] + gamma * du

    u_opt = u_new

    return u_opt


def apply_control(sys, u_opt):
    """ evaluates the controlled system trajectory """

    timesteps = params.timesteps
    states = params.states

    x_new = np.zeros([states, timesteps])
    x_new[:, 0] = sys.state
    cost = 0

    for t in range(timesteps - 1):
        u = u_opt[:, t]

        # returns next state and the reward of that state
        x1, c1 = sys.step(u)

        x_new[:, t] = x1
        cost += c1

    return x_new, cost
