import numpy as np
import pdb


def running_cost(env, x, u):
    """
    :param env: environment from gym environment this stores the
    :param x: state trajectory
    :param u: control trajectory
    :return: gradients and Hessians of the loss function with respect to states and controls

    """
    xf = np.squeeze(env.goal)

    states = env.states
    controllers = env.num_controllers
    R = env.R_ddp
    Qr = env.Q_r_ddp

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


def ddp(env, x, u, opt_time):
    """ takes in the current state and control trajectories and outputs optimal control trajectory """

    states = env.states
    controllers = env.num_controllers

    # these come from the params file of the environment
    # timesteps = env.timesteps
    timesteps = opt_time

    dt = env.dt

    xf = np.squeeze(env.goal)

    # if env.goal.shape[1] == 1:
    #     xf = np.squeeze(env.goal)
    # else:
    #     xf_traj = env.goal # gives the chunk of the goal state trajectory
    #     xf = env.goal[:, env.goal.shape[1]] # use the last state in the value function initialization for the backward pass


    Qf = env.Q_f_ddp

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

        l0, lx, lxx, lu, luu, lux, lxu = running_cost(env, x[:, t], u[:, t])

        q0[:, t] = dt * l0
        qk[:, t] = dt * lx
        Qk[:, :, t] = dt * lxx
        rk[:, t] = dt * lu
        Rk[:, :, t] = dt * luu
        Pk[:, :, t] = dt * lxu

        dfx, dfu = env.state_control_transition(x[:, t], u[:, t])

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

        Lk[:, :, t] = -1 * np.linalg.inv(Quu).dot(Qxu.T)
        lk[:, t] = -1 * np.linalg.inv(Quu).dot(Qu)

        V[:, t] = Q + Qu.T.dot(lk[:, t]) + 1 / 2 * lk[:, t].dot(Quu).dot(lk[:, t])
        Vx[:, t] = Qx + Lk[:, :, t].T.dot(Qu) + Qxu.dot(lk[:, t]) + Lk[:, :, t].T.dot(Quu).dot(lk[:, t])
        Vxx[:, :, t] = Qxx + 2 * Lk[:, :, t].T.dot(Qxu.T) + Lk[:, :, t].T.dot(Quu).dot(Lk[:, :, t])

    dx = np.zeros([states, 1])

    for t in range(timesteps - 1):
        gamma = env.gamma

        du = lk[:, t] + np.squeeze(Lk[:, :, t].dot(dx))
        dx = np.squeeze(A[:, :, t].dot(dx)) + B[:, :, t].dot(du)

        u_new[:, t] = u[:, t] + gamma * du

    u_opt = u_new

    return u_opt


def apply_control(env, u_opt, timesteps):
    """ evaluates the controlled envtem trajectory """

    states = env.states

    # if not MPC:
    # timesteps = env.timesteps

    x_new = np.zeros([states, timesteps])
    x_new[:, 0] = env.state
    cost = 0

    for t in range(timesteps - 1):
        u = u_opt[:, t]

        # returns next state and the reward of that state
        x1, c1 = env.step(u)

        x_new[:, t + 1] = x1
        cost += c1

    # else:
    #     # returns next state and the reward of that state
    #     u = u_opt[:, 0]
    #     cost = 0
    #
    #     x1, c1 = env.step(u)
    #
    #     x_new = x1
    #     cost += c1

    return x_new, -cost


def run_ddp(env, num_iter, timesteps):
    """Run DDP on environment for num_iter timesteps
        Returns:
            costvec(nparray): costs at each timestep
            u(nparray): the control trajectory
            x(nparray): the state trajectory
            xf(nparray): the system goal
    """
    x = np.zeros([env.states, env.timesteps])
    u = np.zeros([env.num_controllers, env.timesteps - 1])
    costvec = []
    for i in range(num_iter):
        u_opt = ddp(env, x, u, env.timesteps)
        x_new, cost = apply_control(env, u_opt)

        # update state and control trajectories
        x = x_new
        u = u_opt

        # reset the system so that the next optimization step starts from the correct initial state
        costvec.append(-cost)

        # reset the system so that the next optimization step starts from the correct initial state
        env.reset()

        print('iteration: ', i, "cost: ", -cost)
    xf = env.goal
    x = np.asarray(x)
    u = np.asarray(u)
    costvec = np.asarray(costvec)
    return costvec, u, x, xf


def run_mpc_ddp(env, num_iter, opt_time):
    """Run MPC DDP on environment for num_iter timesteps
        Returns:
            costvec(nparray): costs at each timestep
            u(nparray): the control trajectory
            x(nparray): the state trajectory
            xf(nparray): the system goal
    """
    ################################ MPC stufff ####################################################

    current_time = 0
    index = 0
    prediction_time_horizon = int(env.timesteps * opt_time)
    total_time = int(env.timesteps * env.dt)


    costvec = []
    x = []
    x.append(env.state) # set initial state
    u = []

    x1 = env.state

    while current_time <  (total_time - env.dt):

        x_ddp = np.zeros([env.states, int(prediction_time_horizon)])
        x_ddp[:, 0] = x1
        u_ddp = np.zeros([env.num_controllers, int(prediction_time_horizon)])

        for i in range(num_iter):
            u_opt = ddp(env, x_ddp, u_ddp, prediction_time_horizon)

            x_new, cost = apply_control(env, u_opt, prediction_time_horizon)

            # update state and control trajectories
            x_ddp = x_new
            u_ddp = u_opt

            env.reset(reset_state=x1)

        # apply first control from the sequence and step one timestep
        x1, c1 = env.step(u_ddp[:, 0])

        x.append(x1)
        u.append(u_ddp[:, 0])
        costvec.append(-c1)

        env.reset(reset_state=x1)

        current_time += env.dt
        index += 1

        print('MPC Iteration: ', index, "Cost: ", -c1, 'curr time: ', current_time)

    xf = env.goal
    x = np.asarray(x)
    u = np.asarray(u)
    costvec = np.asarray(costvec)

    return costvec, u, x, xf

















