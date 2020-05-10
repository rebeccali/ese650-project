from gym.envs.registration import register


# UPDATE THIS IF YOU ADD A LEARNED MODEL
learned_models = ['LearnedPendulum-v0', 'LearnedQuad-v0']

register(
    id='My_FA_Acrobot-v0',
    entry_point='environments.fa_acrobot:AcrobotEnv',
)

register(
    id='My_FA_CartPole-v0',
    entry_point='environments.fa_cartpole:CartPoleEnv',
)

register(
    id='MyCartPole-v0',
    entry_point='environments.cartpole:CartPoleEnv',
)

register(
    id='MyAcrobot-v0',
    entry_point='environments.acrobot:AcrobotEnv',
)

# Symplectic-ODENet reference pendulum
register(
    id='MyPendulum-v0',
    entry_point='environments.symp_pendulum:PendulumEnv',
)

# Pendulum using the learned model
register(
    id='LearnedPendulum-v0',
    entry_point='environments.learned_pendulum:PendulumEnv',
)

# runs with pendulum_params.py and ddp_main.py
register(
    id='DDP-Pendulum-v0',
    entry_point='environments.pendulum:PendulumEnv',
)

# runs with mpc_pendulum_params.py and mpc_ddp_main.py
register(
    id='MPC-DDP-Pendulum-v0',
    entry_point='environments.pendulum_mpc:PendulumEnv',
)

# runs with quadrotor_params.py adn ddp_main.py
register(
    id='Simple-Quad-v0',
    entry_point='environments.simplequad:SimpleQuadEnv',
)

# runs with simplequad_mpc_params.py adn ddp_main.py
register(
    id='MPC-DDP-Simple-Quad-v0',
    entry_point='environments.simplequad_mpc:SimpleQuadEnv',
)

# runs with circlequad_mpc.py and circlequad_mpc.py
register(
    id='MPC-DDP-Circle-Quad-v0',
    entry_point='environments.circlequad_mpc:SimpleQuadEnv',
)

register(
    id='Quad2D-v0',
    entry_point = 'environments.quad2D:Quad2DEnv',
    )
