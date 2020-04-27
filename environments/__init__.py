from gym.envs.registration import register

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

# runs with pendulum_params.py
register(
    id='DDP-Pendulum-v0',
    entry_point='environments.pendulum:PendulumEnv',
)

register(
    id='Simple-Quad-v0',
    entry_point='environments.simplequad:SimpleQuadEnv',
)