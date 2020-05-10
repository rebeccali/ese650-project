from environments import learned_models

def construct_env(env_name):
    """ Construct environment from string"""
    if env_name in learned_models:
        # TODO(rebecca) make this less manual
        if env_name == 'LearnedPendulum-v0':
            from environments.learned_pendulum import LearnedPendulumEnv

            env = LearnedPendulumEnv(model_type='structure')
        elif env_name == 'LearnedQuad-v0':
            from environments.learned_quad import LearnedQuadEnv

            env = LearnedQuadEnv(model_type='structure')
        else:
            raise RuntimeError('Do not recognized learned model %s ' % env_name)
    else:
        import gym
        env = gym.make(env_name)
    return env