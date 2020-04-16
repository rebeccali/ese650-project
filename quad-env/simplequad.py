import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class SimpleQuadEnv(gym.Env):
    
    
    def __init__(self):
        self.dt = 0.05
        