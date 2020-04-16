import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import params

class SimpleQuadEnv(gym.Env):
    
    
    def __init__(self):
        self.dt = params.dt
        self.g = params.gr #gravity
        self.J = params.J #moment of inertia
        self.L = params.L #length (m) from COM to thrust point of action
        self.m = params.m
        
        