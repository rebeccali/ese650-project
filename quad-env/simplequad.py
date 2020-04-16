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
        
        self.state_limits = np.ones((12,),dtype=np.float32)*10000 #initialize really large (no limits) for now
        
        self.observation_space = spaces.Box(self.state_limits*-1,self.state_limits)
        self.action_space = spaces.Box(0, 1, (4,)) #all 4 motors can be actuated 0 to 1
        