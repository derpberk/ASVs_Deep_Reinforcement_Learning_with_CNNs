import numpy as np
from gym import spaces

class dummy(object):

    def __init__(self):

        self.action_space = np.array([0.0,0.0])
        self.action_size = len(self.action_space)
        self.state_size = (4,20,20)
        self.action_space = spaces.Box(np.array([0, 0]), np.array([1, 1]))

    def step(self,action):
        print('Step performed')

        return np.zeros(shape=self.state_size), -1, True

    def reset(self):
        return np.zeros(shape=self.state_size)

