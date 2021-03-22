import random
import numpy as np

class epsilon_greedy_strategy(object):

    def __init__(self, initial_epsilon = 1, decremental_epsilon = 1E-4, epsilon_min = 0.1, seed = 0):

        self.initial_epsilon = initial_epsilon
        self.decremental_epsilon = decremental_epsilon
        self.epsilon_min = epsilon_min
        self.epsilon = initial_epsilon
        self.seed = seed
        random.seed(seed)  # Fix the seed #

    def reset(self):

        self.epsilon = self.initial_epsilon

    def update_parameters(self):

        if self.epsilon <= self.epsilon_min:
            self.epsilon = self.epsilon_min
        else:
            self.epsilon -= self.decremental_epsilon

    def pick_action(self, action_values) -> int:

        assert type(action_values) is np.ndarray, "Action values must be numpy array"

        if random.random() < self.epsilon:  # Explore

            action = random.randint(0, len(action_values)-1)  # Choose a random action

        else: # Exploit

            action = action_values.argmax()

        return action


