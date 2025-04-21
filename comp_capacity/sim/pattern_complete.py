"""
This module contains pattern completion tasks, including:
- Completing the next step of a 1D function
- Producing sequential patterns
- Associations
"""
import numpy as np
import gymnasium as gym
from comp_capacity.sim import Task
from typing import Literal

__all__ = ["NextStepFunction", "SequentialPatterns"]

Functions = Literal["linear", "sinusoidal", "exponential"]

# TODO: switch these to gynmasium environments (allows for one API for all tests)

class NextStepFunction(gym.Env):

    def __init__(self, function: Functions, steps: int):
        """
        Initialize the NextStepFunction task. Accepts a function name and number of steps. 
        """
        self.func = function
        self.steps = steps

        self.action_space = gym.spaces.Space()


    def step(self, action):
        """
        Perform a step in the task. This function will generate the next step in the sequence.
        """
        terminated = True
        truncated = False
        # return absolute difference between the action and the target 
        reward = -np.abs(obs['target'] - action)

        obs = self._get_obs()
        info = self._get_info()


        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        return {'data': self.data[:self.steps], 'target': self.data[self.steps]}
    
    def _get_info(self):
        return {'params': self.params}

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)
        if self.func == "linear":
            # set m, b
            self.params = (self.np_random.normal(), self.np_random.normal())
            self.data = np.arange(self.steps + 1) * self.params[0] + self.params[1]
        elif self.func == "sinusoidal":
            # set freq, phase, amplitude
            self.params = (self.np_random.normal(), self.np_random.normal(), self.np_random.normal())
            self.data = np.sin(np.arange(self.steps + 1) * self.params[0] + self.params[1]) * self.params[2]
        elif self.func == "exponential":
            # set a, b
            self.params = (self.np_random.normal(), self.np_random.normal())
            self.data = np.exp(self.params[0] * np.arange(self.steps + 1) + self.params[1])

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

class SequentialPatterns(Task):
    def __init__(self, patterns: list):
        """
        Initialize the SequentialPatterns task. Accepts a list of patterns.
        """
        pass

    def load(self):
        """
        Load the task. For this module, it generates a sequence of numbers based on the patterns.
        """
        pass

    def step(self):
        """
        Perform a step in the task. This function will generate the next step in the sequence.
        """
        pass

    def __iter__(self):
        """
        Make the task iterable. This allows for easy iteration over the task.
        """
        pass