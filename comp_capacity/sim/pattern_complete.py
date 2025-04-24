"""
This module contains pattern completion tasks, including:
- Completing the next step of a 1D function
- Producing sequential patterns
- Associations
"""

import numpy as np
import gymnasium as gym
from math import ceil
from typing import Literal

__all__ = ["NextStepFunction", "SequentialPatterns"]

Functions = Literal["linear", "sinusoidal", "exponential"]
Pattern = Literal["abab", "increase", "last"]


class NextStepFunction(gym.Env):

    def __init__(self, function: Functions, steps: int):
        """
        Initialize the NextStepFunction task. Accepts a function name and number of steps.
        """
        self.func = function
        self.steps = steps

        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(steps,), dtype=np.float64
        )

    def step(self, action):
        """
        Perform a step in the task.
        """
        terminated = True
        truncated = False
        # return absolute difference between the action and the target
        reward = -np.abs(self.target - action)

        info = self._get_info()
        obs = self._get_obs()

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        return self.data[: self.steps]

    def _get_info(self):
        return {"params": self.params, "function": self.func}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if self.func == "linear":
            # set m, b
            self.params = (self.np_random.normal(), self.np_random.normal())
            self.data = np.arange(self.steps + 1) * self.params[0] + self.params[1]
        elif self.func == "sinusoidal":
            # set freq, phase, amplitude
            self.params = (
                self.np_random.uniform() * 2,
                self.np_random.normal(),
                self.np_random.normal(),
            )
            self.data = (
                np.sin(np.arange(self.steps + 1) * self.params[0] + self.params[1])
                * self.params[2]
            )
        elif self.func == "exponential":
            # set a, b
            self.params = (self.np_random.normal(), self.np_random.normal())
            self.data = np.exp(
                self.params[0] * np.linspace(0, 2, self.steps + 1) + self.params[1]
            )
        self.target = self.data[-1]

        obs = self._get_obs()
        info = self._get_info()

        return obs, info


class SequentialPatterns(gym.Env):
    def __init__(self, pattern: Pattern, steps: int):
        """
        Initialize the SequentialPatterns task. Accepts a pattern name and a number of steps.
        """
        self.pattern = pattern
        self.steps = steps

        if pattern == "abab":
            self.action_space = gym.spaces.Discrete(3)
            self.observation_space = gym.spaces.Box(
                low=0, high=3, shape=(steps,), dtype=np.int64
            )
        else:
            self.action_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64
            )
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(steps,),
                dtype=np.float64 if pattern != "last" else np.int64,
            )

    def step(self, action):
        """
        Perform a step in the task.
        """
        terminated = True
        truncated = False

        if self.pattern == "abab":
            # return 1 if the action is correct, else 0
            reward = 1 if action == self.target else 0
        else:
            # return absolute difference between the action and the target for other patterns
            reward = -np.abs(self.target - action)

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        return self.data[: self.steps]

    def _get_info(self):
        return {"pattern": self.pattern}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if self.pattern == "abab":
            pattern = np.arange(3)
            self.np_random.shuffle(pattern)
            pattern = np.tile(pattern, ceil(self.steps / 3) + 1)

            self.data = pattern[: self.steps + 1].astype(np.int64)

        elif self.pattern == "increase":
            starting_num = self.np_random.integers(0, 10)
            self.data = np.arange(starting_num, starting_num + self.steps + 1).astype(np.float64)
        elif self.pattern == "last":
            self.data = self.np_random.integers(0, 10, size=self.steps)
            self.data = np.append(self.data, self.data[-1])

        self.target = self.data[-1]

        obs = self._get_obs()
        info = self._get_info()
        return obs, info
