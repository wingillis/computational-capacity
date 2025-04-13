"""
This module contains pattern completion tasks, including:
- Completing the next step of a 1D function
- Producing sequential patterns
- Associations
"""
from comp_capacity.sim import Task
from typing import Literal

__all__ = ["NextStepFunction", "SequentialPatterns"]

Functions = Literal["linear", "sinusoidal", "exponential"]

class NextStepFunction(Task):

    def __init__(self, function: Functions, steps: int):
        """
        Initialize the NextStepFunction task. Accepts a function name and number of steps. 
        """
        pass

    def load(self):
        """
        Load the task. For this module, it generates a sequence of numbers based on the function and steps.
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