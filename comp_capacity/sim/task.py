from abc import ABC, abstractmethod

class Task(ABC):
    @abstractmethod
    def step(self, action):
        """
        Run a time step in the RL-like environment.

        Args:
            action: The action to take.

        Returns:
            A tuple containing the next state, reward, done flag, and any additional info.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, dataset_path):
        """
        Load the dataset from the given path.

        Args:
            dataset_path: The path to the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        """
        Make the data iterable across batches.
        """
        raise NotImplementedError
