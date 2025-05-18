"""
This module implements the command-line interface for this project.
"""

import ast
import tyro
import logging
from pathlib import Path
from typing import Literal
from datetime import datetime
from pydantic import BaseModel
from comp_capacity.optim.run import run
from comp_capacity.optim.random_sample import SamplingParameters
from comp_capacity.optim.genetic_evolution import EvolutionParameters


def change_log_file_path(log_file_path: str) -> str:
    """Change the log file path to a new path."""
    logger = logging.getLogger("comp_capacity")

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
    )
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    handler = logging.FileHandler(f"{log_file_path}/comp_capacity_{now}.log")
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return log_file_path


def parse_dict(dict_str: str) -> dict:
    """Parse a JSON string into a dictionary."""
    if dict_str == "":
        return {}
    return ast.literal_eval(dict_str)


class Args(BaseModel):
    algorithm: Literal["random_sampling", "genetic_evolution"]
    """Which sampling algorithm to use."""
    weight_init: Literal["kaiming", "binary", "normal", "uniform"] = "kaiming"
    """Weight initialization method for the neural networks."""
    gym_env_name: str
    """Gym environment name, including our custom environments."""
    sampling_parameters: SamplingParameters
    """Sampling parameters."""
    evolution_parameters: EvolutionParameters
    """Evolution parameters."""
    seed: int = 0
    """Seed for the random number generator."""
    n_networks: int = 100
    """Number of networks to sample."""
    batch_size: int = 10
    """Number of networks to sample in each batch."""
    n_init_nodes: int = 5
    """Number of nodes to initialize the networks with."""
    save_folder: str = "data"
    """Folder to save the data to."""
    n_optim_steps: int = 1_000_000
    """Number of steps to run the optimization for."""
    gym_env_kwargs: str = ""
    """Extra parameters to pass to the gym environment. Will be parsed as a JSON string."""
    multi_step: bool = False
    """Whether the gym environment is a multi-step environment.
    If True, the reward is the sum of the rewards from each step."""
    save_buffer_size: int = 2000
    """Number of data entries to save in a single dataframe file."""


def main():
    args = tyro.cli(Args)
    gym_env_kwargs = parse_dict(args.gym_env_kwargs)

    Path(args.save_folder).mkdir(parents=True, exist_ok=True)
    _ = change_log_file_path(args.save_folder)

    run(
        n_networks=args.n_networks,
        batch_size=args.batch_size,
        gym_env_name=args.gym_env_name,
        gym_env_kwargs=gym_env_kwargs,
        gym_env_is_multi_step=args.multi_step,
        n_steps=args.n_optim_steps,
        n_init_nodes=args.n_init_nodes,
        algorithm=args.algorithm,
        sampling_parameters=args.sampling_parameters,
        extra_params={"evolution_parameters": args.evolution_parameters},
        data_save_folder=args.save_folder,
        seed=args.seed,
        save_buffer_size=args.save_buffer_size,
    )


if __name__ == "__main__":
    main()
