"""Module for running the various optimization algorithms.
Currently supported:
- Evolutionary algorithms (genetic evolution)
- Random sampling
"""

import torch
import random
import logging
from typing import Literal
from comp_capacity.repr.network import Topology
from comp_capacity.optim.genetic_evolution import evolution_step
from comp_capacity.optim.random_sample import random_step, sample_topology, SamplingParameters


# TODO: set up logging and data storage

def evaluate(networks: list[Topology]) -> list[float]:
    logging.info(f"Evaluating {len(networks)} networks")
    # placeholder for actual evaluation
    return [1] * len(networks)


# a generic function to run any of the optimization algorithms
def run(
    n_networks: int,
    batch_size: int,
    n_steps: int,
    n_init_nodes: int,
    algorithm: Literal["genetic_evolution", "random_sampling"],
    sampling_parameters: SamplingParameters,
    extra_params: dict,
    seed: int | None = None,
):
    """
    Run the optimization algorithm for the given batch size.
    For evolutionary algorithms, n_networks is the number of individuals in the population.
    For random sampling, n_networks is the number of samples to draw.

    """
    if torch.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logging.info(f"Using device: {device}")

    rng = random.Random(seed)
    torch.manual_seed(seed)

    logging.info(f"Using random number generator with seed: {seed}. Set torch seed to {seed}.")

    if algorithm == "genetic_evolution":
        optim_step = evolution_step
        if "evolution_parameters" not in extra_params:
            raise ValueError("evolution_parameters must be provided as a key in extra_params for genetic evolution")
    elif algorithm == "random_sampling":
        optim_step = random_step
    else:
        raise ValueError(f"Algorithm {algorithm} not supported")
    
    logging.info(f"Running {algorithm} algorithm")

    # TODO: set up environment
    input_dim = 10
    output_dim = 2

    # setup initial population
    networks = [sample_topology(n_init_nodes, input_dim, output_dim, sampling_parameters, device=device, rng=rng) for _ in range(n_networks)]

    logging.info(f"Set up initial population of {n_networks} networks")

    for step in range(n_steps):
        logging.info(f"Running step {step} of {n_steps}")
        # run evaluation step
        loss = evaluate(networks)
        logging.info(f"Evaluation step {step} complete")
        # run network optimization step
        networks = optim_step(networks, loss, sampling_parameters, input_dim, output_dim, rng=rng, **extra_params)

        logging.info(f"Step {step} complete")
