"""Module for running the various optimization algorithms.
Currently supported:
- Evolutionary algorithms (genetic evolution)
- Random sampling
"""

import torch
import random
import logging
import numpy as np
import comp_capacity  # noqa: F401
import gymnasium as gym
from typing import Literal
from comp_capacity.repr.network import Topology, ProgressiveRNN
from comp_capacity.optim.genetic_evolution import evolution_step
from comp_capacity.optim.random_sample import (
    random_step,
    sample_topology,
    SamplingParameters,
)


# TODO: set up logging and data storage


def evaluate(modules: list[ProgressiveRNN], envs: gym.Env, seed: int) -> list[float]:
    device = modules[0].device
    is_categorical = isinstance(
        envs.action_space, (gym.spaces.MultiDiscrete, gym.spaces.Discrete)
    )
    if is_categorical:
        logging.info(f"Action space is categorical: {envs.action_space}")

    # TODO: allow for multiple time steps
    logging.info(f"Evaluating {len(modules)} modules")
    scores = []
    for module in modules:
        # reset environment to same state for each network
        obs, info = envs.reset(seed=seed)
        obs = torch.from_numpy(obs).float().to(device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        state = None
        module.eval()
        with torch.no_grad():
            try:
                action, state = module(obs, state=state)
            except RuntimeError as e:
                logging.error(f"Error in forward pass: {e}")
                print("Observation")
                print(obs)
                print("Inputs projection")
                print(module.constructor_matrices.input.adjacency)
                print("Outputs projection")
                print(module.constructor_matrices.output.adjacency)
                print("Inner projection")
                print(module.constructor_matrices.inner.adjacency)
                for name, param in module.named_parameters():
                    print(name)
                    print(param)
                raise e
            if is_categorical:
                action = torch.distributions.Categorical(logits=action).sample()
            action = action.squeeze().cpu().numpy()
            # assume we only take one time step
            obs, reward, terminated, truncated, info = envs.step(action)
            scores.append(reward)
    return torch.tensor(np.array(scores))


# a generic function to run any of the optimization algorithms
def run(
    n_networks: int,
    batch_size: int,
    gym_env_name: str,
    gym_env_kwargs: dict,
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

    logging.info(
        f"Using random number generator with seed: {seed}. Set torch seed to {seed}."
    )

    if algorithm == "genetic_evolution":
        optim_step = evolution_step
        if "evolution_parameters" not in extra_params:
            raise ValueError(
                "evolution_parameters must be provided as a key in extra_params for genetic evolution"
            )
    elif algorithm == "random_sampling":
        optim_step = random_step
    else:
        raise ValueError(f"Algorithm {algorithm} not supported")

    logging.info(f"Running {algorithm} algorithm")

    envs = gym.make_vec(
        gym_env_name, num_envs=batch_size, vectorization_mode="async", **gym_env_kwargs
    )
    input_dim = envs.observation_space.shape[-1]
    if isinstance(envs.action_space, gym.spaces.MultiDiscrete):
        output_dim = envs.action_space[0].n
    elif isinstance(envs.action_space, gym.spaces.Discrete):
        output_dim = envs.action_space.n
    else:
        output_dim = envs.action_space.shape[-1]

    # setup initial population
    networks: list[Topology] = [
        sample_topology(
            n_init_nodes,
            input_dim,
            output_dim,
            sampling_parameters,
            device=device,
            rng=rng,
        )
        for _ in range(n_networks)
    ]

    logging.info(f"Set up initial population of {n_networks} networks")

    for step in range(n_steps):
        # generate torch.module from networks
        # *only* used in evaluation step
        modules = [ProgressiveRNN(network, device=device) for network in networks]
        logging.info(f"Running step {step} of {n_steps}")
        # run evaluation step
        loss = evaluate(modules, envs, seed + step)
        logging.info(f"Evaluation step {step} complete")

        # run network optimization step
        networks = optim_step(
            networks,
            loss,
            sampling_parameters,
            input_dim,
            output_dim,
            rng=rng,
            **extra_params,
        )

        logging.info(f"Step {step} complete")
