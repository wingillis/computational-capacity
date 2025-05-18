"""Module for running the various optimization algorithms.
Currently supported:
- Evolutionary algorithms (genetic evolution)
- Random sampling
"""

import torch
import random
import logging
import numpy as np
import gymnasium as gym
from typing import Literal
from tqdm.auto import tqdm
from itertools import count
from comp_capacity.save import SavingBuffer
from toolz import pluck, dissoc, keymap, merge
from comp_capacity.repr.network import Topology, ProgressiveRNN
from comp_capacity.optim.genetic_evolution import evolution_step
from comp_capacity.optim.random_sample import (
    random_step,
    sample_topology,
    SamplingParameters,
)

logger = logging.getLogger(__name__)


def setup_data(data: dict) -> list[dict]:
    """Transforms the data dict from the eval step into a list of dicts, where each item is the
    outcome of a single batch."""
    out = []
    keys = list(data.keys())
    for batch_num in range(len(data[keys[0]])):
        d = {key: data[key][batch_num] for key in keys}
        d["batch_num"] = batch_num
        out.append(d)
    return out


def single_step_eval(
    module: ProgressiveRNN,
    envs: gym.Env,
    seed: int,
    is_categorical: bool,
) -> float:
    device = module.device
    obs, info = envs.reset(seed=seed)
    obs = torch.from_numpy(obs).float().to(device)
    if obs.ndim == 1:
        obs = obs.unsqueeze(0)
    state = None
    module.eval()
    with torch.no_grad():
        action, state = module(obs, state=state)
        if is_categorical:
            action = torch.distributions.Categorical(logits=action).sample()
        action = action.squeeze().cpu().numpy()
        obs, reward, terminated, truncated, info = envs.step(action)
    return reward


def multi_step_eval(
    module: ProgressiveRNN,
    envs: gym.Env,
    seed: int,
    is_categorical: bool,
) -> float:
    device = module.device
    batch_size = envs.num_envs

    done_vec = np.zeros(batch_size, dtype=bool)
    reward_vec = np.zeros(batch_size, dtype=float)
    total_steps = np.zeros(batch_size, dtype=int)
    naninf = np.zeros(batch_size, dtype=bool)

    obs, info = envs.reset(seed=seed)

    state = None
    states = []
    for step in count(1):
        obs_tensor = torch.from_numpy(obs).to(device, dtype=torch.float32)
        out, state = module(obs_tensor, state)
        states.append(state.detach().clone().cpu().numpy())
        # check if things got out of hand
        if torch.any(torch.isnan(out)) or torch.any(torch.isnan(state)):
            # consider it terminated
            naninf |= np.logical_or(
                (~np.isfinite(out.detach().cpu())).any(-1),
                (~np.isfinite(state.detach().cpu())).any(-1),
            )
            done_vec |= naninf

            mask = ~torch.isfinite(out)
            out[mask] = 0.0

        if is_categorical:
            action = torch.distributions.Categorical(logits=out).sample().cpu().numpy()
        else:
            action = out.cpu().numpy()

        obs, reward, terminated, truncated, info = envs.step(action)

        done_vec |= np.logical_or(terminated, truncated)
        reward_vec += reward * (1 - done_vec)
        total_steps += 1 * (1 - done_vec)

        if done_vec.all():
            break

    return reward_vec, total_steps, naninf, done_vec


def evaluate(
    modules: list[ProgressiveRNN], envs: gym.Env, seed: int, multi_step: bool
) -> list[dict]:
    """
    Returns a tensor of shape (n_modules, batch_size) where each element is the
    reward for the corresponding module and environment.
    """
    device = modules[0].device
    is_categorical = isinstance(
        envs.action_space, (gym.spaces.MultiDiscrete, gym.spaces.Discrete)
    )
    if is_categorical:
        logger.info(f"Action space is categorical: {envs.action_space}")

    if multi_step:
        logger.info("Evaluating in multistep mode")
    else:
        logger.info("Evaluating in single step mode")

    logger.info(f"Evaluating {len(modules)} modules")
    out = []
    for module in modules:
        if multi_step:
            reward, total_steps, naninf, done_vec = multi_step_eval(
                module, envs, seed, is_categorical
            )
            out.append(
                {
                    "reward": reward,
                    "total_steps": total_steps,
                    "naninf": naninf,
                    "done_vec": done_vec,
                }
            )
        else:
            reward = single_step_eval(module, envs, seed, is_categorical)
            out.append(
                {
                    "reward": reward,
                }
            )
    return out


# a generic function to run any of the optimization algorithms
def run(
    n_networks: int,
    batch_size: int,
    gym_env_name: str,
    gym_env_kwargs: dict,
    gym_env_is_multi_step: bool,
    n_steps: int,
    n_init_nodes: int,
    algorithm: Literal["genetic_evolution", "random_sampling"],
    sampling_parameters: SamplingParameters,
    extra_params: dict,
    data_save_folder: str,
    seed: int | None = None,
    weight_init: Literal["kaiming", "binary", "normal", "uniform"] = "kaiming",
    save_buffer_size: int = 2000,
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

    logger.info(f"Using device: {device}")

    rng = random.Random(seed)
    torch.manual_seed(seed)

    logger.info(
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
        # remove evolution_parameters key if it exists in this function
        extra_params = dissoc(extra_params, "evolution_parameters")
    else:
        raise ValueError(f"Algorithm {algorithm} not supported")

    logger.info(f"Running {algorithm} algorithm")

    logger.info(f"Setting up environment {gym_env_name} with batch size {batch_size}")
    logger.info(f"env kwargs: {gym_env_kwargs}")

    envs = gym.make_vec(
        gym_env_name, num_envs=batch_size, vectorization_mode="async", **gym_env_kwargs
    )

    input_dim = envs.observation_space.shape[-1]
    if isinstance(envs.action_space, gym.spaces.MultiDiscrete):
        output_dim = int(envs.action_space[0].n)
    elif isinstance(envs.action_space, gym.spaces.Discrete):
        output_dim = int(envs.action_space.n)
    else:
        output_dim = int(envs.action_space.shape[-1])

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

    logger.info(f"Set up initial population of {n_networks} networks")

    saver = SavingBuffer(
        buffer_size=save_buffer_size,
        folder_path=data_save_folder,
        file_base_name=f"{algorithm}_{gym_env_name}",
    )

    pbar = tqdm(range(n_steps))

    for step in pbar:
        # generate torch.module from networks
        # *only* used in evaluation step - some parameters are saved too
        modules = [
            ProgressiveRNN(network, device=device, weight_init=weight_init)
            for network in networks
        ]
        logger.info(f"Running step {step} of {n_steps}")
        # run evaluation step
        eval_out = evaluate(
            modules, envs, seed + step, multi_step=gym_env_is_multi_step
        )
        loss = torch.from_numpy(np.array(list(pluck("reward", eval_out)))).to(
            dtype=torch.float32, device=device
        )
        logger.info(f"Evaluation step {step} complete")

        # data saving step
        for model_num, eval_dict in enumerate(eval_out):
            eval_dict = setup_data(eval_dict)
            for item in eval_dict:
                save_item = {
                    "model_num": model_num,
                    "step": step,
                    "model_hash": networks[model_num].hash,
                    "n_nodes": networks[model_num].n_nodes,
                    "output_dim": output_dim,
                    "input_dim": input_dim,
                    "seed": seed,
                    "env_name": gym_env_name,
                    "nn_weight_init": weight_init,
                    "env_kwargs": str(gym_env_kwargs),
                    "opt_algorithm": algorithm,
                    "parameter_count": sum(
                        p.numel() for p in modules[model_num].parameters()
                    ),
                    "n_nonlinearities": networks[model_num].inner.nonlinearity.shape[1],
                }
                sampling_param_dict = sampling_parameters.model_dump()
                sampling_param_dict = keymap(
                    lambda k: "sampling_param." + k, sampling_param_dict
                )
                save_item = merge(save_item, sampling_param_dict)

                if algorithm == "genetic_evolution":
                    evolution_param_dict = extra_params[
                        "evolution_parameters"
                    ].model_dump()
                    evolution_param_dict = keymap(
                        lambda k: "evolution_param." + k, evolution_param_dict
                    )
                    save_item = merge(save_item, evolution_param_dict)

                # added compressed topology - original topology can be recovered from it
                saver.add(merge(item, save_item, networks[model_num].compress()))
        logger.info("Saved data to buffer")

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

        logger.info(f"Step {step} complete")
        logger.info(f"Best score this epoch: {loss.max()}")

        _n_nodes = networks[0].n_nodes

        pbar.set_description(f"Best score this epoch={loss.max():.1e}; # nodes={_n_nodes}")
