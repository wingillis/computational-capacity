import torch
import random
import logging
import numpy as np
from typing import Literal
from pydantic import BaseModel, Field
from torch.distributions import Multinomial
from scipy.sparse.csgraph import shortest_path
from comp_capacity.repr.network import (
    Topology,
    NONLINEARITY_MAP,
    InnerTopology,
    Projection,
)

logger = logging.getLogger(__name__)

class SamplingParameters(BaseModel):
    connection_prob: float
    recurrent: bool
    # flag to allow modification of input/output projections
    # for now, I keep it on because otherwise the search space can be too large
    use_fully_connected_projections: bool = True
    # this parameter is for random sampling only
    increase_node_prob: float = Field(gt=0, lt=1, default=0.1)


def construct_sampling_mask(
    n_nodes: int,
    sampling_parameters: SamplingParameters,
    device: torch.device | str | None = None,
):
    probs = torch.full(
        (n_nodes, n_nodes),
        fill_value=sampling_parameters.connection_prob,
        device=device,
    )
    if not sampling_parameters.recurrent:
        probs = torch.triu(probs, diagonal=1)
    return probs


def sample_nonlinearity_matrix(
    n_nodes: int = 2,
    n_nonlinearities: int = 2,
    weighting: torch.Tensor | None = None,
    seed: int | None = None,
    device: str | None = None,
) -> torch.Tensor:
    """
    Generate a random nonlinearity matrix of shape (n_nodes, n_nonlinearities).

    Args:
        n_nodes (int): Number of nodes.
        n_nonlinearities (int): Number of nonlinearities.
        weighting (torch.Tensor, optional): Weights for each nonlinearity.
            If None, uniform weights are used.
        seed (int, optional): Random seed for reproducibility.
        device (str, optional): Device to create the tensor on. Defaults to None.
    Returns:
        torch.Tensor: One-hot encoded nonlinearity matrix with shape (n_nodes, n_nonlinearities).
    """
    if seed is not None:
        torch.manual_seed(seed)

    if weighting is None:
        weighting = torch.ones(n_nonlinearities, device=device)

    sampler = Multinomial(probs=weighting)

    # sample single value for each node, creating a one-hot encoding
    return sampler.sample((n_nodes,)).to(device=device, dtype=torch.bool)


def ensure_projection_connectivity(
    topology: Topology,
    parameters: SamplingParameters,
    projection_type: Literal["input", "output"],
) -> Topology:
    """
    Ensure every node is reachable from the `index` node by repeatedly
    adding one edge from the reachable set to the first unreachable node.

    Args:
        topology (Topology): The topology to check connectivity.
        parameters (SamplingParameters): The sampling parameters.
        index (int): Indicates which node to ensure connectivity from.
    Returns:
        Topology: The updated topology.
    """
    device = topology.inner.adjacency.device

    # step 1: collapse input projection onto n_node dimension
    input_row = topology.input.adjacency.any(dim=1)
    # step 2: collapse output projection onto n_node dimension
    output_row = topology.output.adjacency.any(dim=1)

    # add input and output rows to adjacency matrix
    new_adjacency = torch.zeros(
        topology.inner.adjacency.shape[0] + 2,
        topology.inner.adjacency.shape[0] + 2,
        dtype=torch.bool,
        device=device,
    )
    n_nodes = new_adjacency.shape[0]

    new_adjacency[0, 1:-1] = input_row
    new_adjacency[1:-1, -1] = output_row
    new_adjacency[1:-1, 1:-1] = topology.inner.adjacency

    assert new_adjacency[:, 0].sum() == 0
    assert new_adjacency[-1, :].sum() == 0

    probs = construct_sampling_mask(
        n_nodes=n_nodes,
        sampling_parameters=parameters,
        device=device,
    )

    if projection_type == "output":
        node_index = n_nodes - 1
        new_adjacency = new_adjacency.T
        probs = probs.T
        logger.info("Running connectivity check for output projection")
    else:
        node_index = 0
        logger.info("Running connectivity check for input projection")

    probs_mask = probs.cpu().numpy() > 0
    # because this adjacency matrix contains input and output nodes, we need to
    # mask out connections between them
    probs_mask[0, -1] = 0
    probs_mask[:, 0] = 0
    probs_mask[-1, :] = 0

    while True:
        # 1) compute reachability
        dist = shortest_path(
            new_adjacency.cpu().numpy(), directed=True, indices=node_index
        )

        # 2) find unreachable nodes
        unreachable = np.where(np.isinf(dist))[0]
        if node_index == 0:
            # ignore input node - we don't want to connect the input to the output
            unreachable = unreachable[unreachable != n_nodes - 1]
        elif node_index == n_nodes - 1:
            # ignore output node - we don't want to connect the input to the output
            unreachable = unreachable[unreachable != 0]

        # if all nodes are reachable, finish
        if len(unreachable) == 0:
            break
        else:
            logger.info(f"Unreachable nodes: {unreachable}")

        # 3) sample new connection with node closest to input node
        #    - only consider nodes that are reachable
        node = unreachable[0]
        mask = np.isfinite(dist) & probs_mask[:, node]

        # guard against empty mask
        if not mask.any():
            raise RuntimeError(f"No reachable origin to connect node {node!r}")

        # sample exactly one new incoming edge
        probs = torch.tensor(mask, dtype=torch.float, device=device)
        sample_connection = (
            Multinomial(total_count=1, probs=probs)
            .sample()
            .to(dtype=torch.bool, device=device)
        )

        # combine with existing connections
        new_adjacency[:, node] = sample_connection | new_adjacency[:, node]

    if projection_type == "output":
        new_adjacency = new_adjacency.T

    logger.info(f"Shortest path distance: {dist}")

    inner_adjacency = new_adjacency[1:-1, 1:-1]

    # create new Topology object
    candidate = Topology(
        input=topology.input,
        output=topology.output,
        inner=InnerTopology(
            adjacency=inner_adjacency.to(device=device),
            nonlinearity=topology.inner.nonlinearity,
            module=topology.inner.module,
            weights=topology.inner.weights,
        ),
    )

    return candidate


def sample_adjacency_matrix(
    n_nodes: int,
    sampling_parameters: SamplingParameters,
    seed: int | None = None,
    device: str | None = None,
) -> torch.Tensor:
    """
    Generate a random connectivity matrix of shape (n_nodes, n_nodes).
    The connectivity is generated using a Bernoulli distribution with probability p.
    The function ensures that each node is reachable from the input node and
    the output node is reachable from all nodes (except the input).

    Args:
        n_nodes (int): Number of nodes (not including input and output nodes).
        sampling_parameters (SamplingParameters): Sampling parameter data class.
        seed (int, optional): Random seed for reproducibility.
        device (str, optional): Device to create the tensor on. Defaults to None.
    """
    if seed is not None:
        torch.manual_seed(seed)

    probs = construct_sampling_mask(n_nodes, sampling_parameters, device)

    # fill boolean values with probabilities
    sample = torch.bernoulli(probs).to(dtype=torch.bool)

    return sample


def sample_projection(
    n_nodes: int,
    dim: int,
    sampling_parameters: SamplingParameters,
    device: torch.device | None = None,
    rng: random.Random | None = None,
) -> Projection:

    if rng is not None:
        torch.manual_seed(rng.randint(0, 2**32 - 1))
    else:
        rng = random.Random()

    if sampling_parameters.use_fully_connected_projections:
        # sampling probs are just over nodes, rather than projection dims
        sampling_dim = (n_nodes,)
    else:
        # sampling probs are over everything
        sampling_dim = (n_nodes, dim)

    connectivity = torch.bernoulli(
        torch.full(
            sampling_dim,
            fill_value=sampling_parameters.connection_prob,
            device=device,
        )
    ).to(dtype=torch.bool)

    if sampling_parameters.use_fully_connected_projections:
        # always connect input to first node
        connectivity[0] = True

        # expand connectivity to include all projection dims
        connectivity = connectivity.unsqueeze(1).repeat(1, dim)
    else:
        # make sure each projection dimension has at least one connection
        for i in range(dim):
            if not connectivity[:, i].any():
                j = rng.randint(0, n_nodes - 1)
                connectivity[j, i] = True
        if not connectivity[0].any():
            sample = torch.bernoulli(
                torch.full(
                    (dim,),
                    fill_value=sampling_parameters.connection_prob,
                    device=device,
                )
            ).to(dtype=torch.bool)
            while sample.sum() == 0:
                sample = torch.bernoulli(
                    torch.full(
                        (dim,),
                        fill_value=sampling_parameters.connection_prob,
                        device=device,
                    )
                ).to(dtype=torch.bool)
            connectivity[0, sample] = True

    assert connectivity.shape == (n_nodes, dim)

    return Projection(dim=dim, device=device, adjacency=connectivity)


def sample_topology(
    n_nodes: int,
    input_dim: int,
    output_dim: int,
    sampling_parameters: SamplingParameters,
    device: torch.device | None = None,
    rng: random.Random | None = None,
) -> Topology:
    logger.info(f"Randomly sampling new topology with {n_nodes} nodes, {input_dim} input dim, {output_dim} output dim")

    if rng is None:
        rng = random.Random()

    adjacency = sample_adjacency_matrix(
        n_nodes=n_nodes,
        sampling_parameters=sampling_parameters,
        device=device,
        seed=rng.randint(0, 2**32 - 1),
    )
    nonlinearity = sample_nonlinearity_matrix(
        n_nodes=n_nodes,
        n_nonlinearities=len(NONLINEARITY_MAP),
        device=device,
        seed=rng.randint(0, 2**32 - 1),
    )

    matrices = InnerTopology(
        adjacency=adjacency,
        nonlinearity=nonlinearity,
    )

    input_projection = sample_projection(
        n_nodes=n_nodes,
        dim=input_dim,
        sampling_parameters=sampling_parameters,
        device=device,
        rng=rng,
    )

    output_projection = sample_projection(
        n_nodes=n_nodes,
        dim=output_dim,
        sampling_parameters=sampling_parameters,
        device=device,
        rng=rng,
    )

    candidate = Topology(
        input=input_projection,
        output=output_projection,
        inner=matrices,
    )

    # make sure each node is reachable from the input node
    candidate = ensure_projection_connectivity(candidate, sampling_parameters, "input")

    # make sure output node is reachable from all nodes
    candidate = ensure_projection_connectivity(candidate, sampling_parameters, "output")

    return candidate


def random_step(
    networks: list[Topology],
    score: list[float],
    sampling_parameters: SamplingParameters,
    input_dim: int,
    output_dim: int,
    rng: random.Random,
) -> list[Topology]:

    device = networks[0].adjacencies[1].device

    n_networks = len(networks)

    max_nodes = max(x.n_nodes for x in networks)

    node_list = np.array(
        [
            rng.random() < sampling_parameters.increase_node_prob
            for _ in range(n_networks)
        ]
    )
    node_list = np.cumsum(node_list) + max_nodes

    return [
        sample_topology(
            n_nodes, input_dim, output_dim, sampling_parameters, device=device, rng=rng
        )
        for n_nodes in node_list
    ]
