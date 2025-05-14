import torch
import random
import numpy as np
from pydantic import BaseModel, Field
from torch.distributions import Multinomial
from scipy.sparse.csgraph import shortest_path
from comp_capacity.repr.network import Topology, NONLINEARITY_MAP

## Conventions for matrix
# 1. rows are 'from' nodes
# 2. columns are 'to' nodes
# 3. the first row is the input layer
# 4. the last column is the output layer

## to start
# 1. mask out lower triangular and diagonal elements (make DAG)
# 2. module matrix is all 'nodes', i.e. [1, 0, 0, 0]
# 3. nonlinearity matrix is all 'linear / none' and 'relu', 'tanh', i.e. [0, 1, 0, 0]

class SamplingParameters(BaseModel):
    connection_prob: float
    recurrent: bool
    # flag to allow modification of input/output projections
    # for now, I keep it off because it can make the search space too large
    modify_projections: bool = False
    # this parameter is for random sampling only
    increase_node_prob: float = Field(gt=0, lt=1, default=0.1)


def construct_sampling_mask(n_nodes: int, sampling_parameters: SamplingParameters, device: torch.device | str | None = None):
    probs = torch.full((n_nodes, n_nodes), fill_value=sampling_parameters.connection_prob, device=device)
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


def add_connectivity(adj_matrix, orig_mask, index=0):
    """
    Ensure every node is reachable from the `index` node by repeatedly
    adding one edge from the reachable set to the first unreachable node.
    """
    adj_matrix = adj_matrix.clone()
    orig_mask = orig_mask.cpu().numpy() > 0
    while True:
        # 1) compute reachability
        dist = shortest_path(adj_matrix.cpu().numpy(), directed=True, indices=index)

        unreachable = np.where(np.isinf(dist))[0]
        if index == 0:
            unreachable = unreachable[unreachable != len(adj_matrix) - 1]
        elif index == len(adj_matrix) - 1:
            unreachable = unreachable[unreachable != 0]

        # if all nodes are reachable, finish
        if len(unreachable) == 0:
            break

        node = unreachable[0]
        mask = np.isfinite(dist) & orig_mask[:, node]

        # guard against empty mask
        if not mask.any():
            raise RuntimeError(f"No reachable origin to connect node {node!r}")

        # sample exactly one new incoming edge
        probs = torch.tensor(mask, dtype=torch.float, device=adj_matrix.device)
        sample_connection = (
            Multinomial(total_count=1, probs=probs).sample().to(dtype=torch.bool)
        )

        # combine with existing connections
        adj_matrix[:, node] = sample_connection | adj_matrix[:, node]

    return adj_matrix


def sample_connectivity(
    n_nodes: int,
    sampling_parameters: SamplingParameters,
    seed: int | None = None,
    device: str | None = None,
):
    """
    Generate a random connectivity matrix of shape (n_nodes + 2, n_nodes + 2).
    The first and last nodes are the input and output nodes, respectively.
    The connectivity is generated using a Bernoulli distribution with probability p.
    The function ensures that each node is reachable from the input node and
    the output node is reachable from all nodes (except the input).

    Args:
        n_nodes (int): Number of nodes (not including input and output nodes).
        p (float): Probability of connection between nodes.
        recurrent (bool): If True, allows recurrent connections.
        seed (int, optional): Random seed for reproducibility.
        device (str, optional): Device to create the tensor on. Defaults to None.
    """
    if seed is not None:
        torch.manual_seed(seed)

    probs = construct_sampling_mask(n_nodes, sampling_parameters, device)

    # fill boolean values with probabilities
    sample = torch.bernoulli(probs).to(dtype=torch.bool)

    # make sure each node is reachable from the input node
    sample = add_connectivity(sample, probs, index=0)

    # make sure output node is reachable from all nodes
    sample = add_connectivity(sample.T, probs.T, index=n_nodes - 1).T

    return sample


def sample_topology(
    n_nodes: int,
    sampling_parameters: SamplingParameters,
    device: torch.device | None = None,
) -> Topology:
    adjacency = sample_connectivity(
        n_nodes=n_nodes,
        sampling_parameters=sampling_parameters,
        device=device,
    )
    nonlinearity = sample_nonlinearity_matrix(
        n_nodes=n_nodes,
        n_nonlinearities=len(NONLINEARITY_MAP),
        device=device,
    )

    module = torch.ones((n_nodes, 1), dtype=torch.bool, device=device)
    matrices = Topology(
        adjacency=adjacency,
        module=module,
        nonlinearity=nonlinearity,
    )
    return matrices


def random_step(networks: list[Topology], score: list[float], sampling_parameters: SamplingParameters, rng: random.Random) -> list[Topology]:

    device = networks[0].device

    n_networks = len(networks)

    max_nodes = max(x.n_nodes for x in networks)

    node_list = np.array([rng.random() < sampling_parameters.increase_node_prob for _ in range(n_networks)])
    node_list = np.cumsum(node_list) + max_nodes

    return [sample_topology(n_nodes, sampling_parameters, device=device) for n_nodes in node_list]