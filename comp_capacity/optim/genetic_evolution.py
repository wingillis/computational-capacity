import torch
import random
import logging
import numpy as np
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field
from comp_capacity.repr.network import Topology, NONLINEARITY_MAP
from comp_capacity.optim.random_sample import (
    sample_topology,
    add_connectivity,
    sample_nonlinearity_matrix,
    construct_sampling_mask,
)


class MutationType(Enum):
    NODE_MANIPULATION = "node_manipulation"
    EDGE_MANIPULATION = "edge_manipulation"
    DUPLICATION = "duplication"
    INVERSION = "inversion"


# definition of each mutation type
# - node_manipulation: add or remove a node
# - edge_manipulation: add, remove, or move an edge
# - duplication: duplicate an entire region of the adjacency matrix
# - inversion: reverse order of an entire region of the adjacency matrix


class ManipulationType(Enum):
    ADD = "add"
    REMOVE = "remove"
    MOVE = "move"


class SamplingParameters(BaseModel):
    connection_prob: float
    recurrent: bool


class EvolutionParameters(BaseModel):
    survival_rate: float = Field(ge=0, le=1)  # proportion of population to survive
    mutation_rate: float = Field(ge=0, le=1)  # proportion of population to mutate

    # compute reproductive rate after pydantic initialization
    @property
    def reproductive_rate(self) -> float:
        # keep popultation same size
        return 1 - self.survival_rate


def add_node(
    topology: Topology, sampling_parameters: SamplingParameters, rng: random.Random
) -> Topology:
    n_nodes = topology.adjacency.shape[0]
    device = topology.adjacency.device
    where = rng.randint(0, n_nodes - 1)

    logging.info(f"Adding node at {where}")

    new_adjacency = topology.adjacency.cpu().numpy()
    new_adjacency = np.insert(new_adjacency, where, values=0, axis=0)
    new_adjacency = np.insert(new_adjacency, where, values=0, axis=1)

    new_adjacency = torch.from_numpy(new_adjacency).to(device=device)

    sampling_probs = construct_sampling_mask(
        n_nodes + 1,
        sampling_parameters.connection_prob,
        sampling_parameters.recurrent,
        device,
    )

    sample = torch.bernoulli(sampling_probs[where, :])

    new_adjacency[where, :] = sample

    sample = torch.bernoulli(sampling_probs[:, where])
    new_adjacency[:, where] = sample

    new_adjacency = add_connectivity(new_adjacency, sampling_probs, index=0)
    new_adjacency = add_connectivity(new_adjacency.T, sampling_probs.T, index=n_nodes).T

    # insert nonlinearity for new node
    new_nonlinearity_row = sample_nonlinearity_matrix(
        n_nodes=1, n_nonlinearities=len(NONLINEARITY_MAP), device=device
    )
    new_nonlinearity = torch.cat(
        [
            topology.nonlinearity[:where],
            new_nonlinearity_row,
            topology.nonlinearity[where:],
        ]
    )

    new_topology = Topology(
        adjacency=new_adjacency.to(device=device),
        nonlinearity=new_nonlinearity.to(device=device),
        module=torch.ones((n_nodes + 1, 1), device=device),
    )

    return new_topology


def remove_node(
    topology: Topology, sampling_parameters: SamplingParameters, rng: random.Random
) -> Topology:
    device = topology.adjacency.device
    n_nodes = topology.adjacency.shape[0]

    # Ensure we have more than 2 nodes (input and output)
    # and choose a node to remove (avoid removing input or output nodes)
    if n_nodes <= 2:
        # Can't remove any more nodes, return original topology
        logging.warning("Can't remove any more nodes, returning original topology")
        return topology

    where = rng.randint(0, n_nodes - 1)

    logging.info(f"Removing node {where}")

    # Create new adjacency matrix by removing the selected node
    new_adjacency = topology.adjacency.cpu().numpy()
    new_adjacency = np.delete(new_adjacency, where, axis=0)
    new_adjacency = np.delete(new_adjacency, where, axis=1)

    # Convert back to tensor and ensure connectivity
    new_adjacency = torch.from_numpy(new_adjacency).to(device=device)

    # Create mask for connectivity checks
    sampling_probs = torch.full(
        (n_nodes - 1, n_nodes - 1),
        fill_value=sampling_parameters.connection_prob,
    )
    if not sampling_parameters.recurrent:
        sampling_probs = torch.triu(sampling_probs, diagonal=1)

    # Ensure connectivity from input to all nodes and from all nodes to output
    new_adjacency = add_connectivity(new_adjacency, sampling_probs, index=0)
    new_adjacency = add_connectivity(
        new_adjacency.T, sampling_probs.T, index=n_nodes - 2
    ).T

    # Remove nonlinearity for the removed node
    new_nonlinearity = torch.cat(
        [topology.nonlinearity[:where], topology.nonlinearity[where + 1 :]]
    )

    # Create new topology
    new_topology = Topology(
        adjacency=new_adjacency.to(device=device),
        nonlinearity=new_nonlinearity.to(device=device),
        module=torch.ones((n_nodes - 1, 1), device=device),
    )

    return new_topology


def add_edge(
    topology: Topology, sampling_parameters: SamplingParameters, rng: random.Random
) -> Topology:
    n_nodes = topology.adjacency.shape[0]
    device = topology.adjacency.device

    sampling_probs = construct_sampling_mask(
        n_nodes,
        sampling_parameters.connection_prob,
        sampling_parameters.recurrent,
        device,
    )
    # set probs to 0 for all edges that already exist
    sampling_probs = torch.where(topology.adjacency, 0, sampling_probs)

    # get indices for potential new edges
    potential_edges = torch.nonzero(sampling_probs)

    # sample a random edge
    add_edge = rng.choice(potential_edges)

    logging.info(f"Adding edge between {add_edge[0]} and {add_edge[1]}")

    # create new adjacency matrix
    new_adjacency = topology.adjacency.clone()
    new_adjacency[add_edge] = 1

    return Topology(
        adjacency=new_adjacency,
        nonlinearity=topology.nonlinearity,
        module=topology.module,
    )


def remove_edge(topology: Topology, rng: random.Random) -> Topology:
    # get indices for existing edges
    existing_edges = torch.nonzero(topology.adjacency)

    # sample a random edge
    remove_edge = rng.choice(existing_edges)

    # create new adjacency matrix
    new_adjacency = topology.adjacency.clone()
    new_adjacency[remove_edge] = 0

    return Topology(
        adjacency=new_adjacency,
        nonlinearity=topology.nonlinearity,
        module=topology.module,
    )


def move_edge(
    topology: Topology, sampling_parameters: SamplingParameters, rng: random.Random
) -> Topology:
    n_nodes = topology.adjacency.shape[0]
    device = topology.adjacency.device

    sampling_probs = construct_sampling_mask(
        n_nodes,
        sampling_parameters.connection_prob,
        sampling_parameters.recurrent,
        device,
    )
    # set probs to 0 for all edges that already exist
    sampling_probs = torch.where(topology.adjacency, 0, sampling_probs)

    # get indices for existing edges
    existing_edges = torch.nonzero(topology.adjacency)

    # sample a random edge - this is the edge to move
    remove_edge = rng.choice(existing_edges)

    # sample a random edge - this is where the edge will be moved to
    potential_edges = torch.nonzero(sampling_probs)
    add_edge = rng.choice(potential_edges)

    # create new adjacency matrix
    new_adjacency = topology.adjacency.clone()
    new_adjacency[remove_edge] = 0
    new_adjacency[add_edge] = 1

    return Topology(
        adjacency=new_adjacency,
        nonlinearity=topology.nonlinearity,
        module=topology.module,
    )


def manipulate_topology(
    topology: Topology,
    manipulation_type: ManipulationType,
    sampling_parameters: SamplingParameters,
    to_manipulate: Literal["node", "edge"],
    rng: random.Random,
) -> Topology:
    if manipulation_type == ManipulationType.ADD:
        if to_manipulate == "node":
            new_topology = add_node(topology, sampling_parameters, rng)
        elif to_manipulate == "edge":
            new_topology = add_edge(topology, sampling_parameters, rng)

    elif manipulation_type == ManipulationType.REMOVE:
        if to_manipulate == "node":
            new_topology = remove_node(topology, sampling_parameters, rng)
        elif to_manipulate == "edge":
            new_topology = remove_edge(topology, rng)

    elif manipulation_type == ManipulationType.MOVE:
        if to_manipulate == "node":
            raise ValueError("Node movement not implemented because it's irrelevant")
        elif to_manipulate == "edge":
            new_topology = move_edge(topology, sampling_parameters, rng)

    return new_topology


def duplicate_block(
    topology: Topology,
    rng: random.Random,
) -> Topology:
    # Get the adjacency matrix
    adjacency = topology.adjacency
    device = adjacency.device
    n_nodes = adjacency.shape[0]

    if n_nodes < 3:
        logging.warning("Can't duplicate block because there are less than 3 nodes")
        return topology

    # Determine block size to duplicate (between 1 and n_nodes/3)
    block_size = rng.randint(2, n_nodes - 1)

    # Select a random starting row for the block to duplicate
    start_idx = rng.randint(0, n_nodes - block_size)
    end_idx = start_idx + block_size

    logging.info(
        f"Duplicating block of size {block_size} at indices {start_idx} to {start_idx + block_size}"
    )

    # Create a new adjacency matrix with additional rows/columns for the duplicated block
    new_size = n_nodes + block_size

    new_adjacency = torch.zeros((new_size, new_size), device=device)
    new_adjacency[:n_nodes, :n_nodes] = adjacency

    block_slice = slice(start_idx, end_idx)
    dup_block_slice = slice(n_nodes, new_size)

    # duplicate connections from original block to original nodes
    new_adjacency[dup_block_slice, :n_nodes] = adjacency[block_slice, :]
    # duplicate connections to original block from original nodes
    new_adjacency[:n_nodes, dup_block_slice] = adjacency[:, block_slice]

    # zero out between-block connections
    new_adjacency[dup_block_slice, block_slice] = 0
    new_adjacency[block_slice, dup_block_slice] = 0

    # duplicate within-block connections
    new_adjacency[dup_block_slice, dup_block_slice] = adjacency[
        block_slice, block_slice
    ]

    # re-sort the adjacency matrix so duplicated nodes are next to original block
    indices = torch.arange(new_size)
    indices = torch.cat(
        [indices[:end_idx], indices[-block_size:], indices[end_idx:n_nodes]]
    )
    new_adjacency = new_adjacency[indices][:, indices]

    new_nonlinearity = torch.cat(
        [
            topology.nonlinearity[:end_idx],
            topology.nonlinearity[block_slice],
            topology.nonlinearity[end_idx:],
        ]
    )

    return Topology(
        adjacency=new_adjacency,
        nonlinearity=new_nonlinearity,
        module=torch.ones((new_size, 1), device=device),
    )


def invert_block(
    topology: Topology,
    sampling_parameters: SamplingParameters,
    rng: random.Random,
) -> Topology:
    """Reverses the direction of the connections within a block of the adjacency matrix"""
    adjacency = topology.adjacency
    n_nodes = adjacency.shape[0]

    if n_nodes < 3:
        logging.warning("Can't invert block because there are less than 3 nodes")
        return topology

    if not sampling_parameters.recurrent:
        logging.warning(
            "Can't invert block, because recurrent connections are not allowed"
        )
        return topology

    block_size = rng.randint(2, n_nodes - 1)

    # get indices for block to invert
    start_idx = rng.randint(0, n_nodes - block_size)
    end_idx = start_idx + block_size

    logging.info(
        f"Inverting block of size {block_size} at indices {start_idx} to {end_idx}"
    )

    # create new adjacency matrix
    new_adjacency = adjacency.clone()

    block_slice = slice(start_idx, end_idx)

    # invert the block
    new_adjacency[block_slice, block_slice] = new_adjacency[
        block_slice, block_slice
    ].transpose(0, 1)

    new_nonlinearity = topology.nonlinearity.clone()
    new_nonlinearity[block_slice] = new_nonlinearity[block_slice].flip(0)

    return Topology(
        adjacency=new_adjacency,
        nonlinearity=new_nonlinearity,
        module=topology.module,
    )


def mutate_topology(
    topology: Topology, sampling_parameters: SamplingParameters, rng: random.Random
) -> tuple[Topology, MutationType]:
    # select a mutation type
    mutation_type = rng.choice(list(MutationType))

    match mutation_type:
        case MutationType.NODE_MANIPULATION:
            manipulation_type = rng.choice(list(ManipulationType)[:-1])
            mutated_topology = manipulate_topology(
                topology, manipulation_type, sampling_parameters, "node", rng
            )
        case MutationType.EDGE_MANIPULATION:
            manipulation_type = rng.choice(list(ManipulationType))
            mutated_topology = manipulate_topology(
                topology, manipulation_type, sampling_parameters, "edge", rng
            )
        case MutationType.DUPLICATION:
            mutated_topology = duplicate_block(topology, rng)
        case MutationType.INVERSION:
            mutated_topology = invert_block(topology, sampling_parameters, rng)
        case _:
            raise ValueError(f"Invalid mutation type: {mutation_type}")

    return mutated_topology, mutation_type


def crossover(topology1: Topology, topology2: Topology, rng: random.Random) -> Topology:
    """
    Receives two topologies and performs a crossover.
    The crossover itself is done by selecting a random block from one topology and
    duplicating it in the other topology. Returns the crossed-over topology.
    """
    # get adjacency matrices
    adjacency1 = topology1.adjacency
    adjacency2 = topology2.adjacency

    # get nonlinearities
    nonlinearity1 = topology1.nonlinearity
    nonlinearity2 = topology2.nonlinearity

    # pad adjacency matrices so that they have the same size
    size = max(adjacency1.shape[0], adjacency2.shape[0])
    min_size = min(adjacency1.shape[0], adjacency2.shape[0])

    # select a random block from one topology
    start_idx = rng.randint(0, size - 1)
    end_idx = rng.randint(start_idx + 1, size)

    def _pad(mtx: torch.Tensor, square: bool = True) -> torch.Tensor:
        if square:
            new_mtx = torch.zeros((size, size), device=mtx.device)
        else:
            new_mtx = torch.zeros((size, mtx.shape[1]), device=mtx.device)
        new_mtx[: mtx.shape[0], : mtx.shape[1]] = mtx
        return new_mtx

    if end_idx > min_size:
        adjacency1 = _pad(adjacency1)
        adjacency2 = _pad(adjacency2)

        nonlinearity1 = _pad(nonlinearity1, square=False)
        nonlinearity2 = _pad(nonlinearity2, square=False)

    # duplicate the block in the other topology
    adjacency2[start_idx:end_idx, start_idx:end_idx] = adjacency1[
        start_idx:end_idx, start_idx:end_idx
    ]

    nonlinearity2[start_idx:end_idx] = nonlinearity1[start_idx:end_idx]

    # create new topologies
    new_topology2 = Topology(
        adjacency=adjacency2,
        nonlinearity=nonlinearity2,
        module=topology2.module,
    )

    return new_topology2


def crossover_topologies(
    topologies: list[Topology], rng: random.Random
) -> list[Topology]:
    """
    Receives a population of topologies, pairs them up, and performs a crossover.
    The crossover itself is done by selecting a random block from one topology and
    duplicating it in the other topology.
    """
    if len(topologies) % 2 != 0:
        logging.warning("Odd number of topologies, discarding last one")

    pairs = list(zip(topologies[::2], topologies[1::2]))

    new_topologies = []
    # perform crossover
    for pair in pairs:
        new_pair = crossover(pair[0], pair[1], rng)
        new_topologies.extend(new_pair)

    return new_topologies


def survival_selection(
    topologies: list[Topology], fitness: list[float], parameters: EvolutionParameters
) -> list[Topology]:
    # fitness proportional selection approach
    choices = torch.distributions.Categorical(probs=fitness).sample(
        int(len(topologies) * parameters.survival_rate)
    )
    return [topologies[i] for i in choices]


def reproduce(
    topologies: list[Topology], parameters: EvolutionParameters, rng: random.Random
) -> list[Topology]:
    n_babies = int(len(topologies) * parameters.reproductive_rate)

    # shuffle list, then pair topologies up
    rng.shuffle(topologies)

    # select parents
    parents = topologies[: n_babies * 2]

    babies = crossover_topologies(parents, rng)

    return topologies + babies


def evolution_step(
    topologies: list[Topology],
    fitness: list[float],
    evolution_parameters: EvolutionParameters,
    sampling_parameters: SamplingParameters,
    rng: random.Random,
) -> list[Topology]:
    # survival selection
    survived_topologies = survival_selection(topologies, fitness, evolution_parameters)

    # reproduction
    reproduced_topologies = reproduce(survived_topologies, evolution_parameters, rng)

    # mutation
    mutated_topologies = []
    for topology in reproduced_topologies:
        if rng.random() < evolution_parameters.mutation_rate:
            mutated, mutation_type = mutate_topology(topology, sampling_parameters, rng)
            logging.info(f"Mutated topology with mutation type {mutation_type}")
            mutated_topologies.append(mutated)
        else:
            mutated_topologies.append(topology)

    return mutated_topologies
