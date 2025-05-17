import torch
import random
import logging
import numpy as np
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field
from comp_capacity.repr.network import (
    Topology,
    InnerTopology,
    Projection,
    NONLINEARITY_MAP,
)
from comp_capacity.optim.random_sample import (
    sample_topology,
    ensure_projection_connectivity,
    sample_nonlinearity_matrix,
    construct_sampling_mask,
    SamplingParameters,
)

logger = logging.getLogger(__name__)


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


class EvolutionParameters(BaseModel):
    survival_rate: float = Field(ge=0, le=1)  # proportion of population to survive
    mutation_rate: float = Field(ge=0, le=1)  # proportion of population to mutate
    random_sample_rate: float = Field(
        ge=0, le=1, default=0
    )  # probability of adding a new random topology

    def reproduction_amount(self, total_pop: int) -> int:
        # keep population size constant
        return round((1 - self.survival_rate) * total_pop)


def add_node(
    topology: Topology, sampling_parameters: SamplingParameters, rng: random.Random
) -> Topology:
    n_nodes = topology.inner.adjacency.shape[0]
    device = topology.inner.adjacency.device
    where = rng.randint(0, n_nodes - 1)

    logger.info(f"Adding node at {where}")

    new_adjacency = topology.inner.adjacency.cpu().numpy()
    new_adjacency = np.insert(new_adjacency, where, values=0, axis=0)
    new_adjacency = np.insert(new_adjacency, where, values=0, axis=1)

    new_adjacency = torch.from_numpy(new_adjacency).to(device=device)

    sampling_probs = construct_sampling_mask(
        n_nodes + 1,
        sampling_parameters,
        device,
    )

    sample = torch.bernoulli(sampling_probs[where, :])
    new_adjacency[where, :] = sample

    sample = torch.bernoulli(sampling_probs[:, where])
    new_adjacency[:, where] = sample

    # insert nonlinearity for new node
    new_nonlinearity_row = sample_nonlinearity_matrix(
        n_nodes=1, n_nonlinearities=len(NONLINEARITY_MAP), device=device
    )
    new_nonlinearity = torch.cat(
        [
            topology.inner.nonlinearity[:where],
            new_nonlinearity_row,
            topology.inner.nonlinearity[where:],
        ]
    )

    # Update input projection - insert new row
    new_input_adjacency = topology.input.adjacency.cpu().numpy()
    new_input_adjacency = np.insert(
        new_input_adjacency, where, values=0, axis=0
    )
    new_input_adjacency = torch.from_numpy(new_input_adjacency).to(device=device)

    # Sample connections for the new node from input
    if not sampling_parameters.use_fully_connected_projections:
        input_sample = torch.bernoulli(
            torch.full(
                (topology.input.dim,),
                sampling_parameters.connection_prob,
                device=device,
            )
        )
        new_input_adjacency[where, :] = input_sample
    elif rng.random() < sampling_parameters.connection_prob:
        new_input_adjacency[where, :] = 1


    # Update output projection - insert new row
    new_output_adjacency = topology.output.adjacency.cpu().numpy()
    new_output_adjacency = np.insert(
        new_output_adjacency, where, values=0, axis=0
    )
    new_output_adjacency = torch.from_numpy(new_output_adjacency).to(device=device)

    # Sample connections for the new node to output
    if not sampling_parameters.use_fully_connected_projections:
        output_sample = torch.bernoulli(
            torch.full(
                (topology.output.dim,),
                sampling_parameters.connection_prob,
                device=device,
            )
        )
        new_output_adjacency[where, :] = output_sample
    elif rng.random() < sampling_parameters.connection_prob:
        new_output_adjacency[where, :] = 1

    new_inner = InnerTopology(
        adjacency=new_adjacency.to(device=device),
        nonlinearity=new_nonlinearity.to(device=device),
        module=torch.ones((n_nodes + 1, 1), device=device),
    )

    new_input = Projection(
        dim=topology.input.dim,
        device=device,
        adjacency=new_input_adjacency,
        weights=topology.input.weights,
    )

    new_output = Projection(
        dim=topology.output.dim,
        device=device,
        adjacency=new_output_adjacency,
        weights=topology.output.weights,
    )

    candidate = Topology(input=new_input, output=new_output, inner=new_inner)

    candidate = ensure_projection_connectivity(candidate, sampling_parameters, "input")
    candidate = ensure_projection_connectivity(candidate, sampling_parameters, "output")

    return candidate


def remove_node(
    topology: Topology, sampling_parameters: SamplingParameters, rng: random.Random
) -> Topology:
    device = topology.inner.adjacency.device
    n_nodes = topology.inner.adjacency.shape[0]

    # Ensure we have more than 2 nodes (input and output)
    # and choose a node to remove (avoid removing input or output nodes)
    if n_nodes <= 2:
        # Can't remove any more nodes, return original topology
        logger.warning("Can't remove any more nodes, returning original topology")
        return topology

    where = rng.randint(0, n_nodes - 1)

    logger.info(f"Removing node {where}")

    # Create new adjacency matrix by removing the selected node
    new_adjacency = topology.inner.adjacency.cpu().numpy()
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

    # Remove nonlinearity for the removed node
    new_nonlinearity = torch.cat(
        [topology.inner.nonlinearity[:where], topology.inner.nonlinearity[where + 1 :]]
    )

    # Update input projection - remove row
    new_input_adjacency = topology.input.adjacency.cpu().numpy()
    new_input_adjacency = np.delete(new_input_adjacency, where, axis=0)
    new_input_adjacency = torch.from_numpy(new_input_adjacency).to(device=device)

    # Update output projection - remove row
    new_output_adjacency = topology.output.adjacency.cpu().numpy()
    new_output_adjacency = np.delete(new_output_adjacency, where, axis=0)
    new_output_adjacency = torch.from_numpy(new_output_adjacency).to(device=device)

    # Create new inner topology
    new_inner = InnerTopology(
        adjacency=new_adjacency.to(device=device),
        nonlinearity=new_nonlinearity.to(device=device),
        module=torch.ones((n_nodes - 1, 1), device=device),
    )

    new_input = Projection(
        dim=topology.input.dim,
        device=device,
        adjacency=new_input_adjacency,
        weights=topology.input.weights,
    )

    new_output = Projection(
        dim=topology.output.dim,
        device=device,
        adjacency=new_output_adjacency,
        weights=topology.output.weights,
    )

    candidate = Topology(input=new_input, output=new_output, inner=new_inner)

    candidate = ensure_projection_connectivity(candidate, sampling_parameters, "input")
    candidate = ensure_projection_connectivity(candidate, sampling_parameters, "output")

    return candidate


def add_edge(
    topology: Topology, sampling_parameters: SamplingParameters, rng: random.Random
) -> Topology:
    n_nodes = topology.inner.adjacency.shape[0]
    device = topology.inner.adjacency.device

    sampling_probs = construct_sampling_mask(
        n_nodes,
        sampling_parameters,
        device,
    )
    # set probs to 0 for all edges that already exist
    sampling_probs = torch.where(topology.inner.adjacency, 0, sampling_probs)

    # get indices for potential new edges
    potential_edges = torch.nonzero(sampling_probs)

    # sample a random edge
    add_edge = rng.choice(potential_edges)

    logger.info(f"Adding edge between {add_edge[0]} and {add_edge[1]}")

    # create new adjacency matrix
    new_adjacency = topology.inner.adjacency.clone()
    new_adjacency[add_edge] = 1

    new_inner = InnerTopology(
        adjacency=new_adjacency,
        nonlinearity=topology.inner.nonlinearity,
        module=topology.inner.module,
    )

    candidate = Topology(input=topology.input, output=topology.output, inner=new_inner)
    candidate = ensure_projection_connectivity(candidate, sampling_parameters, "input")
    candidate = ensure_projection_connectivity(candidate, sampling_parameters, "output")
    return candidate


def remove_edge(topology: Topology, sampling_parameters: SamplingParameters, rng: random.Random) -> Topology:
    # get indices for existing edges
    existing_edges = torch.nonzero(topology.inner.adjacency)

    # sample a random edge
    remove_edge = rng.choice(existing_edges)
    logger.info(f"Removing edge between {remove_edge[0]} and {remove_edge[1]}")

    # create new adjacency matrix
    new_adjacency = topology.inner.adjacency.clone()
    new_adjacency[remove_edge] = 0

    new_inner = InnerTopology(
        adjacency=new_adjacency,
        nonlinearity=topology.inner.nonlinearity,
        module=topology.inner.module,
    )

    candidate = Topology(input=topology.input, output=topology.output, inner=new_inner)
    candidate = ensure_projection_connectivity(candidate, sampling_parameters, "input")
    candidate = ensure_projection_connectivity(candidate, sampling_parameters, "output")

    return candidate


def move_edge(
    topology: Topology, sampling_parameters: SamplingParameters, rng: random.Random
) -> Topology:
    n_nodes = topology.inner.adjacency.shape[0]
    device = topology.inner.adjacency.device

    sampling_probs = construct_sampling_mask(
        n_nodes,
        sampling_parameters,
        device,
    )
    # set probs to 0 for all edges that already exist
    sampling_probs = torch.where(topology.inner.adjacency, 0, sampling_probs)

    # get indices for existing edges
    existing_edges = torch.nonzero(topology.inner.adjacency)

    # sample a random edge - this is the edge to move
    remove_edge = rng.choice(existing_edges)

    # sample a random edge - this is where the edge will be moved to
    potential_edges = torch.nonzero(sampling_probs)
    add_edge = rng.choice(potential_edges)

    # create new adjacency matrix
    new_adjacency = topology.inner.adjacency.clone()
    new_adjacency[remove_edge] = 0
    new_adjacency[add_edge] = 1

    new_inner = InnerTopology(
        adjacency=new_adjacency,
        nonlinearity=topology.inner.nonlinearity,
        module=topology.inner.module,
    )

    candidate = Topology(input=topology.input, output=topology.output, inner=new_inner)
    candidate = ensure_projection_connectivity(candidate, sampling_parameters, "input")
    candidate = ensure_projection_connectivity(candidate, sampling_parameters, "output")
    return candidate


def manipulate_topology(
    topology: Topology,
    manipulation_type: ManipulationType,
    sampling_parameters: SamplingParameters,
    to_manipulate: Literal["node", "edge"],
    rng: random.Random,
) -> Topology:
    logger.info(f"Manipulating topology with type {manipulation_type}")
    if manipulation_type == ManipulationType.ADD:
        if to_manipulate == "node":
            new_topology = add_node(topology, sampling_parameters, rng)
        elif to_manipulate == "edge":
            new_topology = add_edge(topology, sampling_parameters, rng)

    elif manipulation_type == ManipulationType.REMOVE:
        if to_manipulate == "node":
            new_topology = remove_node(topology, sampling_parameters, rng)
        elif to_manipulate == "edge":
            new_topology = remove_edge(topology, sampling_parameters, rng)

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
    adjacency = topology.inner.adjacency
    device = adjacency.device
    n_nodes = adjacency.shape[0]

    if n_nodes < 3:
        logger.warning("Can't duplicate block because there are less than 3 nodes")
        return topology

    # Determine block size to duplicate (between 1 and n_nodes/3)
    block_size = rng.randint(2, n_nodes - 1)

    # Select a random starting row for the block to duplicate
    start_idx = rng.randint(0, n_nodes - block_size)
    end_idx = start_idx + block_size

    logger.info(
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
            topology.inner.nonlinearity[:end_idx],
            topology.inner.nonlinearity[block_slice],
            topology.inner.nonlinearity[end_idx:],
        ]
    )

    # Update input projection - duplicate corresponding rows
    new_input_adjacency = torch.zeros((new_size, topology.input.dim), device=device)
    new_input_adjacency[:n_nodes, :] = topology.input.adjacency
    new_input_adjacency[dup_block_slice, :] = topology.input.adjacency[block_slice, :]
    # Re-sort to match the adjacency matrix ordering
    new_input_adjacency = new_input_adjacency[indices, :]

    # Update output projection - duplicate corresponding rows
    new_output_adjacency = torch.zeros((new_size, topology.output.dim), device=device)
    new_output_adjacency[:n_nodes, :] = topology.output.adjacency
    new_output_adjacency[dup_block_slice, :] = topology.output.adjacency[block_slice, :]
    # Re-sort to match the adjacency matrix ordering
    new_output_adjacency = new_output_adjacency[indices, :]

    new_inner = InnerTopology(
        adjacency=new_adjacency,
        nonlinearity=new_nonlinearity,
        module=torch.ones((new_size, 1), device=device),
    )

    new_input = Projection(
        dim=topology.input.dim,
        device=device,
        adjacency=new_input_adjacency,
        weights=topology.input.weights,
    )

    new_output = Projection(
        dim=topology.output.dim,
        device=device,
        adjacency=new_output_adjacency,
        weights=topology.output.weights,
    )

    return Topology(input=new_input, output=new_output, inner=new_inner)


def invert_block(
    topology: Topology,
    sampling_parameters: SamplingParameters,
    rng: random.Random,
) -> Topology:
    """Reverses the direction of the connections within a block of the adjacency matrix"""
    adjacency = topology.inner.adjacency
    n_nodes = adjacency.shape[0]

    if n_nodes < 3:
        logger.warning("Can't invert block because there are less than 3 nodes")
        return topology

    if not sampling_parameters.recurrent:
        logger.warning(
            "Can't invert block, because recurrent connections are not allowed"
        )
        return topology

    block_size = rng.randint(2, n_nodes - 1)

    # get indices for block to invert
    start_idx = rng.randint(0, n_nodes - block_size)
    end_idx = start_idx + block_size

    logger.info(
        f"Inverting block of size {block_size} at indices {start_idx} to {end_idx}"
    )

    # create new adjacency matrix
    new_adjacency = adjacency.clone()

    block_slice = slice(start_idx, end_idx)

    # invert the block
    new_adjacency[block_slice, block_slice] = new_adjacency[
        block_slice, block_slice
    ].transpose(0, 1)

    new_nonlinearity = topology.inner.nonlinearity.clone()
    new_nonlinearity[block_slice] = new_nonlinearity[block_slice].flip(0)

    new_inner = InnerTopology(
        adjacency=new_adjacency,
        nonlinearity=new_nonlinearity,
        module=topology.inner.module,
    )

    candidate = Topology(input=topology.input, output=topology.output, inner=new_inner)
    candidate = ensure_projection_connectivity(candidate, sampling_parameters, "input")
    candidate = ensure_projection_connectivity(candidate, sampling_parameters, "output")
    return candidate


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
    adjacency1 = topology1.inner.adjacency
    adjacency2 = topology2.inner.adjacency

    # get nonlinearities
    nonlinearity1 = topology1.inner.nonlinearity
    nonlinearity2 = topology2.inner.nonlinearity

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

    # create new inner topology
    new_inner = InnerTopology(
        adjacency=adjacency2,
        nonlinearity=nonlinearity2,
    )

    # TODO: check that the input and output projections are valid sizes

    return Topology(input=topology2.input, output=topology2.output, inner=new_inner)


def crossover_topologies(
    topologies: list[Topology], rng: random.Random
) -> list[Topology]:
    """
    Receives a population of topologies, pairs them up, and performs a crossover.
    The crossover itself is done by selecting a random block from one topology and
    duplicating it in the other topology.
    """
    if len(topologies) % 2 != 0:
        logger.warning("Odd number of topologies, discarding last one")

    pairs = list(zip(topologies[::2], topologies[1::2]))

    new_topologies = [crossover(*pair, rng) for pair in pairs]

    return new_topologies


def survival_selection(
    topologies: list[Topology], fitness: torch.Tensor, parameters: EvolutionParameters
) -> list[Topology]:

    if not isinstance(fitness, torch.Tensor):
        logger.warning("Fitness is not a torch.Tensor, converting to one")
        fitness = torch.tensor(fitness)
    
    if fitness.ndim == 2:
        fitness = fitness.mean(dim=1)

    # fitness proportional selection approach
    choices = torch.distributions.Categorical(probs=fitness).sample(
        (round(len(topologies) * parameters.survival_rate), )
    )
    return [topologies[i] for i in choices]


def reproduce(
    topologies: list[Topology], n_babies: int, rng: random.Random
) -> list[Topology]:

    original_size = len(topologies)

    if original_size < n_babies * 2:
        logger.info(f"Not enough topologies to reproduce, only {original_size} parents - copying parents to achieve {n_babies} babies")
        topologies = topologies + topologies[:(n_babies * 2) - len(topologies)]

    # shuffle list, then pair topologies up
    rng.shuffle(topologies)

    parents = topologies[: n_babies * 2]

    babies = crossover_topologies(parents, rng)

    return topologies[:original_size] + babies


def evolution_step(
    topologies: list[Topology],
    fitness: list[float],
    sampling_parameters: SamplingParameters,
    input_dim: int,
    output_dim: int,
    evolution_parameters: EvolutionParameters,
    rng: random.Random,
) -> list[Topology]:

    population_size = len(topologies)

    logger.info(f"Measured population size: {population_size}")

    n_babies = evolution_parameters.reproduction_amount(population_size)
    logger.info(f"Producing {n_babies} babies")

    # survival selection
    if not isinstance(fitness, torch.Tensor):
        fitness = torch.tensor(fitness)

    survived_topologies = survival_selection(topologies, fitness, evolution_parameters)
    logger.info(f"{len(survived_topologies)} topologies survived selection")

    # reproduction
    reproduced_topologies = reproduce(survived_topologies, n_babies, rng)
    logger.info(f"{len(reproduced_topologies)} babies were produced")

    logger.info(f"New population size: {len(reproduced_topologies)}")
    if len(reproduced_topologies) != population_size:
        logger.warning(f"New population size does not match expected size: {len(reproduced_topologies)} != {population_size}")

    # mutation
    mutated_topologies = []
    for topology in reproduced_topologies:
        if rng.random() < evolution_parameters.mutation_rate:
            mutated, mutation_type = mutate_topology(topology, sampling_parameters, rng)
            logger.info(f"Mutated topology with type {mutation_type}")
            mutated_topologies.append(mutated)
        else:
            mutated_topologies.append(topology)

    # every once in a while, add a new random topology
    if rng.random() < evolution_parameters.random_sample_rate:
        index = rng.randint(0, len(mutated_topologies))
        logger.info(f"Replacing topology at index {index} with a random one")
        n_nodes = rng.randint(
            min(t.n_nodes for t in mutated_topologies),
            max(t.n_nodes for t in mutated_topologies),
        )
        new_topology = sample_topology(
            n_nodes=n_nodes,
            input_dim=input_dim,
            output_dim=output_dim,
            sampling_parameters=sampling_parameters,
            device=mutated_topologies[0].inner.adjacency.device,
            rng=rng,
        )
        assert isinstance(new_topology, Topology)

        mutated_topologies[index] = new_topology

    return mutated_topologies
