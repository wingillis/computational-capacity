import io
import gzip
import torch
import xxhash
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from typing import List, Tuple, Literal
from dataclasses import dataclass, field
from pydantic import BaseModel, ConfigDict


class SineActivation(nn.Module):

    def forward(self, x):
        return torch.sin(x)


class SoftMinus(nn.Module):

    def forward(self, x):
        return -F.softplus(-x)


NONLINEARITY_MAP = {
    0: nn.ReLU,
    1: nn.Tanh,
    2: nn.LeakyReLU,
    3: nn.Identity,
    4: SineActivation,
    5: nn.Softplus,
    6: SoftMinus,
}


@dataclass
class Projection:
    dim: int
    # (n_nodes, dim)
    adjacency: torch.Tensor 
    device: str | None = None
    weights: torch.Tensor | None = None

    fully_connected: bool = field(init=False)

    def __post_init__(self):
        # check if projection is fully connected - i.e., all projection dims are connected single nodes
        self.adjacency = self.adjacency.to(dtype=torch.bool)
        mask = self.adjacency.all(dim=0)
        not_fully_connected = (self.adjacency.any(dim=0) & ~mask).cpu().numpy()

        self.fully_connected = bool(not_fully_connected.sum() == 0)


    @property
    def hash(self) -> str:
        """
        Generate a hash for the given adjacency matrix. Provides a unique
        string for each adjacency matrix.
        """
        return xxhash.xxh64_hexdigest(self.bytes)

    @property
    def bytes(self) -> bytes:
        return b"".join(
            m.detach().cpu().numpy().tobytes()
            for m in (self.adjacency, self.weights)
            if m is not None
        )


## Conventions for adjacency matrices
# 1. rows indicate which nodes have edges coming 'from' them, i.e., outgoing node connections
# 2. columns indicate which nodes have edges going 'to' them, i.e., incoming node connections


# proposed data structure for representing the inner network topology
@dataclass
class InnerTopology:
    # (n_nodes, n_nodes)
    adjacency: torch.Tensor
    # (n_nodes, n_nonlinearities)
    nonlinearity: torch.Tensor
    # (n_nodes, 1) - optional. If not provided, module is created.
    module: torch.Tensor | None = None
    # (n_nodes, n_nodes) - optional. Not produced if not provided.
    weights: torch.Tensor | None = None

    def __post_init__(self):
        self.adjacency = self.adjacency.to(dtype=torch.bool)
        self.nonlinearity = self.nonlinearity.to(dtype=torch.bool)

        if self.module is None:
            self.module = torch.ones(
                (self.adjacency.shape[0], 1),
                dtype=torch.bool,
                device=self.adjacency.device,
            )

    @property
    def hash(self) -> str:
        """
        Generate a hash for the given matrices. Provides a unique
        string for each matrix configuration. Allows for computationally cheap
        deduplication of networks with the same topology.

        Returns:
            str: The hexadecimal digest of the hash.
        """
        return xxhash.xxh64_hexdigest(self.bytes)

    @property
    def bytes(self) -> bytes:
        # combine byte representations of the matrices
        return b"".join(
            m.detach().cpu().numpy().tobytes()
            for m in (self.adjacency, self.module, self.nonlinearity, self.weights)
            if m is not None
        )


# proposed data structure for representing the full network topology
@dataclass
class Topology:
    """
    Class representing the full network topology. Using this class, a complete
    neural network can be constructed.
    """
    input: Projection
    output: Projection
    inner: InnerTopology

    @property
    def adjacencies(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.input.adjacency, self.inner.adjacency, self.output.adjacency)

    @property
    def hash(self) -> str:
        return xxhash.xxh64_hexdigest(self.bytes)

    @property
    def bytes(self) -> bytes:
        return b"".join(m.bytes for m in self)

    @property
    def n_nodes(self) -> int:
        return self.inner.adjacency.shape[0]

    def compress(self) -> dict:
        """Saves the full network topology into a compressed format, which can be reconstructed."""
        out = {}
        for prop, m in zip(('input', 'inner', 'output'), self):
            bytes_obj = io.BytesIO()
            with gzip.GzipFile(fileobj=bytes_obj, mode="wb") as f:
                f.write(m.bytes)
            out[prop] = bytes_obj.getvalue()
            bytes_obj.close()
        return out

    @staticmethod
    def decompress(
        compressed: dict,
        n_nodes: int,
        n_nonlinearities: int,
        input_dim: int,
        output_dim: int,
    ) -> "Topology":
        """Reconstructs the full network topology from the compressed format."""
        out = {}

        for prop, m in compressed.items():
            bytes_obj = io.BytesIO(m)
            with gzip.GzipFile(fileobj=bytes_obj, mode="rb") as f:
                _bytes = f.read()
            if prop == "input":
                out[prop] = Projection(
                    adjacency=torch.from_numpy(
                        np.frombuffer(_bytes, dtype=np.bool_)
                        .reshape((n_nodes, input_dim))
                        .copy()
                    ),
                    dim=input_dim,
                )
            elif prop == "output":
                out[prop] = Projection(
                    adjacency=torch.from_numpy(
                        np.frombuffer(_bytes, dtype=np.bool_)
                        .reshape((n_nodes, output_dim))
                        .copy()
                    ),
                    dim=output_dim,
                )
            else:
                out[prop] = InnerTopology(
                    adjacency=torch.from_numpy(
                        np.frombuffer(_bytes[: n_nodes * n_nodes], dtype=np.bool_)
                        .reshape((n_nodes, n_nodes))
                        .copy()
                    ),
                    module=torch.from_numpy(
                        np.frombuffer(
                            _bytes[n_nodes * n_nodes : n_nodes * n_nodes + n_nodes],
                            dtype=np.bool_,
                        )
                        .reshape((n_nodes, 1))
                        .copy()
                    ),
                    nonlinearity=torch.from_numpy(
                        np.frombuffer(
                            _bytes[n_nodes * n_nodes + n_nodes :],
                            dtype=np.bool_,
                        )
                        .reshape((n_nodes, n_nonlinearities))
                        .copy()
                    ),
                )
        return Topology(input=out["input"], inner=out["inner"], output=out["output"])

    def __iter__(self):
        return iter((self.input, self.inner, self.output))

    def __repr__(self):
        return f"input:\n{self.input}\ninner:\n{self.inner}\noutput:\n{self.output}"


class OldTopology(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # (n_nodes, n_nodes)
    adjacency: torch.Tensor | None = None
    # (n_nodes, 1)
    module: torch.Tensor | None = None
    # (n_nodes, n_nonlinearities)
    nonlinearity: torch.Tensor | None = None
    # (n_nodes, n_nodes)
    weights: torch.Tensor | None = None
    
    def generate_hash(self) -> str:
        """
        Generate a hash for the given matrices. Provides a unique
        string for each matrix configuration. Allows for computationally cheap
        deduplication of networks with the same topology.    

        Args:
            matrices (MatrixContainer): The matrices to hash.
        Returns:
            str: The hexadecimal digest of the hash.
        """
        # Convert the matrices to a string representation
        subhashes = []
        for m in [self.adjacency, self.weights, self.module, self.nonlinearity]:
            if m is not None:
                h = xxhash.xxh64_hexdigest(m.cpu().detach().numpy().tobytes())
            else:
                h = xxhash.xxh64_hexdigest(b"")
            subhashes.append(h)
        hashed = xxhash.xxh64_hexdigest("".join(subhashes))

        # Return the hexadecimal digest of the hash
        return hashed 

    @property
    def sizes(self) -> Tuple[int, int, int]:
        return (
            self.adjacency.shape[0],
            self.module.shape[0],
            self.nonlinearity.shape[0],
        )


class Network(nn.Module):
    """
    A class representing a neural network with a specific architecture.
    The network is constructed based on the provided matrices, which define
    the weights, module, and nonlinearity of the network.

    Args:
        matrices (Topology | List[Topology]): The container holding the weights, module, and nonlinearity matrices.
        input_dim (int): The input dimensionality.
        output_dim (int): The output dimensionality.
        device (str | None): The device to create the network on. Defaults to None.
    """
    def __init__(
        self,
        toplogy: Topology | List[Topology],
        device: str | None = None,
    ):
        super().__init__()
        
        self.constructor_matrices = toplogy
        self.device = device


class ProgressiveRNN(Network):
    def __init__(
        self,
        topology: Topology,
        device: str | None = None,
        weight_init: Literal["kaiming", "binary", "normal", "uniform"] = "kaiming",
    ):
        super().__init__(
            toplogy=topology,
            device=device,
        )

        self.topology = topology
        self.weight_init = weight_init
        input_mask = topology.input.adjacency.any(dim=1)
        output_mask = topology.output.adjacency.any(dim=1)

        self.input_dim = topology.input.dim
        self.output_dim = topology.output.dim

        n_nodes = topology.inner.adjacency.shape[0]

        new_adjacency = torch.zeros(
            (n_nodes + 2, ) * 2,
            dtype=torch.bool,
            device=device,
        )
        new_adjacency[0, 1 : -1] = input_mask
        new_adjacency[1 : -1, -1] = output_mask
        new_adjacency[1 : -1, 1 : -1] = topology.inner.adjacency.to(device=device)
        self.adj_matrix = new_adjacency
        self.input_mask = new_adjacency[0]
        self.output_mask = new_adjacency[:, -1]

        self.network = self.generate_network()

        # initialize weights with xavier uniform
        self.apply(self._init_weights)

    @torch.no_grad()
    def _init_weights(self, module):
        if self.weight_init == "kaiming":
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.normal_(module.bias, std=0.1)
        elif self.weight_init == "binary":
            # weight and bias can take the values of: -1, 0, 1
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    module.weight = torch.bernoulli(torch.ones_like(module.weight) * 0.5) * -torch.bernoulli(torch.ones_like(module.weight) * 0.5)
                if module.bias is not None:
                    with torch.no_grad():
                        module.bias = torch.bernoulli(torch.ones_like(module.bias) * 0.5) * -torch.bernoulli(torch.ones_like(module.bias) * 0.5)
        elif self.weight_init == "normal":
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight)
                if module.bias is not None:
                    nn.init.normal_(module.bias)
        elif self.weight_init == "uniform":
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight)
                if module.bias is not None:
                    nn.init.uniform_(module.bias)
        else:
            raise ValueError(f"Invalid weight initialization: {self.weight_init}")

    def generate_network(
        self,
    ) -> nn.Module:
        """
        Generate a network from the given matrices.

        Args:
            matrices (Topology): The container holding the weights, module, and nonlinearity matrices.
            device (str, optional): The device to create the network on. Defaults to None.

        Returns:
            nn.Module: The generated network.
        """
        # input and output dimensionality can be saved as instance properties
        network = nn.ModuleDict()

        # TODO: for future - add nonlinearities to oinput and output layers
        network["input"] = nn.Linear(self.input_dim, self.input_mask.sum())
        network["output"] = nn.Linear(self.output_mask.sum(), self.output_dim)

        # for each node N, find number of input connections:
        for N, inputs in enumerate(self.adj_matrix.T[1:-1], start=1):
            # find nonlinearity from map - argmax specifies location of nonlinearity in one-hot encoded matrix
            nonlinearity_fun = NONLINEARITY_MAP[
                self.topology.inner.nonlinearity[N - 1].cpu().numpy()[0]
            ]
            network[f"{N}"] = nn.Sequential(
                nn.Linear(inputs.sum(), 1),
                nonlinearity_fun(),
            )

        return network.to(self.device)

    def forward(
        self,
        X: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through the network.
        Args:
            X (torch.Tensor): Input tensor.
            state (torch.Tensor, optional): State tensor. Defaults to None.
        Returns:
            torch.Tensor: Output tensor.
        """

        batch, *_ = X.shape

        if state is None:
            # initialize state with zeros
            state = torch.zeros(
                (batch, len(self.adj_matrix)), device=self.adj_matrix.device
            )

        # run through input node:
        state[:, self.input_mask] = self.network["input"](X)

        for node in range(1, len(self.adj_matrix[1:-1]) + 1):
            # node receives inputs from following nodes:
            inputs = self.adj_matrix[:, node]

            # run state through node
            state[:, node] = self.network[f"{node}"](state[:, inputs]).squeeze()

        # run projection out
        out = self.network["output"](state[:, self.output_mask])

        return out, state


class VanillaRNN(Network):
    def __init__(
        self,
        toplogies: Tuple[Topology, Topology, Topology],
        device: str | None = None,
    ):
        super().__init__(
            toplogy=toplogies,
            device=device,
        )

        self.network = self.generate_network(toplogies, device)
        self.apply(self._init_weights)

    @staticmethod
    def generate_network(
        topologies: Tuple[Topology, Topology, Topology],
        device: str | None = None,
    ) -> nn.Module:
        """
        Generate a network from the given matrices.

        Args:
            topologies (Tuple[Topology, Topology, Topology]): The container holding the input, output, and recurrent matrices.
            input_dim (int): The input dimensionality.
            output_dim (int): The output dimensionality.
            device (str, optional): The device to create the network on. Defaults to None.

        Returns:
            nn.Module: The generated network.
        """
        adj_in, adj_rec, adj_out = (t.adjacency for t in topologies)
        nl_in, nl_rec, nl_out = (t.nonlinearity for t in topologies)
        
        # n_in, n_rec, n_out = (a.shape[0] for a in (adj_in, adj_rec, adj_out))
        
        ## Make linear layers
        layer_in = nn.Linear(adj_in.shape[0], adj_in.shape[1])  ## input layer
        layer_rec = nn.Linear(adj_rec.shape[0], adj_rec.shape[1])  ## recurrent layer
        layer_out = nn.Linear(adj_out.shape[0], adj_out.shape[1])  ## output layer
        
        ## Make nonlinearity layers
        if (nl_in is not None) or (nl_rec is not None):
            raise NotImplementedError("Nonlinearity for input layer not implemented. Only ReLU is supported for now.")
        layer_nl_rec = NONLINEARITY_MAP[0]()
        layer_nl_out = NONLINEARITY_MAP[0]()
        
        ## Make network
        network = nn.ModuleDict()
        network["in"] = layer_in
        network["rec"] = layer_rec
        network["out"] = layer_out
        network["nl_rec"] = layer_nl_rec
        network["nl_out"] = layer_nl_out
        
        return network.to(device)
    
    def _init_weights(self, module):
        """
        Initialize weights of the network.
        Args:
            module (nn.Module): The module to initialize.
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=0.1)
                
    def forward(
        self,
        X: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through the network.
        Args:
            X (torch.Tensor): Input tensor.
            state (torch.Tensor, optional): State tensor. Defaults to None.
        Returns:
            torch.Tensor: Output tensor.
        """
        if state is None:
            state = torch.zeros(
                (X.shape[0], self.network["rec"].in_features), device=self.device
            )

        # run through input node:
        state = self.network["in"](X) + self.network["rec"](state)
        state = self.network["nl_rec"](state)

        # run through output node:
        out = self.network["out"](state)
        out = self.network["nl_out"](out)

        return out, state


class Sampler_random(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    weights_constraint: torch.Tensor
    module_constraint: torch.Tensor
    nonlinearity_constraint: torch.Tensor

    weights_bounds: tuple = (-1.0, 1.0)
    module_bounds: tuple = (1.0, 1.0)
    nonlinearity_bounds: tuple = (1.0, 1.0)

    dtype_weights: torch.dtype = torch.float32
    dtype_module: torch.dtype = torch.bool
    dtype_nonlinearity: torch.dtype = torch.float32

    def __init__(self, **kwargs):
        super(Sampler_random, self).__init__(**kwargs)

        # self.weights_constraint = kwargs.get("weights_constraint", torch.empty(0))
        # self.module_constraint = kwargs.get("module_constraint", torch.empty(0))
        # self.nonlinearity_constraint = kwargs.get("nonlinearity_constraint", torch.empty(0))

        # if not all([self.weights_constraint, self.module_constraint, self.nonlinearity_constraint]):
        #     raise ValueError("All constraints must be provided.")

        # if not all([self.weights_bounds, self.module_bounds, self.nonlinearity_bounds]):
        #     raise ValueError("All bounds must be provided.")

        # if not all([self.dtype_weights, self.dtype_module, self.dtype_nonlinearity]):
        #     raise ValueError("All dtypes must be provided.")

    def weights(self):
        return self._bounded_random(
            constraint=self.weights_constraint,
            bounds=self.weights_bounds,
            dtype=self.dtype_weights,
        )

    def nonlinearity(self):
        return self._bounded_random(
            constraint=self.nonlinearity_constraint,
            bounds=self.nonlinearity_bounds,
            dtype=self.dtype_nonlinearity,
        )

    def module(self):
        return self._bounded_random(
            constraint=self.module_constraint,
            bounds=self.module_bounds,
            dtype=self.dtype_module,
        )

    def _bounded_random(self, constraint, bounds, dtype):
        """
        Generate a random matrix with the same shape as the constraint matrix,
        with values bounded by the specified bounds.
        """
        rand_mat = torch.empty_like(constraint, dtype=dtype)
        if torch.is_floating_point(rand_mat):
            rand_mat.uniform_(*bounds)
        ## for boolean matrices
        elif rand_mat.dtype == torch.bool:
            rand_mat.bernoulli_(
                p=(bounds[0] + bounds[1]) / 2
            )  ## if it is a boolean array, just take the mean of the bounds values
        else:
            raise ValueError(f"Unsupported dtype: {rand_mat.dtype}")
        return constraint * rand_mat

    def sample(self, network: Network, environment=None, state=None):
        """
        Generate a randomly sampled network based on the provided constraints.
        """
        weights = self.weights()
        nonlinearity = self.nonlinearity()
        module = self.module()

        return {
            "weights": weights,
            "module": module,
            "nonlinearity": nonlinearity,
        }
