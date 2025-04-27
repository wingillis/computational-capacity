import warnings
import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel, ConfigDict
from scipy.sparse.csgraph import shortest_path
from torch.distributions.multinomial import Multinomial


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


class MatrixContainer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    connectivity: torch.Tensor
    module: torch.Tensor
    nonlinearity: torch.Tensor

    @staticmethod
    def from_concat(concat: torch.Tensor, sizes: tuple):
        """
        Args:
            concat (torch.Tensor): Concatenated matrix.
            sizes (tuple): Column dimension of each individual matrix.
        """
        connectivity = concat[:, : sizes[0]]
        module = concat[:, sizes[0] : sizes[0] + sizes[1]]
        nonlinearity = concat[:, sizes[0] + sizes[1] :]
        return MatrixContainer(
            connectivity=connectivity, module=module, nonlinearity=nonlinearity
        )

    def concat(self):
        """Returns:
        (torch.Tensor) Concatenated matrices
        (tuple) Column dimension of each individual matrix
        """
        return (
            torch.cat((self.connectivity, self.module, self.nonlinearity), dim=1),
            self.sizes(),
        )

    def sizes(self):
        """Returns:
        (tuple) Column dimension of each individual matrix
        """
        return (
            self.connectivity.size(1),
            self.module.size(1),
            self.nonlinearity.size(1),
        )

    def plot_matrices(self):
        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        for title, mtx, a in zip(
            ["Connectivity", "Module", "Nonlinearity"],
            [self.connectivity, self.module, self.nonlinearity],
            axs.flat,
        ):
            a.imshow(mtx.cpu().numpy(), cmap="gray", aspect="auto")
            a.set(title=title)

    def plot_graph_representation(self) -> plt.Figure:
        import networkx as nx

        g = nx.from_numpy_array(
            self.connectivity.cpu().numpy(), create_using=nx.DiGraph
        )
        name_map = {
            0: "Input",
            len(self.connectivity) - 1: "Output",
        }
        g = nx.relabel_nodes(g, name_map)
        edge_label = {}

        _reverse_map = {v: k for k, v in name_map.items()}
        for edge in g.edges:
            input_edge = _reverse_map.get(edge[0], edge[0])
            nl = self.nonlinearity[input_edge].cpu().numpy().argmax()
            edge_label[edge] = NONLINEARITY_MAP[nl].__name__

        pos = nx.spring_layout(g, k=0.8)
        fig = plt.figure()
        nx.draw(g, pos=pos, with_labels=True, connectionstyle="arc3,rad=0.1")
        nx.draw_networkx_edge_labels(
            g,
            pos=pos,
            edge_labels=edge_label,
            font_color="red",
            bbox={"facecolor": "white", "alpha": 1, "pad": -2, "linewidth": 0},
        )
        return fig

    def __repr__(self):
        return (
            f"MatrixContainer: \n"
            f"  Connectivity -- shape: {self.connectivity.shape}, dtype: {self.connectivity.dtype}, device: {self.connectivity.device}, requires_grad: {self.connectivity.requires_grad}; \n"
            f"  Module       -- shape: {self.module.shape}, dtype: {self.module.dtype}, device: {self.module.device}, requires_grad: {self.module.requires_grad}; \n"
            f"  Nonlinearity -- shape: {self.nonlinearity.shape}, dtype: {self.nonlinearity.dtype}, device: {self.nonlinearity.device}, requires_grad: {self.nonlinearity.requires_grad}; \n"
        )


def align_connectivity_matrices(matrices: dict[str, MatrixContainer]):

    largest_sizes = np.zeros(3)

    for mtx in matrices.values():
        sizes = mtx.sizes()
        largest_sizes = np.maximum(largest_sizes, sizes)

    new_matrices = {}
    for _hash, mtx in matrices.items():
        pad = (largest_sizes - np.array(mtx.sizes())).astype(int)

        new_mtx = MatrixContainer(
            connectivity=F.pad(
                mtx.connectivity, (0, pad[0], 0, pad[0]), mode="constant", value=0
            ),
            module=F.pad(mtx.module, (0, pad[1], 0, pad[0]), mode="constant", value=0),
            nonlinearity=F.pad(
                mtx.nonlinearity, (0, pad[2], 0, pad[0]), mode="constant", value=0
            ),
        )
        new_matrices[_hash] = new_mtx
    return new_matrices


class Network(nn.Module):
    def __init__(
        self,
        matrices: MatrixContainer,
        input_dim: int,
        output_dim: int,
        device: str | None = None,
    ):
        super().__init__()

        self.constructor_matrices = matrices
        self.adj_matrix = matrices.connectivity
        self.network = self.generate_network(matrices, input_dim, output_dim, device)

        # initialize weights with xavier uniform
        self.apply(self._init_weights)

    @torch.no_grad()
    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=0.1)

    @staticmethod
    def generate_network(
        matrices: MatrixContainer,
        input_dim: int,
        output_dim: int,
        device: str | None = None,
    ) -> nn.Module:
        """
        Generate a network from the given matrices.

        Args:
            matrices (MatrixContainer): The container holding the connectivity, module, and nonlinearity matrices.
            input_dim (int): The input dimensionality.
            output_dim (int): The output dimensionality.
            device (str, optional): The device to create the network on. Defaults to None.

        Returns:
            nn.Module: The generated network.
        """
        # input and output dimensionality can be saved as instance properties
        network = nn.ModuleDict()

        # for each node N, find number of input connections:
        for N, outputs in enumerate(matrices.connectivity):
            inputs = matrices.connectivity[:, N]

            # find nonlinearity from map - argmax specifies location of nonlinearity in one-hot encoded matrix
            nonlinearity_fun = NONLINEARITY_MAP[
                matrices.nonlinearity[N].cpu().numpy().argmax()
            ]

            if N == 0:
                layer = nn.Linear(input_dim, outputs.sum())
            elif N == len(matrices.connectivity) - 1:
                layer = nn.Linear(inputs.sum(), output_dim)
            else:
                layer = nn.Linear(inputs.sum(), 1)
            network[f"{N}"] = nn.Sequential(
                layer,
                nonlinearity_fun(),
            )
        return network.to(device)

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
            state = torch.zeros(
                (batch, len(self.adj_matrix)), device=self.adj_matrix.device
            )

        # run through input node:
        state[:, self.adj_matrix[0]] = self.network["0"](X)

        for node in range(1, len(self.adj_matrix) - 1):
            # node receives inputs from following nodes:
            inputs = self.adj_matrix[:, node]
            state[:, node] = self.network[f"{node}"](state[:, inputs]).squeeze()

        # run through output node:
        node = len(self.adj_matrix) - 1
        inputs = self.adj_matrix[:, -1]
        out = self.network[f"{node}"](state[:, inputs])

        return out, state

    def plot_matrices(self):
        self.constructor_matrices.plot_matrices()

    def __repr__(self):
        return f"Network. Constructor matrices: {repr(self.constructor_matrices)}"


class Sampler(BaseModel):
    def connectivity(self, *args, **kwargs):
        pass

    def nonlinearity(self, *args, **kwargs):
        pass

    def module(self, *args, **kwargs):
        pass

    def sample(self, network: Network, environment=None, state=None):
        warnings.warn(
            "Sampler.sample is not implemented. This is a placeholder function.",
            UserWarning,
        )
        return {
            "connectivity": network.constructor_matrices.connectivity,
            "module": network.constructor_matrices.module,
            "nonlinearity": network.constructor_matrices.nonlinearity,
        }

    def forward(self, network: Network, environment=None, state=None):
        return Network(
            matrices=MatrixContainer(**self.sample(network, environment, state))
        )

    def __call__(self, **kwargs):
        return self.forward(**kwargs)


class Sampler_random(Sampler):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    connectivity_constraint: torch.Tensor
    module_constraint: torch.Tensor
    nonlinearity_constraint: torch.Tensor

    connectivity_bounds: tuple = (-1.0, 1.0)
    module_bounds: tuple = (1.0, 1.0)
    nonlinearity_bounds: tuple = (1.0, 1.0)

    dtype_connectivity: torch.dtype = torch.float32
    dtype_module: torch.dtype = torch.bool
    dtype_nonlinearity: torch.dtype = torch.float32

    def __init__(self, **kwargs):
        super(Sampler_random, self).__init__(**kwargs)

        # self.connectivity_constraint = kwargs.get("connectivity_constraint", torch.empty(0))
        # self.module_constraint = kwargs.get("module_constraint", torch.empty(0))
        # self.nonlinearity_constraint = kwargs.get("nonlinearity_constraint", torch.empty(0))

        # if not all([self.connectivity_constraint, self.module_constraint, self.nonlinearity_constraint]):
        #     raise ValueError("All constraints must be provided.")

        # if not all([self.connectivity_bounds, self.module_bounds, self.nonlinearity_bounds]):
        #     raise ValueError("All bounds must be provided.")

        # if not all([self.dtype_connectivity, self.dtype_module, self.dtype_nonlinearity]):
        #     raise ValueError("All dtypes must be provided.")

    def connectivity(self):
        return self._bounded_random(
            constraint=self.connectivity_constraint,
            bounds=self.connectivity_bounds,
            dtype=self.dtype_connectivity,
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
        connectivity = self.connectivity()
        nonlinearity = self.nonlinearity()
        module = self.module()

        return {
            "connectivity": connectivity,
            "module": module,
            "nonlinearity": nonlinearity,
        }


## Conventions for matrix
# 1. rows are 'from' nodes
# 2. columns are 'to' nodes
# 3. the first row is the input layer
# 4. the last column is the output layer

## to start
# 1. mask out lower triangular and diagonal elements (make DAG)
# 2. module matrix is all 'nodes', i.e. [1, 0, 0, 0]
# 3. nonlinearity matrix is all 'linear / none' and 'relu', 'tanh', i.e. [0, 1, 0, 0]


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
    n_nodes += 2  # add input and output nodes
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
    n_nodes: int = 2,
    p: float = 0.5,
    recurrent: bool = False,
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

    n_nodes += 2  # add input and output nodes

    if seed is not None:
        torch.manual_seed(seed)

    probs = torch.full((n_nodes, n_nodes), fill_value=p, device=device)
    # mask out the diagonal - no self-loops ever
    probs.fill_diagonal_(0)

    probs[:, 0] = 0  # nothing should connect to the input node
    # future TODO: allow output to connect back to hidden nodes (requires extra projection logic)
    probs[-1] = 0  # output node should not connect to anything
    if not recurrent:
        # Ensure no self-loops and no backward connections
        probs = torch.triu(probs, diagonal=1)
        probs[0, 1] = 1  # ensure input node connects to first hidden node

    sample = torch.bernoulli(probs).to(dtype=torch.bool)

    # make sure each node is reachable from the input node
    sample = add_connectivity(sample, probs, index=0)

    # make sure output node is reachable from all nodes (except input)
    sample = add_connectivity(sample.T, probs.T, index=n_nodes - 1).T

    return sample


def sample_network(
    n_nodes: int,
    connection_prob: float,
    recurrent: bool,
    device: torch.device | None = None,
) -> MatrixContainer:
    connectivity = sample_connectivity(
        n_nodes=n_nodes,
        p=connection_prob,
        recurrent=recurrent,
        device=device,
    )
    nonlinearity = sample_nonlinearity_matrix(
        n_nodes=n_nodes,
        n_nonlinearities=len(NONLINEARITY_MAP),
        device=device,
    )

    module = torch.ones((n_nodes + 2, 1), dtype=torch.bool, device=device)
    matrices = MatrixContainer(
        connectivity=connectivity,
        module=module,
        nonlinearity=nonlinearity,
    )
    return matrices
