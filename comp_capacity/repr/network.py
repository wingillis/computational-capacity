import warnings
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from pydantic import BaseModel, ConfigDict

class MatrixContainer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    connectivity: torch.Tensor
    module: torch.Tensor
    nonlinearity: torch.Tensor

    def concat(self):
        """Returns:
        (torch.Tensor) Concatenated matrices
        (tuple) Column dimension of each individual matrix
        """
        column_sizes = (
            self.connectivity.size(1),
            self.module.size(1),
            self.nonlinearity.size(1),
        )
        return (
            torch.cat((self.connectivity, self.module, self.nonlinearity), dim=1),
            column_sizes,
        )
        
    def plot_matrices(self):
        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        axs[0].imshow(self.connectivity.cpu().numpy(), cmap='gray', aspect='auto')
        axs[0].set_title('Connectivity')
        axs[1].imshow(self.module.cpu().numpy(), cmap='gray', aspect='auto')
        axs[1].set_title('Module')
        axs[2].imshow(self.nonlinearity.cpu().numpy(), cmap='gray', aspect='auto')
    
    def __repr__(self):
        return (
            f"MatrixContainer: \n"
            f"  Connectivity -- shape: {self.connectivity.shape}, dtype: {self.connectivity.dtype}, device: {self.connectivity.device}, requires_grad: {self.connectivity.requires_grad}; \n"
            f"  Module       -- shape: {self.module.shape}, dtype: {self.module.dtype}, device: {self.module.device}, requires_grad: {self.module.requires_grad}; \n"
            f"  Nonlinearity -- shape: {self.nonlinearity.shape}, dtype: {self.nonlinearity.dtype}, device: {self.nonlinearity.device}, requires_grad: {self.nonlinearity.requires_grad}; \n"
        )


class Network(torch.nn.Module):
    def __init__(self, matrices: MatrixContainer):
        super().__init__()

        self.constructor_matrices = matrices
        self.network = self.generate_network(matrices)

    @staticmethod
    def generate_network(matrices: MatrixContainer):
        # input and output dimensionality can be saved as instance properties
        pass

    def forward(self):
        pass
    
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

    connectivity_constraint:    torch.Tensor
    module_constraint:          torch.Tensor
    nonlinearity_constraint:    torch.Tensor
    
    connectivity_bounds:    tuple = (-1.0, 1.0)
    module_bounds:          tuple = (1.0, 1.0)
    nonlinearity_bounds:    tuple = (1.0, 1.0)
    
    dtype_connectivity: torch.dtype = torch.float32
    dtype_module:       torch.dtype = torch.bool
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
            rand_mat.bernoulli_(p=(bounds[0] + bounds[1]) / 2)  ## if it is a boolean array, just take the mean of the bounds values
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
