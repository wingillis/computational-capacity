import torch
from torch import nn
from pydantic import BaseModel


class MatrixContainer(BaseModel):
    connectivity: torch.tensor
    module: torch.tensor
    nonlinearity: torch.tensor

    def concat(self):
        """Returns:
        (torch.tensor) Concatenated matrices
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


class Network(nn.Module):
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


class Sampler(BaseModel):
    connectivity_priors: torch.tensor
    nonlinearity_priors: torch.tensor
    module_priors: torch.tensor

    def connectivity(self, input_matrix=None, size=2):
        # first row is assigned to handle the input

        # last column is assigned to the ouput
        pass

    def nonlinearity(self, input_matrix=None):
        pass

    def module(self, input_matrix=None):
        pass


# TODO: mutate?
