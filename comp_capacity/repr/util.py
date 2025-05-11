import numpy as np
import torch.nn.functional as F
from comp_capacity.repr.network import Topology

def align_connectivity_matrices(matrices: dict[str, Topology]):

    largest_sizes = np.zeros(3)

    for mtx in matrices.values():
        largest_sizes = np.maximum(largest_sizes, mtx.sizes)

    new_matrices = {}
    for _hash, mtx in matrices.items():
        pad = (largest_sizes - np.array(mtx.sizes())).astype(int)

        new_mtx = Topology(
            adjacency=F.pad(
                mtx.connectivity, (0, pad[0], 0, pad[0]), mode="constant", value=0
            ),
            module=F.pad(mtx.module, (0, pad[1], 0, pad[0]), mode="constant", value=0),
            nonlinearity=F.pad(
                mtx.nonlinearity, (0, pad[2], 0, pad[0]), mode="constant", value=0
            ),
        )
        new_matrices[_hash] = new_mtx
    return new_matrices