import torch
import random
import matplotlib.pyplot as plt
from comp_capacity.optim.genetic_evolution import duplicate_block
from comp_capacity.repr.network import Topology
from comp_capacity.optim.genetic_evolution import SamplingParameters

adj = torch.tensor([
    [0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0],
], dtype=torch.bool)

topo = Topology(adjacency=adj, nonlinearity=torch.arange((10)).reshape(5, 2))

torch.manual_seed(42)

sampling_parameters = SamplingParameters(connection_prob=0.75, recurrent=False)

rng = random.Random(42)

dup = duplicate_block(topo, rng=rng)

plt.figure(figsize=(6, 6))
plt.imshow(topo.adjacency.cpu().numpy())
plt.title("Original Topology")

plt.figure(figsize=(6, 6))
plt.imshow(dup.adjacency.cpu().numpy())
plt.title("Duplicated Topology")
plt.show()