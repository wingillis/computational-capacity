import torch
from comp_capacity.repr.network import Topology
from comp_capacity.optim.random_sample import sample_topology, SamplingParameters


def test_network_compression():
    n_nodes = 10
    sampling_parameters = SamplingParameters(connection_prob=0.5, recurrent=True, use_fully_connected_projections=True)

    network = sample_topology(n_nodes, input_dim=10, output_dim=2, sampling_parameters=sampling_parameters, device='cpu')
    n_nonlinearities = network.inner.nonlinearity.shape[1]
    compressed = network.compress()
    decompressed = Topology.decompress(compressed, n_nodes, n_nonlinearities, input_dim=10, output_dim=2)
    print("Network hash:", network.hash)
    print("Decompressed hash:", decompressed.hash)

    assert torch.all(network.inner.adjacency.cpu() == decompressed.inner.adjacency.cpu())
    assert torch.all(network.inner.nonlinearity.cpu() == decompressed.inner.nonlinearity.cpu())
    assert torch.all(network.inner.module.cpu() == decompressed.inner.module.cpu())
    assert torch.all(network.input.adjacency.cpu() == decompressed.input.adjacency.cpu())
    assert torch.all(network.output.adjacency.cpu() == decompressed.output.adjacency.cpu())
    assert network.hash == decompressed.hash

if __name__ == "__main__":
    test_network_compression()