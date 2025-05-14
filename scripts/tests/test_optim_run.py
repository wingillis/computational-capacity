"""
Test optimization run with random sampling
"""

from comp_capacity.optim.run import run
from comp_capacity.optim.random_sample import SamplingParameters
from comp_capacity.optim.genetic_evolution import EvolutionParameters


def test_random_optim_run():
    sampling_parameters = SamplingParameters(
        connection_prob=0.5,
        recurrent=True,
        increase_node_prob=1 / 100,
        modify_projections=False,
    )
    run(
        n_networks=10,
        batch_size=10,
        n_steps=10,
        n_init_nodes=5,
        algorithm="random_sampling",
        sampling_parameters=sampling_parameters,
        extra_params={},
        seed=5,
    )


def test_genetic_optim_run():
    sampling_parameters = SamplingParameters(
        connection_prob=0.5,
        recurrent=True,
        increase_node_prob=1 / 100,
        modify_projections=False,
    )
    evolution_parameters = EvolutionParameters(
        survival_rate=0.5,
        mutation_rate=0.1,
        random_sample_rate=0.1,
    )
    run(
        n_networks=10,
        batch_size=10,
        n_steps=10,
        n_init_nodes=5,
        algorithm="genetic_evolution",
        sampling_parameters=sampling_parameters,
        extra_params={"evolution_parameters": evolution_parameters},
        seed=5,
    )

if __name__ == "__main__":
    test_random_optim_run()
    test_genetic_optim_run()