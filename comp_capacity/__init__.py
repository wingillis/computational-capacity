import gymnasium as gym

__all__ = [
    "optim",
    "repr",
    # "sim",
]

for pkg in __all__:
    exec('from . import ' + pkg)

gym.register(
    id="NextStepFunction-v0",
    entry_point="comp_capacity.sim.pattern_complete:NextStepFunction",
)

gym.register(
    id="SequentialPatterns-v0",
    entry_point="comp_capacity.sim.pattern_complete:SequentialPatterns",
)
