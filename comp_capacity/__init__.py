"""This code gets run when the package or a submodule is imported."""
import logging
import gymnasium as gym

#### register all custom envs ####

gym.register(
    id="NextStepFunction-v0",
    entry_point="comp_capacity.sim.pattern_complete:NextStepFunction",
)

gym.register(
    id="SequentialPatterns-v0",
    entry_point="comp_capacity.sim.pattern_complete:SequentialPatterns",
)

#### set up logging ####

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s")

logger = logging.getLogger("comp_capacity")
logger.setLevel(logging.INFO)

# save to file
handler = logging.FileHandler("comp_capacity.log")
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.WARNING)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
