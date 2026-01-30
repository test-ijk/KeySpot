"""SUMO Environment for Traffic Signal Control."""

from gymnasium.envs.registration import register

# 导入observations模块，使其可以通过from sumo_rl.environment import observations访问
from . import observations

register(
    id="sumo-rl-v0",
    entry_point="sumo_rl.environment.env:SumoEnvironment",
    kwargs={"single_agent": True},
)
