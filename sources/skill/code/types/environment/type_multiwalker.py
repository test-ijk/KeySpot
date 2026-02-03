from dataclasses import dataclass, field
from typing import Any, Optional, List, cast
import omegaconf


@dataclass
class MultiWalkerConfig:


    n_walkers: int = 3
    position_noise: float = 1e-3
    angle_noise: float = 1e-3
    forward_reward: float = 1.0
    terminate_reward: float = -100.0
    fall_reward: float = -10.0
    shared_reward: bool = True
    terminate_on_fall: bool = True
    remove_on_fall: bool = True
    terrain_length: int = 200
    max_cycles: int = 500
    scenario: str = "default"
    terrain_config: Optional[List[dict]] = None
    custom: Optional[dict] = None


@dataclass
class MultiWalkerTweakConfig:


    tweak_types: List[str] = field(default_factory=list)

    n_walkers: Optional[int] = None
    position_noise: Optional[float] = None
    angle_noise: Optional[float] = None
    forward_reward: Optional[float] = None
    terminate_reward: Optional[float] = None
    fall_reward: Optional[float] = None
    shared_reward: Optional[bool] = None
    terminate_on_fall: Optional[bool] = None
    remove_on_fall: Optional[bool] = None
    terrain_length: Optional[int] = None
    max_cycles: Optional[int] = None
    scenario: Optional[str] = None
    custom: Optional[dict] = None

    terrain_config: Optional[List[dict]] = None

    reward_factor: Optional[float] = None
    move_idle_reward: Optional[float] = None


@dataclass
class DisturbanceConfig:
    name: str
    start_at: int
    end_at: int
    disturbance_args: Any


@dataclass
class MultiWalkerEvalScenarioConfig:
    name: str
    desc: str
    is_raw: Optional[bool] = False
    disturbances: Optional[List[DisturbanceConfig]] = None


def _to_dict(cfg1) -> dict:
    dict_result = omegaconf.OmegaConf.to_container(
        cfg1, resolve=True, throw_on_missing=True
    )
    if type(dict_result) is not dict:
        raise ValueError("dict_result is not a dict")
    return dict_result


def multiwalker_customize_dict(cfg, algo_dict: dict, env_dict: dict):
    from ..task.train_type import TrainConfig

    cfg = cast(TrainConfig, cfg)
    # disturbances
    env_dict["custom"]["eval_disturb"] = _to_dict(cfg.eval_scenario).get(
        "disturbances", []
    )
    if algo_dict["train"].get("episode_length") is not None:
        algo_dict["train"]["episode_length"] = env_dict["max_cycles"]

    return algo_dict, env_dict
