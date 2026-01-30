from harl.envs.sumo.pettingzoo_sumo_env import SumoEnvConfig
from dataclasses import dataclass, field
from typing import Optional, List, Union, cast
import omegaconf
from harl.envs.harl_env_with_events import Event


@dataclass
class SumoTweakConfig:
    tweak_types: List[str] = field(default_factory=list)

    net_file: Optional[str] = None
    route_file: Optional[str] = None
    out_csv_name: Optional[str] = None
    use_gui: Optional[bool] = None
    virtual_display: Optional[tuple[int, int]] = None
    begin_time: Optional[int] = None
    num_seconds: Optional[int] = None
    max_depart_delay: Optional[int] = None
    waiting_time_memory: Optional[int] = None
    time_to_teleport: Optional[int] = None
    delta_time: Optional[int] = None
    yellow_time: Optional[int] = None
    min_green: Optional[int] = None
    max_green: Optional[int] = None
    enforce_max_green: Optional[bool] = None
    single_agent: Optional[bool] = None
    reward_fn: Optional[str] = None
    reward_weights: Optional[list[float]] = None
    add_system_info: Optional[bool] = None
    add_per_agent_info: Optional[bool] = None
    sumo_seed: Optional[Union[str, int]] = None
    fixed_ts: Optional[bool] = None
    sumo_warnings: Optional[bool] = None
    additional_sumo_cmd: Optional[str] = None
    render_mode: Optional[str] = None

    max_cycles: Optional[int] = None


@dataclass
class SumoEvalScenarioConfig:
    name: str
    events: Optional[list[Event]] = None


def _to_dict(cfg1) -> dict:
    dict_result = omegaconf.OmegaConf.to_container(
        cfg1, resolve=True, throw_on_missing=True
    )
    if type(dict_result) is not dict:
        raise ValueError("dict_result is not a dict")
    return dict_result


def sumo_customize_dict(cfg, algo_dict: dict, env_dict: dict, save_group: str):
    from ..task.train_type import TrainConfig

    cfg = cast(TrainConfig, cfg)
    # disturbances的引入
    if algo_dict["train"].get("episode_length") is not None:
        max_cycles = (env_dict["num_seconds"]) // env_dict["delta_time"]
        algo_dict["train"]["episode_length"] = max_cycles - 1

    env_dict["out_csv_name"] = f"{env_dict['out_csv_name']}/{save_group}/log"
    print(f"env_dict['out_csv_name'] = {env_dict['out_csv_name']}")
    return algo_dict, env_dict


__all__ = [
    "SumoEnvConfig",
    "SumoTweakConfig",
    "SumoEvalScenarioConfig",
    "sumo_customize_dict",
]
