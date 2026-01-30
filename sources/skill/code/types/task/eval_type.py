from dataclasses import dataclass
from ..environment.type_multiwalker import MultiWalkerTweakConfig
from .train_type import TrainConfig
from typing import Any, Optional, List, Union
from harl.envs.harl_env_with_events import Event


@dataclass
class EvalModelConfig:
    save_group: str
    model_path: str = "default"


@dataclass
class EvalGeneralConfig:
    seed: int
    eval_episodes: int
    eval_threads: int


@dataclass
class EvalFunctionsConfig:
    load_results: bool = False
    result_file_name: str = "hi"

    render: bool = False
    render_episodes: int = 5

    ablation: bool = False
    use_gpu: bool = False
    export_angle_data: bool = False


@dataclass
class EvalSettingsConfig:
    general: EvalGeneralConfig
    functions: EvalFunctionsConfig


@dataclass
class DisturbanceConfig:
    name: str
    start_at: int
    end_at: int
    disturbance_args: Any


@dataclass
class EvalScenarioConfig:
    name: str
    desc: str
    is_raw: Optional[bool] = False
    env_tweak: Optional[MultiWalkerTweakConfig] = None
    disturbances: Optional[List[DisturbanceConfig]] = None
    events: Union[List[Event], None] = None


@dataclass
class EvalConfig(TrainConfig):
    model: EvalModelConfig
    eval_settings: EvalSettingsConfig
    eval_scenario: EvalScenarioConfig
