from dataclasses import dataclass
from typing import Optional, Union
from ..algorithm.mappo_type import MappoConfig
from ..environment.type_multiwalker import (
    MultiWalkerTweakConfig,
    MultiWalkerConfig,
    MultiWalkerEvalScenarioConfig,
)
from ..environment.type_sumo import (
    SumoTweakConfig,
    SumoEnvConfig,
    SumoEvalScenarioConfig,
)
from harl.envs.harl_env_with_events import Event


@dataclass
class EvalScenarioConfig:
    name: str
    desc: str
    events: Optional[list[Event]] = None


EnvConfigType = Union[MultiWalkerConfig, SumoEnvConfig]
TweakConfigType = Union[MultiWalkerTweakConfig, SumoTweakConfig]
ScenarioConfigType = Union[MultiWalkerEvalScenarioConfig, SumoEvalScenarioConfig]


@dataclass
class WandbConfig:

    wandb_name: str
    wandb_group: str = "latest"
    wandb_project: str = "mw_skill"


@dataclass
class ModelConfig:

    should_load_model: bool = False
    load_group: Optional[str] = None
    save_group: str = "latest"

    def __post_init__(self):

        if self.should_load_model and self.load_group is None:
            raise ValueError("")


@dataclass
class AlgorithmConfig:
    name: str


@dataclass
class EnvironmentConfig:
    name: str
    env_tweak: TweakConfigType


@dataclass
class ScenarioConfig:
    name: str


@dataclass
class TrainConfig:


    wandb: WandbConfig
    model: ModelConfig

    algorithm: AlgorithmConfig
    algorithm_parameters: MappoConfig

    environment: EnvironmentConfig
    environment_parameters: EnvConfigType

    scenario: ScenarioConfig
    environment_scenario: Optional[
        dict
    ]  # environment scenario updates some parameters of environment_parameters

    eval_scenario: ScenarioConfigType
