from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
from mapdn.environments.var_voltage_control.disturbances.DisturbanceConfig import (
    DisturbanceConfig,
)


class AlgorithmType(Enum):
    coma = "coma"
    iddpg = "iddpg"
    ippo = "ippo"
    maddpg = "maddpg"
    mappo = "mappo"
    matd3 = "matd3"
    sqddpg = "sqddpg"


class EnvType(Enum):
    var_voltage_control = "var_voltage_control"


class ModeType(Enum):
    distributed = "distributed"
    # decentralised = "decentralised"


class ScenarioType(Enum):
    case33_3min_final = "case33_3min_final"
    case141_3min_final = "case141_3min_final"
    case322_3min_final = "case322_3min_final"


class VoltageBarrierType(Enum):
    l1 = "l1"
    l2 = "l2"
    bowl = "bowl"


class TestModeType(Enum):
    single = "single"
    batch = "batch"


@dataclass
class TrainConfig:
    alias: str
    save_path: str
    env: EnvType
    mode: ModeType

    alg: AlgorithmType
    scenario: ScenarioType
    voltage_barrier_type: VoltageBarrierType


@dataclass
class AlgOverrideConfig:
    max_steps: int


@dataclass
class TrainHydraEntryConfig:
    save_group: str
    run_group: str
    train_config: TrainConfig
    disturbances: Optional[List[DisturbanceConfig]]
    alg_override: Optional[AlgOverrideConfig]
