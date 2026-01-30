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
    decentralised = "decentralised"


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
class EvalConfig:
    save_path: str
    alg: AlgorithmType
    env: EnvType
    alias: str
    mode: ModeType
    scenario: ScenarioType
    voltage_barrier_type: VoltageBarrierType
    test_mode: TestModeType
    test_day: int
    render: bool
    eval_episodes: int


@dataclass
class EvalHydraEntryConfig:
    save_group: str
    run_group: str
    eval_config: EvalConfig
    disturbances: Optional[List[DisturbanceConfig]]
