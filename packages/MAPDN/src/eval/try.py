from dataclasses import dataclass
from enum import Enum
from omegaconf import OmegaConf


class AlgorithmType(Enum):
    coma = "coma"
    iddpg = "iddpg"
    ippo = "ippo"
    maddpg = "maddpg"
    mappo = "mappo"
    matd3 = "matd3"
    sqddpg = "sqddpg"


@dataclass
class EvalConfig:
    alg: AlgorithmType = AlgorithmType.coma


conf1 = OmegaConf.structured(EvalConfig)
conf2 = OmegaConf.structured(EvalConfig(alg="iddpg"))
conf2 = OmegaConf.structured(EvalConfig(alg="iddpg1"))
