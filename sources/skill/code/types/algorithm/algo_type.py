from dataclasses import dataclass
from typing import Any


@dataclass
class LoggerConfig:

    log_dir: str = "./results"


@dataclass
class SeedConfig:


    seed_specify: bool = True
    seed: int = 1


@dataclass
class DeviceConfig:


    cuda: bool = True
    cuda_deterministic: bool = True
    torch_threads: int = 4


@dataclass
class EvalConfig:

    use_eval: bool = True
    n_eval_rollout_threads: int = 10
    eval_episodes: int = 20


@dataclass
class RenderConfig:


    use_render: bool = False
    render_episodes: int = 10


@dataclass
class AlgorithmAbstractConfig:


    logger: LoggerConfig
    seed: SeedConfig
    device: DeviceConfig
    eval: EvalConfig
    render: RenderConfig

    train: Any
    model: Any
    algo: Any
