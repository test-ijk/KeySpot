from dataclasses import dataclass
from typing import Any


@dataclass
class LoggerConfig:
    """日志配置
    log_dir: 日志保存目录
    """

    log_dir: str = "./results"


@dataclass
class SeedConfig:
    """随机种子配置

    seed_specify: 是否指定种子
    seed: 种子
    """

    seed_specify: bool = True
    seed: int = 1


@dataclass
class DeviceConfig:
    """设备配置

    cuda: 是否使用cuda
    cuda_deterministic: 启用后，确保运算结果稳定一致
    torch_threads: torch线程数
    """

    cuda: bool = True
    cuda_deterministic: bool = True
    torch_threads: int = 4


@dataclass
class EvalConfig:
    """评估配置

    use_eval: 是否使用评估
    n_eval_rollout_threads: 评估时的并行环境数量
    eval_episodes: 每次评估的回合数
    eval_interval: 评估间隔
    """

    use_eval: bool = True
    n_eval_rollout_threads: int = 10
    eval_episodes: int = 20


@dataclass
class RenderConfig:
    """渲染配置

    use_render: 是否使用渲染
    render_episodes: 渲染的回合数
    """

    use_render: bool = False
    render_episodes: int = 10


@dataclass
class AlgorithmAbstractConfig:
    """算法配置基础类，定义所有算法通用的配置参数"""

    logger: LoggerConfig
    seed: SeedConfig
    device: DeviceConfig
    eval: EvalConfig
    render: RenderConfig

    train: Any
    model: Any
    algo: Any
