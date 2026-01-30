from .algo_type import AlgorithmAbstractConfig
from dataclasses import dataclass
from .online_type import OnlineTrainConfig, OnlineModelConfig
from typing import Literal


@dataclass
class MappoAlgoConfig:
    """Mappo算法配置

    包含PPO算法相关的超参数配置，用于控制训练过程中的各种行为。

    Attributes:
        ppo_epoch: actor更新的epoch数
        critic_epoch: critic更新的epoch数
        use_clipped_value_loss: 是否使用clipped value loss
        clip_param: clip参数
        actor_num_mini_batch: actor更新时每个epoch的mini-batch数量
        critic_num_mini_batch: critic更新时每个epoch的mini-batch数量
        entropy_coef: actor loss中熵项的系数
        value_loss_coef: value loss的系数
        use_max_grad_norm: 是否裁剪梯度范数
        max_grad_norm: 最大梯度范数
        use_gae: 是否使用广义优势估计(GAE)
        gamma: 折扣因子
        gae_lambda: GAE lambda参数
        use_huber_loss: 是否使用huber loss
        use_policy_active_masks: 是否使用policy active masks
        huber_delta: huber delta参数
        action_aggregation: 多维动作概率聚合方法，可选prod或mean
        share_param: 是否在actors之间共享参数
        fixed_order: 是否使用固定的优化顺序
    """

    # PPO参数
    ppo_epoch: int = 5
    critic_epoch: int = 5

    use_clipped_value_loss: bool = True
    clip_param: float = 0.2

    actor_num_mini_batch: int = 1

    critic_num_mini_batch: int = 1
    entropy_coef: float = 0.01
    value_loss_coef: float = 1.0
    use_max_grad_norm: bool = True
    max_grad_norm: float = 10.0
    use_gae: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    use_huber_loss: bool = True
    use_policy_active_masks: bool = True
    huber_delta: float = 10.0
    action_aggregation: Literal["prod", "mean"] = "prod"
    share_param: bool = True
    fixed_order: bool = True


@dataclass
class MappoConfig(AlgorithmAbstractConfig):
    """Mappo配置"""

    train: OnlineTrainConfig
    model: OnlineModelConfig
    algo: MappoAlgoConfig
