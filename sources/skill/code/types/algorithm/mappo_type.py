from .algo_type import AlgorithmAbstractConfig
from dataclasses import dataclass
from .online_type import OnlineTrainConfig, OnlineModelConfig
from typing import Literal


@dataclass
class MappoAlgoConfig:

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

    train: OnlineTrainConfig
    model: OnlineModelConfig
    algo: MappoAlgoConfig
