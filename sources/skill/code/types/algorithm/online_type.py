from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class OnlineTrainConfig:

    n_rollout_threads: int = 20
    num_env_steps: int = 10000000
    episode_length: int = 200
    log_interval: int = 5
    eval_interval: int = 25
    use_valuenorm: bool = True
    use_linear_lr_decay: bool = False
    use_proper_time_limits: bool = True
    model_dir: Optional[str] = None


@dataclass
class OnlineModelConfig:


    hidden_sizes: "list[int]" = field(default_factory=lambda: [128, 128])
    activation_func: Literal["sigmoid", "tanh", "relu", "leaky_relu", "selu"] = "relu"
    use_feature_normalization: bool = True
    initialization_method: Literal[
        "xavier_uniform_",
        "orthogonal_",
        "xavier_normal_",
        "kaiming_uniform_",
        "kaiming_normal_",
    ] = "orthogonal_"
    gain: float = 0.01

    use_naive_recurrent_policy: bool = False
    use_recurrent_policy: bool = False
    recurrent_n: int = 1
    data_chunk_length: int = 10

    lr: float = 0.0005
    critic_lr: float = 0.0005
    opti_eps: float = 0.00001
    weight_decay: float = 0
    std_x_coef: float = 1
    std_y_coef: float = 0.5
