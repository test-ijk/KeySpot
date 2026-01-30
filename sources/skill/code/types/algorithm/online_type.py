from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class OnlineTrainConfig:
    """Online-训练配置.

    num_env_steps: 总训练步数

    n_rollout_threads: 训练数据收集的并行环境数量
    episode_length: 每个环境每次训练数据收集的步数

    log_interval: 日志记录间隔
    eval_interval: 评估间隔

    use_valuenorm: 是否使用ValueNorm
    use_linear_lr_decay: 是否使用线性学习率衰减
    use_proper_time_limits: 当回合结束时是否考虑截断情况

    model_dir: 如果设置，从此目录加载模型；否则随机初始化模型
    """

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
    """Online-模型配置.

    hidden_sizes: MLP网络的隐藏层大小列表
    activation_func: 激活函数，可选：sigmoid, tanh, relu, leaky_relu, selu
    use_feature_normalization: 是否使用特征归一化
    initialization_method: 网络参数初始化方法，如：xavier_uniform_, orthogonal_
    gain: 网络输出层的增益

    use_naive_recurrent_policy: 是否使用简单循环策略（训练时数据不分块）
    use_recurrent_policy: 是否使用循环策略（训练时数据分块）
    recurrent_n: 循环层数量
    data_chunk_length: 数据块长度，仅在use_recurrent_policy为True时有用

    lr: actor学习率
    critic_lr: critic学习率
    opti_eps: Adam优化器的eps参数
    weight_decay: Adam优化器的权重衰减
    std_x_coef: 对角高斯分布的x系数参数
    std_y_coef: 对角高斯分布的y系数参数
    """

    # 网络参数
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

    # 循环神经网络参数
    use_naive_recurrent_policy: bool = False
    use_recurrent_policy: bool = False
    recurrent_n: int = 1
    data_chunk_length: int = 10

    # 优化器参数
    lr: float = 0.0005
    critic_lr: float = 0.0005
    opti_eps: float = 0.00001
    weight_decay: float = 0
    std_x_coef: float = 1
    std_y_coef: float = 0.5
