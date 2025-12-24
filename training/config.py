"""
训练配置

定义训练相关的超参数和配置
"""
from dataclasses import dataclass, field
from typing import Optional, List, Literal


@dataclass
class TrainConfig:
    """
    训练配置

    Attributes:
        learning_rate: 学习率
        weight_decay: 权重衰减
        batch_size: 批次大小
        num_epochs: 训练轮数
        value_loss_coef: 价值损失系数
        entropy_coef: 熵正则系数
        max_grad_norm: 梯度裁剪阈值
        gamma: 折扣因子
        gae_lambda: GAE lambda 参数
    """
    # 优化器参数
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    lr_scheduler: str = "cosine"
    warmup_steps: int = 1000

    # 批次设置
    batch_size: int = 256
    num_epochs: int = 4
    accumulate_grad_batches: int = 1

    # 损失系数
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    bid_loss_coef: float = 0.1

    # 梯度裁剪
    max_grad_norm: float = 0.5

    # RL 参数
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # PPO 参数
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None

    # 采样参数
    n_steps: int = 128
    n_envs: int = 8

    # 设备
    device: str = "auto"
    precision: str = "32"

    @classmethod
    def from_dict(cls, d: dict) -> 'TrainConfig':
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class RolloutConfig:
    """
    Rollout 配置

    Attributes:
        num_workers: 工作进程数
        steps_per_worker: 每个工作进程的步数
        buffer_size: 缓冲区大小
    """
    num_workers: int = 4
    steps_per_worker: int = 256
    buffer_size: int = 100000
    temperature: float = 1.0
    epsilon: float = 0.0


@dataclass
class SelfPlayConfig:
    """
    自博弈配置

    Attributes:
        update_freq: 对手更新频率
        save_freq: 模型保存频率
        pool_size: 历史模型池大小
    """
    update_freq: int = 1000
    save_freq: int = 5000
    pool_size: int = 10
    sample_latest_prob: float = 0.8
    elo_k_factor: float = 32.0


@dataclass
class DMCConfig:
    """
    Deep Monte Carlo 配置

    DouZero 风格的训练配置
    """
    # 基础配置
    learning_rate: float = 1e-4
    batch_size: int = 32
    buffer_size: int = 100000

    # Actor 配置
    num_actors: int = 5
    epsilon: float = 0.01

    # Learner 配置
    target_update_freq: int = 1000
    save_freq: int = 10000

    # 训练设置
    total_frames: int = 10_000_000_000
    training_device: str = "cuda"
    acting_device: str = "cpu"
