"""
Training Layer - 训练框架

Modules:
    config: 训练配置
    buffer: 经验缓冲区
    rollout: Rollout Worker
    learner: 损失计算与更新
    trainer: 训练循环
"""
from .config import (
    TrainConfig,
    RolloutConfig,
    SelfPlayConfig,
    DMCConfig,
)
from .buffer import (
    Transition,
    Trajectory,
    ReplayBuffer,
    RolloutBuffer,
    PrioritizedReplayBuffer,
)
from .rollout import (
    RolloutResult,
    RolloutWorker,
    VectorRolloutWorker,
    AsyncRolloutManager,
    collect_rollout,
)
from .learner import (
    LossInfo,
    PPOLoss,
    A2CLoss,
    DMCLoss,
    Learner,
)
from .trainer import (
    TrainStats,
    Callback,
    TensorBoardCallback,
    WandbCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
    Trainer,
    DMCTrainer,
    SelfPlayTrainer,
)

__all__ = [
    # config
    "TrainConfig",
    "RolloutConfig",
    "SelfPlayConfig",
    "DMCConfig",
    # buffer
    "Transition",
    "Trajectory",
    "ReplayBuffer",
    "RolloutBuffer",
    "PrioritizedReplayBuffer",
    # rollout
    "RolloutResult",
    "RolloutWorker",
    "VectorRolloutWorker",
    "AsyncRolloutManager",
    "collect_rollout",
    # learner
    "LossInfo",
    "PPOLoss",
    "A2CLoss",
    "DMCLoss",
    "Learner",
    # trainer
    "TrainStats",
    "Callback",
    "TensorBoardCallback",
    "WandbCallback",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "Trainer",
    "DMCTrainer",
    "SelfPlayTrainer",
]
