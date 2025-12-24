"""
Environment Layer - Gymnasium 兼容环境

Modules:
    doudizhu_env: 主环境类
    observation: 观测空间构建
    reward: 奖励函数
    wrappers: 环境包装器
"""
from .doudizhu_env import (
    DoudizhuEnv,
    MultiAgentDoudizhuEnv,
    make_env,
)

from .observation import (
    Observation,
    ObservationBuilder,
    ActionEncoder,
    get_action_encoder,
)

from .reward import (
    RewardType,
    RewardConfig,
    RewardCalculator,
    MultiAgentReward,
    create_reward_calculator,
)

from .wrappers import (
    FlattenObservationWrapper,
    LegalActionMaskWrapper,
    SelfPlayWrapper,
    FrameStackWrapper,
    RewardScaleWrapper,
    TimeLimit,
    RecordEpisodeStatistics,
    wrap_env,
)

__all__ = [
    # env
    "DoudizhuEnv",
    "MultiAgentDoudizhuEnv",
    "make_env",
    # observation
    "Observation",
    "ObservationBuilder",
    "ActionEncoder",
    "get_action_encoder",
    # reward
    "RewardType",
    "RewardConfig",
    "RewardCalculator",
    "MultiAgentReward",
    "create_reward_calculator",
    # wrappers
    "FlattenObservationWrapper",
    "LegalActionMaskWrapper",
    "SelfPlayWrapper",
    "FrameStackWrapper",
    "RewardScaleWrapper",
    "TimeLimit",
    "RecordEpisodeStatistics",
    "wrap_env",
]
