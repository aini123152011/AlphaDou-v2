"""
Evaluation Layer - 评估框架

Modules:
    evaluator: 评估器和智能体
    arena: 对战竞技场
    metrics: 评估指标
    elo: ELO 评分系统
"""
from .evaluator import (
    EvalResult,
    Agent,
    RandomAgent,
    RuleBasedAgent,
    ModelAgent,
    Evaluator,
    MultiAgentEvaluator,
)
from .arena import (
    MatchResult,
    TournamentResult,
    Arena,
    ParallelArena,
    LeaderBoard,
)
from .metrics import (
    MetricType,
    GameMetrics,
    MetricsCollector,
    RunningStats,
    MetricsAggregator,
    compute_action_entropy,
    compute_value_accuracy,
    compute_policy_kl,
    compute_explained_variance,
)
from .elo import (
    PlayerRating,
    EloSystem,
    MultiPlayerElo,
    TrueSkillLite,
    GlickoSystem,
)

__all__ = [
    # evaluator
    "EvalResult",
    "Agent",
    "RandomAgent",
    "RuleBasedAgent",
    "ModelAgent",
    "Evaluator",
    "MultiAgentEvaluator",
    # arena
    "MatchResult",
    "TournamentResult",
    "Arena",
    "ParallelArena",
    "LeaderBoard",
    # metrics
    "MetricType",
    "GameMetrics",
    "MetricsCollector",
    "RunningStats",
    "MetricsAggregator",
    "compute_action_entropy",
    "compute_value_accuracy",
    "compute_policy_kl",
    "compute_explained_variance",
    # elo
    "PlayerRating",
    "EloSystem",
    "MultiPlayerElo",
    "TrueSkillLite",
    "GlickoSystem",
]
