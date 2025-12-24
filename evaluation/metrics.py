"""
评估指标

定义和计算各种评估指标
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from enum import Enum


class MetricType(Enum):
    """指标类型"""
    WIN_RATE = "win_rate"
    REWARD = "reward"
    LENGTH = "length"
    SPRING_RATE = "spring_rate"
    BOMB_RATE = "bomb_rate"
    ACTION_ENTROPY = "action_entropy"
    VALUE_ACCURACY = "value_accuracy"


@dataclass
class GameMetrics:
    """单局游戏指标"""
    winner: str
    landlord: str
    farmers: Tuple[str, str]
    length: int
    bombs: int
    spring: bool
    rewards: Dict[str, float] = field(default_factory=dict)
    actions: Dict[str, List[int]] = field(default_factory=dict)


class MetricsCollector:
    """
    指标收集器

    收集和计算游戏指标
    """

    def __init__(self):
        self.games: List[GameMetrics] = []
        self._stats: Dict[str, Dict] = defaultdict(lambda: defaultdict(list))

    def add_game(self, metrics: GameMetrics):
        """添加游戏指标"""
        self.games.append(metrics)

        # 更新统计
        landlord = metrics.landlord
        self._stats[landlord]["as_landlord"].append(1)
        self._stats[landlord]["landlord_wins"].append(
            1 if metrics.winner == "landlord" else 0
        )

        for farmer in metrics.farmers:
            self._stats[farmer]["as_farmer"].append(1)
            self._stats[farmer]["farmer_wins"].append(
                1 if metrics.winner == "farmer" else 0
            )

        all_players = [landlord] + list(metrics.farmers)
        for player in all_players:
            self._stats[player]["games"].append(1)
            self._stats[player]["lengths"].append(metrics.length)
            self._stats[player]["bombs"].append(metrics.bombs)
            self._stats[player]["springs"].append(1 if metrics.spring else 0)

            if player in metrics.rewards:
                self._stats[player]["rewards"].append(metrics.rewards[player])

    def compute_metrics(self, player: Optional[str] = None) -> Dict[str, float]:
        """
        计算指标

        Args:
            player: 指定玩家，None 表示全局

        Returns:
            指标字典
        """
        if player is not None:
            stats = self._stats[player]
            n_games = len(stats["games"])

            if n_games == 0:
                return {}

            return {
                "games": n_games,
                "win_rate": (
                    (sum(stats["landlord_wins"]) + sum(stats["farmer_wins"])) /
                    n_games
                ),
                "landlord_games": len(stats["as_landlord"]),
                "landlord_win_rate": (
                    sum(stats["landlord_wins"]) / len(stats["as_landlord"])
                    if stats["as_landlord"] else 0.0
                ),
                "farmer_games": len(stats["as_farmer"]),
                "farmer_win_rate": (
                    sum(stats["farmer_wins"]) / len(stats["as_farmer"])
                    if stats["as_farmer"] else 0.0
                ),
                "avg_length": np.mean(stats["lengths"]),
                "avg_bombs": np.mean(stats["bombs"]),
                "spring_rate": np.mean(stats["springs"]),
                "avg_reward": np.mean(stats["rewards"]) if stats["rewards"] else 0.0,
            }
        else:
            # 全局统计
            n_games = len(self.games)
            if n_games == 0:
                return {}

            landlord_wins = sum(1 for g in self.games if g.winner == "landlord")
            farmer_wins = sum(1 for g in self.games if g.winner == "farmer")

            return {
                "total_games": n_games,
                "landlord_win_rate": landlord_wins / n_games,
                "farmer_win_rate": farmer_wins / n_games,
                "avg_length": np.mean([g.length for g in self.games]),
                "avg_bombs": np.mean([g.bombs for g in self.games]),
                "spring_rate": np.mean([1 if g.spring else 0 for g in self.games]),
            }

    def reset(self):
        """重置"""
        self.games.clear()
        self._stats.clear()


class RunningStats:
    """
    运行时统计

    在线计算均值和方差
    """

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')

    def update(self, x: float):
        """更新统计"""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        self.min_val = min(self.min_val, x)
        self.max_val = max(self.max_val, x)

    @property
    def variance(self) -> float:
        """方差"""
        if self.n < 2:
            return 0.0
        return self.M2 / (self.n - 1)

    @property
    def std(self) -> float:
        """标准差"""
        return np.sqrt(self.variance)

    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            "count": self.n,
            "mean": self.mean,
            "std": self.std,
            "min": self.min_val if self.n > 0 else 0.0,
            "max": self.max_val if self.n > 0 else 0.0,
        }


class MetricsAggregator:
    """
    指标聚合器

    聚合多个来源的指标
    """

    def __init__(self):
        self.metrics: Dict[str, RunningStats] = defaultdict(RunningStats)

    def add(self, name: str, value: float):
        """添加指标值"""
        self.metrics[name].update(value)

    def add_dict(self, values: Dict[str, float]):
        """添加字典形式的指标"""
        for name, value in values.items():
            self.add(name, value)

    def get(self, name: str) -> Dict[str, float]:
        """获取指标统计"""
        if name in self.metrics:
            return self.metrics[name].to_dict()
        return {}

    def get_all(self) -> Dict[str, Dict[str, float]]:
        """获取所有指标"""
        return {name: stats.to_dict() for name, stats in self.metrics.items()}

    def get_means(self) -> Dict[str, float]:
        """获取所有均值"""
        return {name: stats.mean for name, stats in self.metrics.items()}

    def reset(self):
        """重置"""
        self.metrics.clear()


def compute_action_entropy(action_probs: np.ndarray) -> float:
    """
    计算动作熵

    Args:
        action_probs: 动作概率分布

    Returns:
        熵值
    """
    # 避免 log(0)
    probs = np.clip(action_probs, 1e-10, 1.0)
    probs = probs / probs.sum()
    return -np.sum(probs * np.log(probs))


def compute_value_accuracy(
    predicted_values: np.ndarray,
    actual_returns: np.ndarray,
    threshold: float = 0.1,
) -> float:
    """
    计算价值预测准确率

    Args:
        predicted_values: 预测价值
        actual_returns: 实际回报
        threshold: 误差阈值

    Returns:
        准确率
    """
    errors = np.abs(predicted_values - actual_returns)
    return np.mean(errors < threshold)


def compute_policy_kl(
    old_probs: np.ndarray,
    new_probs: np.ndarray,
) -> float:
    """
    计算策略 KL 散度

    Args:
        old_probs: 旧策略概率
        new_probs: 新策略概率

    Returns:
        KL 散度
    """
    old_probs = np.clip(old_probs, 1e-10, 1.0)
    new_probs = np.clip(new_probs, 1e-10, 1.0)
    return np.sum(old_probs * np.log(old_probs / new_probs))


def compute_explained_variance(
    predicted: np.ndarray,
    actual: np.ndarray,
) -> float:
    """
    计算解释方差

    Args:
        predicted: 预测值
        actual: 实际值

    Returns:
        解释方差比
    """
    var_actual = np.var(actual)
    if var_actual == 0:
        return 0.0
    return 1 - np.var(actual - predicted) / var_actual
