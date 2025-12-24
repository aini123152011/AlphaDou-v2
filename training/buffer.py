"""
经验缓冲区

存储和采样训练数据
"""
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import random
import numpy as np
import torch


@dataclass
class Transition:
    """
    单步转移

    Attributes:
        obs: 观测
        action: 动作
        reward: 奖励
        next_obs: 下一观测
        done: 是否终止
        info: 额外信息
    """
    obs: Dict[str, np.ndarray]
    action: int
    reward: float
    next_obs: Optional[Dict[str, np.ndarray]]
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)

    # 可选的额外信息
    log_prob: Optional[float] = None
    value: Optional[float] = None
    advantage: Optional[float] = None
    returns: Optional[float] = None


@dataclass
class Trajectory:
    """
    完整轨迹

    存储一局游戏的所有转移
    """
    transitions: List[Transition] = field(default_factory=list)
    player: Optional[str] = None
    winner: Optional[str] = None
    total_reward: float = 0.0

    def __len__(self) -> int:
        return len(self.transitions)

    def add(self, transition: Transition):
        """添加转移"""
        self.transitions.append(transition)
        self.total_reward += transition.reward

    def compute_returns(self, gamma: float = 0.99) -> None:
        """计算折扣回报"""
        returns = 0.0
        for t in reversed(self.transitions):
            returns = t.reward + gamma * returns * (1 - t.done)
            t.returns = returns

    def compute_advantages(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
        """计算 GAE 优势"""
        advantages = []
        gae = 0.0

        for i in reversed(range(len(self.transitions))):
            t = self.transitions[i]

            if i == len(self.transitions) - 1:
                next_value = 0.0
            else:
                next_value = self.transitions[i + 1].value or 0.0

            delta = t.reward + gamma * next_value * (1 - t.done) - (t.value or 0.0)
            gae = delta + gamma * gae_lambda * (1 - t.done) * gae
            t.advantage = gae

    def to_batch(self) -> Dict[str, torch.Tensor]:
        """转换为训练批次"""
        batch = {
            "hand": [],
            "played_cards": [],
            "history": [],
            "last_action": [],
            "position": [],
            "bid_info": [],
            "cards_left": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }

        for t in self.transitions:
            for key in ["hand", "played_cards", "history", "last_action",
                       "position", "bid_info", "cards_left"]:
                if key in t.obs:
                    batch[key].append(torch.from_numpy(t.obs[key]))

            batch["actions"].append(t.action)
            batch["rewards"].append(t.reward)
            batch["dones"].append(t.done)

        # 转换为张量
        result = {}
        for key, values in batch.items():
            if values:
                if key in ["actions"]:
                    result[key] = torch.tensor(values, dtype=torch.long)
                elif key in ["dones"]:
                    result[key] = torch.tensor(values, dtype=torch.bool)
                elif isinstance(values[0], torch.Tensor):
                    result[key] = torch.stack(values)
                else:
                    result[key] = torch.tensor(values, dtype=torch.float32)

        # 添加额外信息
        if self.transitions[0].log_prob is not None:
            result["log_probs"] = torch.tensor(
                [t.log_prob for t in self.transitions], dtype=torch.float32
            )
        if self.transitions[0].value is not None:
            result["values"] = torch.tensor(
                [t.value for t in self.transitions], dtype=torch.float32
            )
        if self.transitions[0].advantage is not None:
            result["advantages"] = torch.tensor(
                [t.advantage for t in self.transitions], dtype=torch.float32
            )
        if self.transitions[0].returns is not None:
            result["returns"] = torch.tensor(
                [t.returns for t in self.transitions], dtype=torch.float32
            )

        return result


class ReplayBuffer:
    """
    经验回放缓冲区

    支持随机采样的 off-policy 缓冲区
    """

    def __init__(
        self,
        capacity: int = 100000,
        batch_size: int = 256,
    ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, transition: Transition):
        """添加单个转移"""
        self.buffer.append(transition)

    def add_trajectory(self, trajectory: Trajectory):
        """添加完整轨迹"""
        for t in trajectory.transitions:
            self.buffer.append(t)

    def sample(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """随机采样批次"""
        batch_size = batch_size or self.batch_size
        batch_size = min(batch_size, len(self.buffer))

        transitions = random.sample(list(self.buffer), batch_size)

        batch = {
            "hand": [],
            "played_cards": [],
            "history": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }

        for t in transitions:
            for key in ["hand", "played_cards", "history"]:
                if key in t.obs:
                    batch[key].append(torch.from_numpy(t.obs[key]))
            batch["actions"].append(t.action)
            batch["rewards"].append(t.reward)
            batch["dones"].append(t.done)

        result = {}
        for key, values in batch.items():
            if values:
                if key == "actions":
                    result[key] = torch.tensor(values, dtype=torch.long)
                elif key == "dones":
                    result[key] = torch.tensor(values, dtype=torch.bool)
                elif isinstance(values[0], torch.Tensor):
                    result[key] = torch.stack(values)
                else:
                    result[key] = torch.tensor(values, dtype=torch.float32)

        return result

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self) -> bool:
        """是否准备好采样"""
        return len(self.buffer) >= self.batch_size


class RolloutBuffer:
    """
    Rollout 缓冲区

    用于 on-policy 算法 (PPO 等)
    """

    def __init__(
        self,
        buffer_size: int,
        obs_shape: Dict[str, Tuple[int, ...]],
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.pos = 0
        self.full = False

        # 初始化存储
        self.observations = {
            key: np.zeros((buffer_size,) + shape, dtype=np.float32)
            for key, shape in obs_shape.items()
        }
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ):
        """添加一步经验"""
        for key, value_arr in obs.items():
            if key in self.observations:
                self.observations[key][self.pos] = value_arr

        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(self, last_value: float = 0.0):
        """计算回报和优势"""
        last_gae = 0.0

        for step in reversed(range(self.pos)):
            if step == self.pos - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[step]
            else:
                next_value = self.values[step + 1]
                next_non_terminal = 1.0 - self.dones[step]

            delta = (
                self.rewards[step]
                + self.gamma * next_value * next_non_terminal
                - self.values[step]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae
            self.returns[step] = last_gae + self.values[step]

    def get(self) -> Dict[str, torch.Tensor]:
        """获取所有数据"""
        indices = np.arange(self.pos if not self.full else self.buffer_size)

        data = {
            key: torch.from_numpy(arr[indices])
            for key, arr in self.observations.items()
        }
        data["actions"] = torch.from_numpy(self.actions[indices])
        data["values"] = torch.from_numpy(self.values[indices])
        data["log_probs"] = torch.from_numpy(self.log_probs[indices])
        data["advantages"] = torch.from_numpy(self.advantages[indices])
        data["returns"] = torch.from_numpy(self.returns[indices])

        return data

    def reset(self):
        """重置缓冲区"""
        self.pos = 0
        self.full = False


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    优先级经验回放

    基于 TD-error 的优先级采样
    """

    def __init__(
        self,
        capacity: int = 100000,
        batch_size: int = 256,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
    ):
        super().__init__(capacity, batch_size)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

    def add(self, transition: Transition):
        """添加带最大优先级"""
        super().add(transition)
        idx = (len(self.buffer) - 1) % self.capacity
        self.priorities[idx] = self.max_priority

    def sample(self, batch_size: Optional[int] = None) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """优先级采样"""
        batch_size = batch_size or self.batch_size
        batch_size = min(batch_size, len(self.buffer))

        # 计算采样概率
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # 采样
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)

        # 计算重要性权重
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        # 更新 beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        # 获取数据
        transitions = [self.buffer[i] for i in indices]
        batch = self._transitions_to_batch(transitions)

        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def _transitions_to_batch(self, transitions: List[Transition]) -> Dict[str, torch.Tensor]:
        """转换为批次"""
        batch = {"hand": [], "played_cards": [], "history": [], "actions": [], "rewards": []}
        for t in transitions:
            for key in ["hand", "played_cards", "history"]:
                if key in t.obs:
                    batch[key].append(torch.from_numpy(t.obs[key]))
            batch["actions"].append(t.action)
            batch["rewards"].append(t.reward)

        result = {}
        for key, values in batch.items():
            if values:
                if key == "actions":
                    result[key] = torch.tensor(values, dtype=torch.long)
                elif isinstance(values[0], torch.Tensor):
                    result[key] = torch.stack(values)
                else:
                    result[key] = torch.tensor(values, dtype=torch.float32)
        return result
