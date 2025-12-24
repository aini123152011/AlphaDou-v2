"""
评估器

评估模型性能
"""
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """评估结果"""
    win_rate: float
    avg_reward: float
    avg_length: float
    games_played: int
    landlord_win_rate: float = 0.0
    farmer_win_rate: float = 0.0
    spring_rate: float = 0.0
    bomb_rate: float = 0.0
    extra_stats: Dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"EvalResult(win_rate={self.win_rate:.2%}, "
            f"avg_reward={self.avg_reward:.2f}, "
            f"games={self.games_played})"
        )


class Agent:
    """智能体基类"""

    def __init__(self, name: str = "agent"):
        self.name = name

    def act(self, obs: Dict[str, Any], legal_actions: List) -> Any:
        """选择动作"""
        raise NotImplementedError

    def reset(self):
        """重置状态"""
        pass


class RandomAgent(Agent):
    """随机智能体"""

    def __init__(self, name: str = "random"):
        super().__init__(name)

    def act(self, obs: Dict[str, Any], legal_actions: List) -> Any:
        if not legal_actions:
            return 0
        idx = np.random.randint(len(legal_actions))
        return legal_actions[idx]


class RuleBasedAgent(Agent):
    """规则智能体"""

    def __init__(self, name: str = "rule"):
        super().__init__(name)

    def act(self, obs: Dict[str, Any], legal_actions: List) -> Any:
        if not legal_actions:
            return 0

        # 简单规则: 尽量出牌，不 pass
        # pass 动作通常是 0
        non_pass = [a for a in legal_actions if a != 0]
        if non_pass:
            return non_pass[0]
        return legal_actions[0]


class ModelAgent(Agent):
    """模型智能体"""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        deterministic: bool = True,
        name: str = "model",
    ):
        super().__init__(name)
        self.model = model
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.model.to(self.device)
        self.model.eval()

    def act(self, obs: Dict[str, Any], legal_actions: List) -> Any:
        with torch.no_grad():
            # 转换观测
            obs_tensor = {}
            for k, v in obs.items():
                if isinstance(v, np.ndarray):
                    obs_tensor[k] = torch.from_numpy(v).unsqueeze(0).to(self.device)

            # 获取动作
            output = self.model(obs_tensor)

            if self.deterministic:
                # 贪婪选择
                logits = output.policy_logits[0]
                # 屏蔽非法动作
                mask = torch.full_like(logits, float('-inf'))
                for a in legal_actions:
                    mask[a] = 0
                logits = logits + mask
                action = logits.argmax().item()
            else:
                # 采样
                logits = output.policy_logits[0]
                mask = torch.full_like(logits, float('-inf'))
                for a in legal_actions:
                    mask[a] = 0
                logits = logits + mask
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()

            return action


class Evaluator:
    """
    评估器

    评估模型在环境中的表现
    """

    def __init__(
        self,
        env_fn: Callable,
        device: str = "cpu",
    ):
        self.env_fn = env_fn
        self.device = device

    def evaluate(
        self,
        agent: Agent,
        n_games: int = 100,
        opponents: Optional[List[Agent]] = None,
        verbose: bool = False,
    ) -> EvalResult:
        """
        评估智能体

        Args:
            agent: 待评估智能体
            n_games: 游戏数量
            opponents: 对手列表
            verbose: 是否输出详情

        Returns:
            评估结果
        """
        if opponents is None:
            opponents = [RandomAgent("opp1"), RandomAgent("opp2")]

        env = self.env_fn()

        # 角色到索引的映射
        role_to_idx = {"landlord": 0, "farmer_down": 1, "farmer_up": 2}

        wins = 0
        total_reward = 0.0
        total_length = 0
        landlord_wins = 0
        landlord_games = 0
        farmer_wins = 0
        farmer_games = 0
        springs = 0
        bombs = 0

        for game_idx in range(n_games):
            obs, info = env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0

            # 确定智能体位置 (轮流)
            agent_position = game_idx % 3
            agent_roles = ["landlord", "farmer_down", "farmer_up"]
            agent_role_name = agent_roles[agent_position]

            while not done:
                current_player = info.get("current_player", "landlord")
                current_idx = role_to_idx.get(current_player, 0)
                legal_actions = env.get_legal_actions()

                if current_idx == agent_position:
                    action = agent.act(obs, legal_actions)
                else:
                    opp_idx = (current_idx - agent_position - 1) % 2
                    action = opponents[opp_idx].act(obs, legal_actions)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if current_idx == agent_position:
                    episode_reward += reward
                episode_length += 1

            # 统计
            winner = info.get("winner")

            if agent_role_name == "landlord":
                landlord_games += 1
                if winner == "landlord":
                    wins += 1
                    landlord_wins += 1
            else:
                farmer_games += 1
                if winner == "farmer":
                    wins += 1
                    farmer_wins += 1

            if info.get("spring", False):
                springs += 1
            bombs += info.get("bombs", 0)

            total_reward += episode_reward
            total_length += episode_length

            if verbose and (game_idx + 1) % 10 == 0:
                logger.info(f"Game {game_idx + 1}/{n_games}, Win rate: {wins/(game_idx+1):.2%}")

        return EvalResult(
            win_rate=wins / n_games if n_games > 0 else 0.0,
            avg_reward=total_reward / n_games if n_games > 0 else 0.0,
            avg_length=total_length / n_games if n_games > 0 else 0.0,
            games_played=n_games,
            landlord_win_rate=landlord_wins / landlord_games if landlord_games > 0 else 0.0,
            farmer_win_rate=farmer_wins / farmer_games if farmer_games > 0 else 0.0,
            spring_rate=springs / n_games if n_games > 0 else 0.0,
            bomb_rate=bombs / n_games if n_games > 0 else 0.0,
        )

    def compare(
        self,
        agent1: Agent,
        agent2: Agent,
        n_games: int = 100,
        third_agent: Optional[Agent] = None,
    ) -> Dict[str, float]:
        """
        对比两个智能体

        Args:
            agent1: 智能体1
            agent2: 智能体2
            n_games: 游戏数量
            third_agent: 第三个智能体 (农民2)

        Returns:
            对比结果
        """
        if third_agent is None:
            third_agent = RandomAgent("third")

        env = self.env_fn()

        agent1_wins = 0
        agent2_wins = 0

        for game_idx in range(n_games):
            obs, info = env.reset()
            done = False

            # agent1 做地主, agent2 和 third_agent 做农民
            agents = [agent1, agent2, third_agent]
            role_to_idx = {"landlord": 0, "farmer_down": 1, "farmer_up": 2}

            while not done:
                current_player = info.get("current_player", "landlord")
                current_idx = role_to_idx.get(current_player, 0)
                legal_actions = env.get_legal_actions()

                action = agents[current_idx].act(obs, legal_actions)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            winner = info.get("winner")
            if winner == "landlord":
                agent1_wins += 1
            else:
                agent2_wins += 1

        return {
            "agent1_wins": agent1_wins,
            "agent2_wins": agent2_wins,
            "agent1_win_rate": agent1_wins / n_games,
            "agent2_win_rate": agent2_wins / n_games,
        }


class MultiAgentEvaluator:
    """
    多智能体评估器

    评估多智能体场景
    """

    def __init__(self, env_fn: Callable):
        self.env_fn = env_fn

    def evaluate_all(
        self,
        agents: List[Agent],
        n_games: int = 100,
    ) -> Dict[str, EvalResult]:
        """
        评估所有智能体

        Args:
            agents: 智能体列表 (3个)
            n_games: 游戏数量

        Returns:
            每个智能体的评估结果
        """
        assert len(agents) == 3, "Need exactly 3 agents"

        env = self.env_fn()
        stats = {agent.name: defaultdict(float) for agent in agents}

        for game_idx in range(n_games):
            obs, info = env.reset()
            done = False

            while not done:
                current_player = info.get("current_player", 0)
                legal_actions = env.get_legal_actions()

                action = agents[current_player].act(obs, legal_actions)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            # 记录结果
            winner = info.get("winner")
            for i, agent in enumerate(agents):
                role = info.get("roles", {}).get(i, "unknown")
                won = (
                    (winner == "landlord" and role == "landlord") or
                    (winner == "farmer" and role in ["farmer_down", "farmer_up"])
                )
                stats[agent.name]["games"] += 1
                stats[agent.name]["wins"] += int(won)
                stats[agent.name]["total_reward"] += reward if i == 0 else 0

        results = {}
        for agent in agents:
            s = stats[agent.name]
            results[agent.name] = EvalResult(
                win_rate=s["wins"] / s["games"] if s["games"] > 0 else 0.0,
                avg_reward=s["total_reward"] / s["games"] if s["games"] > 0 else 0.0,
                avg_length=0.0,
                games_played=int(s["games"]),
            )

        return results
