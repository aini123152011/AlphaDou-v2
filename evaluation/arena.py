"""
对战竞技场

组织多智能体对战
"""
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from itertools import permutations
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .evaluator import Agent, EvalResult

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """对局结果"""
    agents: Tuple[str, str, str]  # (landlord, farmer_down, farmer_up)
    winner: str  # "landlord" or "farmer"
    landlord_agent: str
    farmer_agents: Tuple[str, str]
    length: int
    bombs: int
    spring: bool


@dataclass
class TournamentResult:
    """锦标赛结果"""
    standings: Dict[str, Dict[str, float]]
    total_games: int
    matches: List[MatchResult]

    def get_ranking(self) -> List[Tuple[str, float]]:
        """获取排名"""
        return sorted(
            [(name, stats["win_rate"]) for name, stats in self.standings.items()],
            key=lambda x: x[1],
            reverse=True,
        )

    def __repr__(self) -> str:
        ranking = self.get_ranking()
        lines = [f"Tournament Results ({self.total_games} games):"]
        for i, (name, win_rate) in enumerate(ranking):
            lines.append(f"  {i+1}. {name}: {win_rate:.2%}")
        return "\n".join(lines)


class Arena:
    """
    对战竞技场

    组织智能体之间的对战
    """

    def __init__(self, env_fn: Callable):
        self.env_fn = env_fn

    def play_match(
        self,
        agents: List[Agent],
        n_games: int = 1,
    ) -> List[MatchResult]:
        """
        进行对局

        Args:
            agents: 3个智能体
            n_games: 对局数

        Returns:
            对局结果列表
        """
        assert len(agents) == 3

        env = self.env_fn()
        results = []
        role_to_idx = {"landlord": 0, "farmer_down": 1, "farmer_up": 2}

        for _ in range(n_games):
            obs, info = env.reset()
            done = False
            length = 0

            while not done:
                current_player = info.get("current_player", "landlord")
                current_idx = role_to_idx.get(current_player, 0)
                legal_actions = env.get_legal_actions()

                action = agents[current_idx].act(obs, legal_actions)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                length += 1

            winner = info.get("winner", "unknown")
            result = MatchResult(
                agents=(agents[0].name, agents[1].name, agents[2].name),
                winner=winner,
                landlord_agent=agents[0].name,
                farmer_agents=(agents[1].name, agents[2].name),
                length=length,
                bombs=info.get("bombs", 0),
                spring=info.get("spring", False),
            )
            results.append(result)

        return results

    def round_robin(
        self,
        agents: List[Agent],
        games_per_match: int = 10,
    ) -> TournamentResult:
        """
        循环赛

        每个智能体组合都对战

        Args:
            agents: 智能体列表
            games_per_match: 每场比赛的对局数

        Returns:
            锦标赛结果
        """
        standings = {agent.name: defaultdict(float) for agent in agents}
        all_matches = []

        # 生成所有位置组合
        for perm in permutations(range(len(agents)), 3):
            match_agents = [agents[i] for i in perm]

            results = self.play_match(match_agents, games_per_match)
            all_matches.extend(results)

            for result in results:
                # 地主
                landlord = result.landlord_agent
                standings[landlord]["landlord_games"] += 1
                if result.winner == "landlord":
                    standings[landlord]["landlord_wins"] += 1
                    standings[landlord]["wins"] += 1

                # 农民
                for farmer in result.farmer_agents:
                    standings[farmer]["farmer_games"] += 1
                    if result.winner == "farmer":
                        standings[farmer]["farmer_wins"] += 1
                        standings[farmer]["wins"] += 1

                # 总场次
                for name in [landlord] + list(result.farmer_agents):
                    standings[name]["games"] += 1

        # 计算胜率
        for name, stats in standings.items():
            if stats["games"] > 0:
                stats["win_rate"] = stats["wins"] / stats["games"]
            if stats["landlord_games"] > 0:
                stats["landlord_win_rate"] = stats["landlord_wins"] / stats["landlord_games"]
            if stats["farmer_games"] > 0:
                stats["farmer_win_rate"] = stats["farmer_wins"] / stats["farmer_games"]

        return TournamentResult(
            standings=dict(standings),
            total_games=len(all_matches),
            matches=all_matches,
        )

    def tournament(
        self,
        agents: List[Agent],
        n_rounds: int = 100,
    ) -> TournamentResult:
        """
        锦标赛

        随机配对进行多轮比赛

        Args:
            agents: 智能体列表
            n_rounds: 轮数

        Returns:
            锦标赛结果
        """
        standings = {agent.name: defaultdict(float) for agent in agents}
        all_matches = []

        for round_idx in range(n_rounds):
            # 随机选择3个智能体
            if len(agents) >= 3:
                selected = np.random.choice(len(agents), 3, replace=False)
                match_agents = [agents[i] for i in selected]
            else:
                match_agents = agents[:3]

            results = self.play_match(match_agents, n_games=1)
            all_matches.extend(results)

            for result in results:
                landlord = result.landlord_agent
                standings[landlord]["landlord_games"] += 1
                if result.winner == "landlord":
                    standings[landlord]["landlord_wins"] += 1
                    standings[landlord]["wins"] += 1

                for farmer in result.farmer_agents:
                    standings[farmer]["farmer_games"] += 1
                    if result.winner == "farmer":
                        standings[farmer]["farmer_wins"] += 1
                        standings[farmer]["wins"] += 1

                for name in [landlord] + list(result.farmer_agents):
                    standings[name]["games"] += 1

        # 计算胜率
        for name, stats in standings.items():
            if stats["games"] > 0:
                stats["win_rate"] = stats["wins"] / stats["games"]

        return TournamentResult(
            standings=dict(standings),
            total_games=len(all_matches),
            matches=all_matches,
        )


class ParallelArena(Arena):
    """
    并行对战竞技场

    使用多线程加速对战
    """

    def __init__(self, env_fn: Callable, n_workers: int = 4):
        super().__init__(env_fn)
        self.n_workers = n_workers

    def _play_single_match(
        self,
        agents: List[Agent],
    ) -> MatchResult:
        """单场对局"""
        env = self.env_fn()
        obs, info = env.reset()
        done = False
        length = 0
        role_to_idx = {"landlord": 0, "farmer_down": 1, "farmer_up": 2}

        while not done:
            current_player = info.get("current_player", "landlord")
            current_idx = role_to_idx.get(current_player, 0)
            legal_actions = env.get_legal_actions()

            action = agents[current_idx].act(obs, legal_actions)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            length += 1

        return MatchResult(
            agents=(agents[0].name, agents[1].name, agents[2].name),
            winner=info.get("winner", "unknown"),
            landlord_agent=agents[0].name,
            farmer_agents=(agents[1].name, agents[2].name),
            length=length,
            bombs=info.get("bombs", 0),
            spring=info.get("spring", False),
        )

    def play_match(
        self,
        agents: List[Agent],
        n_games: int = 1,
    ) -> List[MatchResult]:
        """并行对局"""
        if n_games <= self.n_workers:
            return super().play_match(agents, n_games)

        results = []
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(self._play_single_match, agents)
                for _ in range(n_games)
            ]
            for future in as_completed(futures):
                results.append(future.result())

        return results


class LeaderBoard:
    """
    排行榜

    追踪智能体历史表现
    """

    def __init__(self):
        self.records: Dict[str, Dict[str, float]] = {}
        self.history: List[Dict] = []

    def update(self, tournament_result: TournamentResult):
        """更新排行榜"""
        for name, stats in tournament_result.standings.items():
            if name not in self.records:
                self.records[name] = defaultdict(float)

            self.records[name]["total_games"] += stats.get("games", 0)
            self.records[name]["total_wins"] += stats.get("wins", 0)

            if self.records[name]["total_games"] > 0:
                self.records[name]["overall_win_rate"] = (
                    self.records[name]["total_wins"] /
                    self.records[name]["total_games"]
                )

        self.history.append({
            "standings": tournament_result.standings,
            "total_games": tournament_result.total_games,
        })

    def get_ranking(self) -> List[Tuple[str, float, int]]:
        """获取排名 (名称, 胜率, 总场次)"""
        return sorted(
            [
                (name, stats.get("overall_win_rate", 0.0), int(stats.get("total_games", 0)))
                for name, stats in self.records.items()
            ],
            key=lambda x: (x[1], x[2]),
            reverse=True,
        )

    def __repr__(self) -> str:
        ranking = self.get_ranking()
        lines = ["Leaderboard:"]
        for i, (name, win_rate, games) in enumerate(ranking):
            lines.append(f"  {i+1}. {name}: {win_rate:.2%} ({games} games)")
        return "\n".join(lines)
