"""
ELO 评分系统

实现 ELO 评分算法
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import math
import json
from pathlib import Path


@dataclass
class PlayerRating:
    """玩家评分"""
    name: str
    rating: float = 1500.0
    games: int = 0
    wins: int = 0
    losses: int = 0
    peak_rating: float = 1500.0

    @property
    def win_rate(self) -> float:
        if self.games == 0:
            return 0.0
        return self.wins / self.games

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "rating": self.rating,
            "games": self.games,
            "wins": self.wins,
            "losses": self.losses,
            "peak_rating": self.peak_rating,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "PlayerRating":
        return cls(**d)


class EloSystem:
    """
    ELO 评分系统

    标准 ELO 实现
    """

    def __init__(
        self,
        k_factor: float = 32.0,
        initial_rating: float = 1500.0,
        floor_rating: float = 100.0,
    ):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.floor_rating = floor_rating
        self.players: Dict[str, PlayerRating] = {}

    def get_player(self, name: str) -> PlayerRating:
        """获取或创建玩家"""
        if name not in self.players:
            self.players[name] = PlayerRating(name=name, rating=self.initial_rating)
        return self.players[name]

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        计算期望得分

        Args:
            rating_a: 玩家 A 评分
            rating_b: 玩家 B 评分

        Returns:
            玩家 A 的期望得分
        """
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update_rating(
        self,
        player: PlayerRating,
        opponent_rating: float,
        score: float,
    ) -> float:
        """
        更新评分

        Args:
            player: 玩家
            opponent_rating: 对手评分
            score: 实际得分 (1=胜, 0.5=平, 0=负)

        Returns:
            新评分
        """
        expected = self.expected_score(player.rating, opponent_rating)
        new_rating = player.rating + self.k_factor * (score - expected)
        new_rating = max(new_rating, self.floor_rating)

        player.rating = new_rating
        player.peak_rating = max(player.peak_rating, new_rating)
        player.games += 1

        if score > 0.5:
            player.wins += 1
        elif score < 0.5:
            player.losses += 1

        return new_rating

    def record_match(
        self,
        winner_name: str,
        loser_name: str,
    ) -> Tuple[float, float]:
        """
        记录对局

        Args:
            winner_name: 胜者名称
            loser_name: 负者名称

        Returns:
            (胜者新评分, 负者新评分)
        """
        winner = self.get_player(winner_name)
        loser = self.get_player(loser_name)

        winner_old = winner.rating
        loser_old = loser.rating

        self.update_rating(winner, loser_old, 1.0)
        self.update_rating(loser, winner_old, 0.0)

        return winner.rating, loser.rating

    def record_draw(
        self,
        player1_name: str,
        player2_name: str,
    ) -> Tuple[float, float]:
        """记录平局"""
        player1 = self.get_player(player1_name)
        player2 = self.get_player(player2_name)

        p1_old = player1.rating
        p2_old = player2.rating

        self.update_rating(player1, p2_old, 0.5)
        self.update_rating(player2, p1_old, 0.5)

        return player1.rating, player2.rating

    def get_ranking(self) -> List[PlayerRating]:
        """获取排名"""
        return sorted(
            self.players.values(),
            key=lambda p: p.rating,
            reverse=True,
        )

    def save(self, path: str):
        """保存评分"""
        data = {
            "k_factor": self.k_factor,
            "initial_rating": self.initial_rating,
            "players": {
                name: player.to_dict()
                for name, player in self.players.items()
            },
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str):
        """加载评分"""
        data = json.loads(Path(path).read_text())
        self.k_factor = data.get("k_factor", self.k_factor)
        self.initial_rating = data.get("initial_rating", self.initial_rating)
        self.players = {
            name: PlayerRating.from_dict(p)
            for name, p in data.get("players", {}).items()
        }


class MultiPlayerElo(EloSystem):
    """
    多玩家 ELO 系统

    适用于斗地主等多玩家游戏
    """

    def record_game(
        self,
        landlord: str,
        farmers: Tuple[str, str],
        winner: str,  # "landlord" or "farmer"
    ):
        """
        记录斗地主对局

        Args:
            landlord: 地主名称
            farmers: 农民名称元组
            winner: 胜者 ("landlord" or "farmer")
        """
        landlord_player = self.get_player(landlord)
        farmer1 = self.get_player(farmers[0])
        farmer2 = self.get_player(farmers[1])

        # 农民平均评分
        farmer_avg_rating = (farmer1.rating + farmer2.rating) / 2

        if winner == "landlord":
            # 地主赢
            self.update_rating(landlord_player, farmer_avg_rating, 1.0)
            self.update_rating(farmer1, landlord_player.rating, 0.0)
            self.update_rating(farmer2, landlord_player.rating, 0.0)
        else:
            # 农民赢
            self.update_rating(landlord_player, farmer_avg_rating, 0.0)
            self.update_rating(farmer1, landlord_player.rating, 1.0)
            self.update_rating(farmer2, landlord_player.rating, 1.0)


class TrueSkillLite:
    """
    简化版 TrueSkill

    使用高斯分布估计技能
    """

    def __init__(
        self,
        mu: float = 25.0,
        sigma: float = 25.0 / 3.0,
        beta: float = 25.0 / 6.0,
        tau: float = 25.0 / 300.0,
    ):
        self.default_mu = mu
        self.default_sigma = sigma
        self.beta = beta
        self.tau = tau
        self.players: Dict[str, Tuple[float, float]] = {}  # (mu, sigma)

    def get_player(self, name: str) -> Tuple[float, float]:
        """获取玩家 (mu, sigma)"""
        if name not in self.players:
            self.players[name] = (self.default_mu, self.default_sigma)
        return self.players[name]

    def win_probability(
        self,
        team1: List[str],
        team2: List[str],
    ) -> float:
        """计算 team1 胜率"""
        mu1 = sum(self.get_player(p)[0] for p in team1)
        mu2 = sum(self.get_player(p)[0] for p in team2)

        sigma1_sq = sum(self.get_player(p)[1] ** 2 for p in team1)
        sigma2_sq = sum(self.get_player(p)[1] ** 2 for p in team2)

        denom = math.sqrt(
            2 * self.beta ** 2 + sigma1_sq + sigma2_sq
        )

        return self._phi((mu1 - mu2) / denom)

    def _phi(self, x: float) -> float:
        """标准正态 CDF"""
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def update(
        self,
        winners: List[str],
        losers: List[str],
    ):
        """更新评分"""
        # 简化更新: 只调整 mu
        for w in winners:
            mu, sigma = self.get_player(w)
            # 增加动态 tau
            sigma = math.sqrt(sigma ** 2 + self.tau ** 2)
            # 更新 mu
            mu += sigma ** 2 / (sigma ** 2 + self.beta ** 2)
            self.players[w] = (mu, sigma * 0.95)  # 降低不确定性

        for l in losers:
            mu, sigma = self.get_player(l)
            sigma = math.sqrt(sigma ** 2 + self.tau ** 2)
            mu -= sigma ** 2 / (sigma ** 2 + self.beta ** 2)
            self.players[l] = (mu, sigma * 0.95)

    def get_rating(self, name: str) -> float:
        """获取保守评分 (mu - 3*sigma)"""
        mu, sigma = self.get_player(name)
        return mu - 3 * sigma

    def get_ranking(self) -> List[Tuple[str, float, float, float]]:
        """获取排名 (name, rating, mu, sigma)"""
        return sorted(
            [
                (name, self.get_rating(name), mu, sigma)
                for name, (mu, sigma) in self.players.items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )


class GlickoSystem:
    """
    Glicko 评分系统

    带评分偏差的 ELO 变体
    """

    def __init__(
        self,
        initial_rating: float = 1500.0,
        initial_rd: float = 350.0,
        c: float = 34.6,  # RD 增长常数
        q: float = 0.00575646273,  # ln(10)/400
    ):
        self.initial_rating = initial_rating
        self.initial_rd = initial_rd
        self.c = c
        self.q = q
        self.players: Dict[str, Tuple[float, float]] = {}  # (rating, RD)

    def get_player(self, name: str) -> Tuple[float, float]:
        """获取玩家 (rating, RD)"""
        if name not in self.players:
            self.players[name] = (self.initial_rating, self.initial_rd)
        return self.players[name]

    def _g(self, rd: float) -> float:
        """g(RD) 函数"""
        return 1.0 / math.sqrt(1.0 + 3.0 * self.q ** 2 * rd ** 2 / (math.pi ** 2))

    def _e(self, r: float, rj: float, rdj: float) -> float:
        """期望得分"""
        return 1.0 / (1.0 + 10.0 ** (-self._g(rdj) * (r - rj) / 400.0))

    def update(
        self,
        player_name: str,
        opponents: List[Tuple[str, float]],  # [(opponent_name, score), ...]
    ):
        """
        更新评分

        Args:
            player_name: 玩家名称
            opponents: 对手列表和得分
        """
        r, rd = self.get_player(player_name)

        # 计算 d^2
        d_sq_inv = 0.0
        for opp_name, score in opponents:
            rj, rdj = self.get_player(opp_name)
            g_rdj = self._g(rdj)
            e = self._e(r, rj, rdj)
            d_sq_inv += g_rdj ** 2 * e * (1 - e)

        d_sq_inv *= self.q ** 2
        d_sq = 1.0 / d_sq_inv if d_sq_inv > 0 else float('inf')

        # 更新评分
        sum_term = 0.0
        for opp_name, score in opponents:
            rj, rdj = self.get_player(opp_name)
            g_rdj = self._g(rdj)
            e = self._e(r, rj, rdj)
            sum_term += g_rdj * (score - e)

        new_rd = 1.0 / math.sqrt(1.0 / rd ** 2 + 1.0 / d_sq)
        new_r = r + self.q * new_rd ** 2 * sum_term

        self.players[player_name] = (new_r, new_rd)

    def get_ranking(self) -> List[Tuple[str, float, float]]:
        """获取排名 (name, rating, RD)"""
        return sorted(
            [(name, r, rd) for name, (r, rd) in self.players.items()],
            key=lambda x: x[1],
            reverse=True,
        )
