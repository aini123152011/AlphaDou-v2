"""
奖励函数

支持多种奖励设计:
- 终局奖励 (sparse)
- 过程奖励 (shaped)
- ADP 奖励 (Average Difference Position)
"""
from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum

from core.state import GameState, Phase, Role, PLAY_ORDER
from core.rules import RuleEngine


class RewardType(Enum):
    """奖励类型"""
    SPARSE = "sparse"      # 仅终局奖励
    SHAPED = "shaped"      # 过程奖励
    ADP = "adp"            # DouZero 使用的 ADP


@dataclass
class RewardConfig:
    """奖励配置"""
    reward_type: RewardType = RewardType.ADP
    win_reward: float = 1.0
    lose_reward: float = -1.0
    draw_reward: float = 0.0
    bomb_bonus: float = 0.0      # 炸弹奖励
    spring_bonus: float = 0.0   # 春天奖励
    card_penalty: float = 0.0   # 剩余牌惩罚


class RewardCalculator:
    """
    奖励计算器

    根据配置计算不同类型的奖励
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()

    def compute(
        self,
        state: GameState,
        prev_state: Optional[GameState] = None,
        player: Optional[Role] = None,
    ) -> float:
        """
        计算奖励

        Args:
            state: 当前状态
            prev_state: 前一状态 (用于 shaped 奖励)
            player: 计算奖励的玩家视角

        Returns:
            奖励值
        """
        if player is None:
            player = state.current_player

        if self.config.reward_type == RewardType.SPARSE:
            return self._sparse_reward(state, player)
        elif self.config.reward_type == RewardType.SHAPED:
            return self._shaped_reward(state, prev_state, player)
        elif self.config.reward_type == RewardType.ADP:
            return self._adp_reward(state, player)
        else:
            return 0.0

    def _sparse_reward(self, state: GameState, player: Role) -> float:
        """
        稀疏奖励：仅在游戏结束时给予

        Returns:
            胜利: +1, 失败: -1, 其他: 0
        """
        if state.phase != Phase.FINISHED:
            return 0.0

        if state.winner == "draw":
            return self.config.draw_reward

        is_landlord = player == Role.LANDLORD
        landlord_wins = state.winner == "landlord"

        if is_landlord == landlord_wins:
            return self.config.win_reward
        else:
            return self.config.lose_reward

    def _shaped_reward(
        self,
        state: GameState,
        prev_state: Optional[GameState],
        player: Role,
    ) -> float:
        """
        过程奖励：包含中间步骤的塑形奖励

        奖励组成:
        1. 终局奖励
        2. 出牌数量变化
        3. 炸弹奖励
        """
        reward = 0.0

        # 终局奖励
        if state.phase == Phase.FINISHED:
            reward += self._sparse_reward(state, player)

            # 春天加成
            if state.is_spring() and self.config.spring_bonus > 0:
                is_landlord = player == Role.LANDLORD
                if (state.winner == "landlord") == is_landlord:
                    reward += self.config.spring_bonus
                else:
                    reward -= self.config.spring_bonus

            return reward

        # 过程奖励
        if prev_state is not None and state.phase == Phase.PLAYING:
            # 出牌奖励 (减少手牌)
            prev_cards = len(prev_state.get_hand(player))
            curr_cards = len(state.get_hand(player))
            cards_played = prev_cards - curr_cards
            if cards_played > 0:
                reward += cards_played * 0.01  # 小奖励

            # 炸弹奖励
            if state.bombs_count > prev_state.bombs_count:
                reward += self.config.bomb_bonus

        return reward

    def _adp_reward(self, state: GameState, player: Role) -> float:
        """
        ADP (Average Difference Position) 奖励

        DouZero 论文使用的奖励设计:
        - 游戏结束时，根据最终得分计算
        - 考虑炸弹翻倍和春天

        Returns:
            基于得分的奖励
        """
        if state.phase != Phase.FINISHED:
            return 0.0

        if state.winner == "draw":
            return self.config.draw_reward

        # 计算得分
        scores = RuleEngine.calculate_score(
            winner=state.winner,
            bid_count=state.bid_multiplier,
            bombs_count=state.bombs_count,
            is_spring=state.is_spring(),
        )

        # 返回该玩家的得分
        player_key = player.value
        return float(scores.get(player_key, 0))


class MultiAgentReward:
    """
    多智能体奖励计算

    为所有玩家同时计算奖励
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.calculator = RewardCalculator(config)

    def compute_all(
        self,
        state: GameState,
        prev_state: Optional[GameState] = None,
    ) -> Dict[str, float]:
        """
        计算所有玩家的奖励

        Args:
            state: 当前状态
            prev_state: 前一状态

        Returns:
            {player_name: reward} 字典
        """
        rewards = {}

        if state.phase == Phase.BIDDING:
            # 叫牌阶段暂无奖励
            for role in [Role.FIRST, Role.SECOND, Role.THIRD]:
                rewards[role.value] = 0.0
        else:
            for role in PLAY_ORDER:
                rewards[role.value] = self.calculator.compute(
                    state, prev_state, role
                )

        return rewards


def create_reward_calculator(
    reward_type: str = "adp",
    **kwargs
) -> RewardCalculator:
    """
    工厂函数：创建奖励计算器

    Args:
        reward_type: 奖励类型 ("sparse", "shaped", "adp")
        **kwargs: 其他配置参数

    Returns:
        RewardCalculator 实例
    """
    config = RewardConfig(
        reward_type=RewardType(reward_type),
        **kwargs
    )
    return RewardCalculator(config)
