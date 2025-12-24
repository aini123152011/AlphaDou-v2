"""
游戏状态定义

使用不可变数据结构，支持:
- 哈希 (用于 MCTS)
- 线程安全
- 易于序列化
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List, FrozenSet
from enum import Enum
import random
import copy

from .cards import FULL_DECK, Card
from .actions import Action, ActionType, ActionGenerator
from .rules import RuleEngine


class Phase(Enum):
    """游戏阶段"""
    BIDDING = "bidding"    # 叫牌阶段
    PLAYING = "playing"    # 出牌阶段
    FINISHED = "finished"  # 游戏结束


class Role(Enum):
    """玩家角色"""
    # 叫牌阶段位置
    FIRST = "first"
    SECOND = "second"
    THIRD = "third"
    # 出牌阶段位置
    LANDLORD = "landlord"
    LANDLORD_DOWN = "landlord_down"
    LANDLORD_UP = "landlord_up"


# 叫牌顺序
BID_ORDER: Tuple[Role, ...] = (Role.FIRST, Role.SECOND, Role.THIRD)

# 出牌顺序
PLAY_ORDER: Tuple[Role, ...] = (Role.LANDLORD, Role.LANDLORD_DOWN, Role.LANDLORD_UP)

# 叫牌位置到出牌位置的映射 (根据谁成为地主)
def get_play_roles(landlord_bid_pos: Role) -> Dict[Role, Role]:
    """
    根据地主的叫牌位置，返回叫牌位置到出牌位置的映射

    Args:
        landlord_bid_pos: 成为地主的叫牌位置

    Returns:
        叫牌位置 -> 出牌位置的映射
    """
    if landlord_bid_pos == Role.FIRST:
        return {
            Role.FIRST: Role.LANDLORD,
            Role.SECOND: Role.LANDLORD_DOWN,
            Role.THIRD: Role.LANDLORD_UP,
        }
    elif landlord_bid_pos == Role.SECOND:
        return {
            Role.FIRST: Role.LANDLORD_UP,
            Role.SECOND: Role.LANDLORD,
            Role.THIRD: Role.LANDLORD_DOWN,
        }
    else:  # THIRD
        return {
            Role.FIRST: Role.LANDLORD_DOWN,
            Role.SECOND: Role.LANDLORD_UP,
            Role.THIRD: Role.LANDLORD,
        }


@dataclass(frozen=True)
class GameState:
    """
    不可变游戏状态

    使用 frozen=True 保证:
    - 可哈希 (用于 MCTS 状态缓存)
    - 线程安全
    - 易于序列化

    Attributes:
        hands: 各玩家手牌 (出牌阶段使用 LANDLORD/DOWN/UP，叫牌阶段使用 FIRST/SECOND/THIRD)
        three_cards: 底牌
        phase: 游戏阶段
        current_player: 当前行动玩家
        landlord: 地主角色 (叫牌结束后确定)
        landlord_bid_position: 地主的叫牌位置 (FIRST/SECOND/THIRD)
        bid_info: 叫牌信息 (first_bid, second_bid, third_bid)，-1 表示未叫
        bid_history: 叫牌历史 ((position, bid_value), ...)
        play_history: 出牌历史 ((player, cards), ...)
        last_action: 最近一次有效出牌
        last_player: 最近出牌的玩家
        bombs_count: 已打出的炸弹数
        step_count: 当前步数
        winner: 赢家
    """
    # 手牌 (frozen 保证不可变)
    hands: Tuple[Tuple[str, Tuple[int, ...]], ...]

    # 底牌
    three_cards: Tuple[int, ...]

    # 游戏阶段
    phase: Phase

    # 当前行动玩家
    current_player: Role

    # 地主角色
    landlord: Optional[Role] = None

    # 地主的叫牌位置
    landlord_bid_position: Optional[Role] = None

    # 叫牌信息
    bid_info: Tuple[int, int, int] = (-1, -1, -1)

    # 叫牌历史
    bid_history: Tuple[Tuple[Role, int], ...] = ()

    # 出牌历史
    play_history: Tuple[Tuple[Role, Tuple[int, ...]], ...] = ()

    # 最近一次有效出牌 (非 PASS)
    last_action: Optional[Action] = None

    # 最近出牌的玩家
    last_player: Optional[Role] = None

    # 炸弹数
    bombs_count: int = 0

    # 步数
    step_count: int = 0

    # 赢家
    winner: Optional[str] = None

    def get_hand(self, role: Role) -> Tuple[int, ...]:
        """获取指定玩家的手牌"""
        for r, cards in self.hands:
            if r == role.value:
                return cards
        return ()

    def get_hands_dict(self) -> Dict[str, List[int]]:
        """获取手牌字典"""
        return {r: list(cards) for r, cards in self.hands}

    @classmethod
    def initial(cls, seed: Optional[int] = None) -> 'GameState':
        """
        创建初始游戏状态

        Args:
            seed: 随机种子

        Returns:
            初始状态 (叫牌阶段)
        """
        if seed is not None:
            random.seed(seed)

        # 洗牌
        deck = list(FULL_DECK)
        random.shuffle(deck)

        # 发牌
        hands = (
            (Role.FIRST.value, tuple(sorted(deck[:17]))),
            (Role.SECOND.value, tuple(sorted(deck[17:34]))),
            (Role.THIRD.value, tuple(sorted(deck[34:51]))),
        )

        three_cards = tuple(sorted(deck[51:54]))

        return cls(
            hands=hands,
            three_cards=three_cards,
            phase=Phase.BIDDING,
            current_player=Role.FIRST,
        )

    def with_bid(self, bid_value: int) -> 'GameState':
        """
        叫牌动作后的新状态

        Args:
            bid_value: 叫牌值 (0=不叫, 1/2/3=叫牌)

        Returns:
            新状态
        """
        if self.phase != Phase.BIDDING:
            raise ValueError("Not in bidding phase")

        # 更新叫牌信息
        bid_idx = BID_ORDER.index(self.current_player)
        new_bid_info = list(self.bid_info)
        new_bid_info[bid_idx] = bid_value

        # 更新叫牌历史
        new_bid_history = self.bid_history + ((self.current_player, bid_value),)

        # 确定下一个玩家
        next_bid_idx = bid_idx + 1

        # 检查叫牌是否结束
        max_bid = max(new_bid_info)
        bid_count = sum(1 for b in new_bid_info if b >= 0)

        # 叫牌结束条件:
        # 1. 三家都叫过
        # 2. 有人叫3分
        # 3. 三家都不叫
        if bid_count == 3 or max_bid == 3:
            # 流局: 三家都不叫
            if max_bid <= 0:
                return GameState(
                    hands=self.hands,
                    three_cards=self.three_cards,
                    phase=Phase.FINISHED,
                    current_player=self.current_player,
                    bid_info=tuple(new_bid_info),
                    bid_history=new_bid_history,
                    winner="draw",
                )

            # 确定地主
            landlord_bid_idx = new_bid_info.index(max_bid)
            landlord_bid_pos = BID_ORDER[landlord_bid_idx]

            # 获取位置映射
            role_map = get_play_roles(landlord_bid_pos)

            # 转换手牌到出牌阶段角色
            bid_hands = {r: list(cards) for r, cards in self.hands}
            play_hands = {}

            for bid_role, play_role in role_map.items():
                play_hands[play_role.value] = bid_hands[bid_role.value]

            # 地主获得底牌
            play_hands[Role.LANDLORD.value] = sorted(
                play_hands[Role.LANDLORD.value] + list(self.three_cards)
            )

            new_hands = tuple(
                (role, tuple(play_hands[role]))
                for role in [Role.LANDLORD.value, Role.LANDLORD_DOWN.value, Role.LANDLORD_UP.value]
            )

            return GameState(
                hands=new_hands,
                three_cards=self.three_cards,
                phase=Phase.PLAYING,
                current_player=Role.LANDLORD,
                landlord=Role.LANDLORD,
                landlord_bid_position=landlord_bid_pos,
                bid_info=tuple(new_bid_info),
                bid_history=new_bid_history,
            )

        # 继续叫牌
        next_player = BID_ORDER[next_bid_idx]

        return GameState(
            hands=self.hands,
            three_cards=self.three_cards,
            phase=Phase.BIDDING,
            current_player=next_player,
            bid_info=tuple(new_bid_info),
            bid_history=new_bid_history,
        )

    def with_play(self, action: Action) -> 'GameState':
        """
        出牌动作后的新状态

        Args:
            action: 出牌动作

        Returns:
            新状态
        """
        if self.phase != Phase.PLAYING:
            raise ValueError("Not in playing phase")

        # 更新手牌
        hands_dict = self.get_hands_dict()
        current_hand = hands_dict[self.current_player.value]

        if not action.is_pass:
            for card in action.cards:
                current_hand.remove(card)
            hands_dict[self.current_player.value] = current_hand

        new_hands = tuple(
            (role, tuple(sorted(cards)))
            for role, cards in hands_dict.items()
        )

        # 更新出牌历史
        new_play_history = self.play_history + ((self.current_player, action.cards),)

        # 更新炸弹数
        new_bombs_count = self.bombs_count
        if action.action_type in (ActionType.BOMB, ActionType.ROCKET):
            new_bombs_count += 1

        # 更新最近出牌
        new_last_action = self.last_action
        new_last_player = self.last_player
        if not action.is_pass:
            new_last_action = action
            new_last_player = self.current_player

        # 检查游戏是否结束
        winner = None
        new_phase = Phase.PLAYING
        if not current_hand:  # 当前玩家出完牌
            if self.current_player == Role.LANDLORD:
                winner = "landlord"
            else:
                winner = "farmer"
            new_phase = Phase.FINISHED

        # 确定下一个玩家
        current_idx = PLAY_ORDER.index(self.current_player)
        next_idx = (current_idx + 1) % 3
        next_player = PLAY_ORDER[next_idx]

        # 如果轮到最后出牌的玩家，重置 last_action
        if next_player == new_last_player:
            new_last_action = None

        return GameState(
            hands=new_hands,
            three_cards=self.three_cards,
            phase=new_phase,
            current_player=next_player,
            landlord=self.landlord,
            landlord_bid_position=self.landlord_bid_position,
            bid_info=self.bid_info,
            bid_history=self.bid_history,
            play_history=new_play_history,
            last_action=new_last_action,
            last_player=new_last_player,
            bombs_count=new_bombs_count,
            step_count=self.step_count + 1,
            winner=winner,
        )

    def with_action(self, action) -> 'GameState':
        """
        执行动作后的新状态

        Args:
            action: 叫牌值 (int) 或 出牌动作 (Action)

        Returns:
            新状态
        """
        if self.phase == Phase.BIDDING:
            if isinstance(action, int):
                return self.with_bid(action)
            raise ValueError("Bidding phase requires int action")
        elif self.phase == Phase.PLAYING:
            if isinstance(action, Action):
                return self.with_play(action)
            raise ValueError("Playing phase requires Action")
        else:
            raise ValueError("Game is finished")

    def get_legal_actions(self) -> List:
        """
        获取当前玩家的合法动作

        Returns:
            叫牌阶段: List[int]
            出牌阶段: List[Action]
        """
        if self.phase == Phase.BIDDING:
            # 叫牌合法动作
            max_bid = max(self.bid_info)
            if max_bid <= 0:
                return [0, 1, 2, 3]  # 可以叫任意分数
            elif max_bid == 1:
                return [0, 2, 3]
            elif max_bid == 2:
                return [0, 3]
            else:
                return [0]  # 只能不叫

        elif self.phase == Phase.PLAYING:
            hand = list(self.get_hand(self.current_player))
            generator = ActionGenerator(hand)

            if self.last_action is None:
                # 主动出牌
                return generator.generate_all()
            else:
                # 跟牌
                return generator.generate_responses(self.last_action)

        return []

    def is_spring(self) -> bool:
        """检查是否春天 (地主未出牌或农民未出牌)"""
        if self.winner is None:
            return False

        landlord_played = any(
            player == Role.LANDLORD
            for player, cards in self.play_history
            if cards  # 非 PASS
        )
        farmer_played = any(
            player in (Role.LANDLORD_DOWN, Role.LANDLORD_UP)
            for player, cards in self.play_history
            if cards
        )

        if self.winner == "landlord":
            return not farmer_played
        else:
            return not landlord_played

    @property
    def is_finished(self) -> bool:
        return self.phase == Phase.FINISHED

    @property
    def bid_multiplier(self) -> int:
        """叫牌倍数"""
        return max(1, max(self.bid_info))
