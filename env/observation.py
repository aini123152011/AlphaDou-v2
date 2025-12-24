"""
观察空间编码

将游戏状态转换为神经网络可用的特征表示
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import itertools
import numpy as np

from core.state import GameState, Phase, Role, PLAY_ORDER
from core.cards import cards_to_array, FULL_DECK
from core.actions import Action, ActionType, ActionGenerator


# 动作索引映射 (预计算)
# 所有可能的动作数量约为 27472
# 实际实现中使用动态生成 + 缓存


@dataclass
class Observation:
    """
    结构化观测

    Attributes:
        hand: 自己的手牌 (54,)
        other_hands: 其他玩家手牌 (用于完全信息设置) (2, 54)
        played_cards: 各玩家已出牌累计 (3, 54)
        history: 最近 N 步出牌历史 (N, 54)
        last_action: 上一个非 PASS 动作 (54,)
        legal_mask: 合法动作掩码 (动态大小)
        position: 位置编码 (6,) one-hot
        bid_info: 叫牌信息 (3,)
        cards_left: 各玩家剩余牌数 (3,)
        bombs_count: 已打出炸弹数
        phase: 游戏阶段
    """
    hand: np.ndarray
    other_hands: np.ndarray
    played_cards: np.ndarray
    history: np.ndarray
    last_action: np.ndarray
    legal_actions: List
    position: np.ndarray
    bid_info: np.ndarray
    cards_left: np.ndarray
    bombs_count: int
    phase: str

    def to_dict(self) -> Dict[str, np.ndarray]:
        """转换为字典格式"""
        return {
            "hand": self.hand,
            "other_hands": self.other_hands,
            "played_cards": self.played_cards,
            "history": self.history,
            "last_action": self.last_action,
            "position": self.position,
            "bid_info": self.bid_info,
            "cards_left": self.cards_left,
        }

    def to_flat_array(self) -> np.ndarray:
        """
        展平为单一向量 (用于简单网络)

        特征维度:
        - hand: 54
        - played_cards: 3 * 54 = 162
        - history: N * 54
        - last_action: 54
        - position: 6
        - bid_info: 3
        - cards_left: 3
        - bombs_count: 1
        """
        flat = np.concatenate([
            self.hand,
            self.played_cards.flatten(),
            self.history.flatten(),
            self.last_action,
            self.position,
            self.bid_info,
            self.cards_left,
            np.array([self.bombs_count], dtype=np.float32),
        ])
        return flat


class ObservationBuilder:
    """
    观测构建器

    负责将 GameState 转换为 Observation
    """

    def __init__(
        self,
        history_length: int = 15,
        include_other_hands: bool = False,
    ):
        """
        Args:
            history_length: 历史记录长度
            include_other_hands: 是否包含其他玩家手牌 (完全信息)
        """
        self.history_length = history_length
        self.include_other_hands = include_other_hands

    def build(self, state: GameState, perspective: Optional[Role] = None) -> Observation:
        """
        从游戏状态构建观测

        Args:
            state: 游戏状态
            perspective: 视角玩家 (默认为当前玩家)

        Returns:
            Observation 对象
        """
        if perspective is None:
            perspective = state.current_player

        # 编码手牌
        hand = self._encode_hand(state, perspective)

        # 编码其他玩家手牌
        other_hands = self._encode_other_hands(state, perspective)

        # 编码已出牌
        played_cards = self._encode_played_cards(state)

        # 编码历史
        history = self._encode_history(state)

        # 编码上一个动作
        last_action = self._encode_last_action(state)

        # 获取合法动作
        legal_actions = self._get_legal_actions(state)

        # 编码位置
        position = self._encode_position(perspective)

        # 叫牌信息
        bid_info = np.array(state.bid_info, dtype=np.float32)

        # 剩余牌数
        cards_left = self._encode_cards_left(state)

        return Observation(
            hand=hand,
            other_hands=other_hands,
            played_cards=played_cards,
            history=history,
            last_action=last_action,
            legal_actions=legal_actions,
            position=position,
            bid_info=bid_info,
            cards_left=cards_left,
            bombs_count=state.bombs_count,
            phase=state.phase.value,
        )

    def _encode_hand(self, state: GameState, player: Role) -> np.ndarray:
        """编码指定玩家的手牌"""
        hand = state.get_hand(player)
        return cards_to_array(list(hand))

    def _encode_other_hands(self, state: GameState, player: Role) -> np.ndarray:
        """编码其他玩家手牌"""
        result = np.zeros((2, 54), dtype=np.float32)

        if not self.include_other_hands:
            return result

        if state.phase != Phase.PLAYING:
            return result

        # 获取其他两个玩家
        others = [r for r in PLAY_ORDER if r != player]

        for i, other in enumerate(others):
            hand = state.get_hand(other)
            result[i] = cards_to_array(list(hand))

        return result

    def _encode_played_cards(self, state: GameState) -> np.ndarray:
        """
        编码各玩家已出的牌

        Returns:
            (3, 54) 数组，按 LANDLORD, LANDLORD_DOWN, LANDLORD_UP 顺序
        """
        result = np.zeros((3, 54), dtype=np.float32)

        if state.phase != Phase.PLAYING:
            return result

        # 统计各玩家已出的牌
        played = {role: [] for role in PLAY_ORDER}
        for player, cards in state.play_history:
            if cards:  # 非 PASS
                played[player].extend(cards)

        for i, role in enumerate(PLAY_ORDER):
            if played[role]:
                result[i] = cards_to_array(played[role])

        return result

    def _encode_history(self, state: GameState) -> np.ndarray:
        """
        编码最近 N 步出牌历史

        Returns:
            (history_length, 54) 数组
        """
        result = np.zeros((self.history_length, 54), dtype=np.float32)

        if state.phase != Phase.PLAYING:
            return result

        # 取最近 N 步
        recent = list(state.play_history[-self.history_length:])

        for i, (player, cards) in enumerate(recent):
            if cards:  # 非 PASS
                result[i] = cards_to_array(list(cards))
            # PASS 时保持全 0

        return result

    def _encode_last_action(self, state: GameState) -> np.ndarray:
        """编码上一个有效动作"""
        if state.last_action is None or state.last_action.is_pass:
            return np.zeros(54, dtype=np.float32)

        return cards_to_array(list(state.last_action.cards))

    def _get_legal_actions(self, state: GameState) -> List:
        """获取合法动作列表"""
        return state.get_legal_actions()

    def _encode_position(self, player: Role) -> np.ndarray:
        """
        编码玩家位置

        Returns:
            (6,) one-hot 数组
        """
        positions = [
            Role.FIRST, Role.SECOND, Role.THIRD,
            Role.LANDLORD, Role.LANDLORD_DOWN, Role.LANDLORD_UP
        ]
        result = np.zeros(6, dtype=np.float32)

        if player in positions:
            result[positions.index(player)] = 1

        return result

    def _encode_cards_left(self, state: GameState) -> np.ndarray:
        """
        编码各玩家剩余牌数

        Returns:
            (3,) 数组，归一化到 [0, 1]
        """
        result = np.zeros(3, dtype=np.float32)

        if state.phase == Phase.BIDDING:
            # 叫牌阶段都是 17 张
            result[:] = 17 / 20
        elif state.phase == Phase.PLAYING:
            for i, role in enumerate(PLAY_ORDER):
                hand = state.get_hand(role)
                result[i] = len(hand) / 20  # 归一化

        return result


class ActionEncoder:
    """
    动作编码器

    将 Action 对象与索引相互转换
    """

    def __init__(self):
        self._action_to_idx: Dict[Tuple, int] = {}
        self._idx_to_action: Dict[int, Tuple] = {}
        self._build_action_space()

    def _build_action_space(self):
        """
        构建完整动作空间

        斗地主的动作空间较大，采用分组编码:
        - PASS: 1
        - 单张: 15 (3-2, 小王, 大王)
        - 对子: 13
        - 三张: 13
        - 炸弹: 13
        - 王炸: 1
        - 三带一/对: 组合
        - 顺子/连对/飞机: 组合
        - 四带: 组合
        """
        idx = 0

        # PASS
        self._action_to_idx[()] = idx
        self._idx_to_action[idx] = ()
        idx += 1

        # 单张
        for card in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 20, 30]:
            key = (card,)
            self._action_to_idx[key] = idx
            self._idx_to_action[idx] = key
            idx += 1

        # 对子 (不含王)
        for card in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17]:
            key = (card, card)
            self._action_to_idx[key] = idx
            self._idx_to_action[idx] = key
            idx += 1

        # 三张
        for card in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17]:
            key = (card, card, card)
            self._action_to_idx[key] = idx
            self._idx_to_action[idx] = key
            idx += 1

        # 炸弹
        for card in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17]:
            key = (card, card, card, card)
            self._action_to_idx[key] = idx
            self._idx_to_action[idx] = key
            idx += 1

        # 王炸
        key = (20, 30)
        self._action_to_idx[key] = idx
        self._idx_to_action[idx] = key
        idx += 1

        # 三带一
        for main in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17]:
            for kicker in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 20, 30]:
                if kicker != main:
                    key = tuple(sorted([main, main, main, kicker]))
                    if key not in self._action_to_idx:
                        self._action_to_idx[key] = idx
                        self._idx_to_action[idx] = key
                        idx += 1

        # 三带对
        for main in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17]:
            for kicker in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17]:
                if kicker != main:
                    key = tuple(sorted([main, main, main, kicker, kicker]))
                    if key not in self._action_to_idx:
                        self._action_to_idx[key] = idx
                        self._idx_to_action[idx] = key
                        idx += 1

        # 顺子 (5-12 张)
        for length in range(5, 13):
            for start in range(3, 15 - length + 1):
                key = tuple(range(start, start + length))
                self._action_to_idx[key] = idx
                self._idx_to_action[idx] = key
                idx += 1

        # 连对 (3-10 对)
        for length in range(3, 11):
            for start in range(3, 15 - length + 1):
                key = tuple(c for c in range(start, start + length) for _ in range(2))
                self._action_to_idx[key] = idx
                self._idx_to_action[idx] = key
                idx += 1

        # 飞机不带 (2-6 个三张)
        for length in range(2, 7):
            for start in range(3, 15 - length + 1):
                key = tuple(c for c in range(start, start + length) for _ in range(3))
                self._action_to_idx[key] = idx
                self._idx_to_action[idx] = key
                idx += 1

        # 飞机带单 (2-6 个三张 + 等量单张)
        # 所有可用单张 (不含 2，因为 2 不参与连续牌型的主体)
        all_singles = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 20, 30]
        for length in range(2, 7):  # 2-6 个三张
            for start in range(3, 15 - length + 1):
                airplane_cards = set(range(start, start + length))
                # 可用的单张 (不能是飞机主体的牌)
                available_kickers = [c for c in all_singles if c not in airplane_cards]
                # 从可用单张中选 length 个
                for kickers in itertools.combinations(available_kickers, length):
                    airplane = [c for c in range(start, start + length) for _ in range(3)]
                    key = tuple(sorted(airplane + list(kickers)))
                    if key not in self._action_to_idx:
                        self._action_to_idx[key] = idx
                        self._idx_to_action[idx] = key
                        idx += 1

        # 飞机带对 (2-6 个三张 + 等量对子)
        all_pairs_cards = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17]  # 不含王
        for length in range(2, 7):  # 2-6 个三张
            for start in range(3, 15 - length + 1):
                airplane_cards = set(range(start, start + length))
                # 可用的对子牌面 (不能是飞机主体的牌)
                available_pairs = [c for c in all_pairs_cards if c not in airplane_cards]
                # 从可用对子中选 length 个
                for pair_cards in itertools.combinations(available_pairs, length):
                    airplane = [c for c in range(start, start + length) for _ in range(3)]
                    pairs = [c for c in pair_cards for _ in range(2)]
                    key = tuple(sorted(airplane + pairs))
                    if key not in self._action_to_idx:
                        self._action_to_idx[key] = idx
                        self._idx_to_action[idx] = key
                        idx += 1

        # 四带二单 (1个四张 + 2个不同单张)
        quad_cards = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17]
        for quad in quad_cards:
            # 可用的单张 (不能是四张本身)
            available_singles = [c for c in all_singles if c != quad]
            for kickers in itertools.combinations(available_singles, 2):
                key = tuple(sorted([quad] * 4 + list(kickers)))
                if key not in self._action_to_idx:
                    self._action_to_idx[key] = idx
                    self._idx_to_action[idx] = key
                    idx += 1

        # 四带二对 (1个四张 + 2个不同对子)
        for quad in quad_cards:
            # 可用的对子牌面 (不能是四张本身)
            available_pairs = [c for c in all_pairs_cards if c != quad]
            for pair_cards in itertools.combinations(available_pairs, 2):
                pairs = [c for c in pair_cards for _ in range(2)]
                key = tuple(sorted([quad] * 4 + pairs))
                if key not in self._action_to_idx:
                    self._action_to_idx[key] = idx
                    self._idx_to_action[idx] = key
                    idx += 1

        self._num_actions = idx

    @property
    def num_actions(self) -> int:
        """动作空间大小"""
        return self._num_actions

    def encode(self, action: Action) -> int:
        """
        将 Action 编码为索引

        Args:
            action: Action 对象

        Returns:
            动作索引，未找到返回 -1
        """
        key = action.cards
        return self._action_to_idx.get(key, -1)

    def decode(self, idx: int) -> Optional[Action]:
        """
        将索引解码为 Action

        Args:
            idx: 动作索引

        Returns:
            Action 对象，未找到返回 None
        """
        if idx not in self._idx_to_action:
            return None

        cards = self._idx_to_action[idx]
        if not cards:
            return Action.pass_action()

        return Action.from_cards(list(cards))

    def get_legal_action_indices(self, legal_actions: List[Action]) -> List[int]:
        """
        获取合法动作的索引列表

        Args:
            legal_actions: 合法 Action 列表

        Returns:
            索引列表
        """
        indices = []
        for action in legal_actions:
            idx = self.encode(action)
            if idx >= 0:
                indices.append(idx)
        return indices

    def build_legal_mask(self, legal_actions: List[Action]) -> np.ndarray:
        """
        构建合法动作掩码

        Args:
            legal_actions: 合法 Action 列表

        Returns:
            (num_actions,) 布尔数组
        """
        mask = np.zeros(self._num_actions, dtype=np.float32)
        for action in legal_actions:
            idx = self.encode(action)
            if idx >= 0:
                mask[idx] = 1
        return mask


# 全局单例
_action_encoder: Optional[ActionEncoder] = None


def get_action_encoder() -> ActionEncoder:
    """获取全局动作编码器"""
    global _action_encoder
    if _action_encoder is None:
        _action_encoder = ActionEncoder()
    return _action_encoder
