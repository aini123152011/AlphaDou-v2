"""
动作类型定义与动作生成器

斗地主共有 15 种牌型 (含 PASS 和 WRONG)
"""
from enum import IntEnum
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterator
from collections import defaultdict
import itertools

from .cards import Card, CARD_TO_COLUMN


class ActionType(IntEnum):
    """动作/牌型类型"""
    PASS = 0              # 不出 / 过
    SINGLE = 1            # 单张
    PAIR = 2              # 对子
    TRIPLE = 3            # 三张
    BOMB = 4              # 炸弹 (四张相同)
    ROCKET = 5            # 王炸
    TRIPLE_SINGLE = 6     # 三带一
    TRIPLE_PAIR = 7       # 三带二
    STRAIGHT = 8          # 顺子 (至少5张)
    STRAIGHT_PAIR = 9     # 连对 (至少3对)
    AIRPLANE = 10         # 飞机不带 (至少2个三张)
    AIRPLANE_SINGLE = 11  # 飞机带单
    AIRPLANE_PAIR = 12    # 飞机带对
    QUAD_SINGLE = 13      # 四带二单
    QUAD_PAIR = 14        # 四带二对
    WRONG = 15            # 非法牌型


# 顺子/连对/飞机的最小长度
MIN_STRAIGHT_LEN = 5     # 顺子至少 5 张
MIN_STRAIGHT_PAIR_LEN = 3  # 连对至少 3 对
MIN_AIRPLANE_LEN = 2     # 飞机至少 2 个三张


@dataclass(frozen=True, slots=True)
class Action:
    """
    不可变动作表示

    Attributes:
        cards: 出牌的牌面值元组 (已排序)
        action_type: 动作类型
    """
    cards: Tuple[int, ...]
    action_type: ActionType

    @classmethod
    def pass_action(cls) -> 'Action':
        """创建 PASS 动作"""
        return cls(cards=(), action_type=ActionType.PASS)

    @classmethod
    def from_cards(cls, cards: List[int], action_type: Optional[ActionType] = None) -> 'Action':
        """从牌列表创建动作"""
        sorted_cards = tuple(sorted(cards))
        if action_type is None:
            from .rules import RuleEngine
            action_type = RuleEngine.detect_action_type(list(sorted_cards))
        return cls(cards=sorted_cards, action_type=action_type)

    @property
    def is_pass(self) -> bool:
        return self.action_type == ActionType.PASS

    @property
    def is_bomb(self) -> bool:
        return self.action_type in (ActionType.BOMB, ActionType.ROCKET)

    @property
    def rank(self) -> int:
        """获取主牌面值 (用于大小比较)"""
        if not self.cards:
            return 0
        if self.action_type == ActionType.ROCKET:
            return 100  # 王炸最大
        if self.action_type == ActionType.BOMB:
            return self.cards[0]
        # 对于其他牌型，取中间位置的牌面值
        return self.cards[len(self.cards) // 2]

    def __len__(self) -> int:
        return len(self.cards)


class ActionGenerator:
    """
    合法动作生成器

    根据手牌生成所有可能的出牌组合
    """

    def __init__(self, hand_cards: List[int]):
        """
        Args:
            hand_cards: 手牌列表
        """
        self.hand = sorted(hand_cards)
        self.card_count: defaultdict[int, int] = defaultdict(int)

        for card in self.hand:
            self.card_count[card] += 1

        # 预生成基础牌型
        self._singles: List[List[int]] = []
        self._pairs: List[List[int]] = []
        self._triples: List[List[int]] = []
        self._bombs: List[List[int]] = []
        self._rocket: Optional[List[int]] = None

        self._gen_basic_types()

    def _gen_basic_types(self):
        """预生成单张、对子、三张、炸弹"""
        for card, count in self.card_count.items():
            if card in (Card.BLACK_JOKER, Card.RED_JOKER):
                continue
            if count >= 1:
                self._singles.append([card])
            if count >= 2:
                self._pairs.append([card, card])
            if count >= 3:
                self._triples.append([card, card, card])
            if count >= 4:
                self._bombs.append([card, card, card, card])

        # 王炸
        if self.card_count[Card.BLACK_JOKER] and self.card_count[Card.RED_JOKER]:
            self._rocket = [Card.BLACK_JOKER, Card.RED_JOKER]

        # 单张中加入王
        if self.card_count[Card.BLACK_JOKER]:
            self._singles.append([Card.BLACK_JOKER])
        if self.card_count[Card.RED_JOKER]:
            self._singles.append([Card.RED_JOKER])

    def gen_singles(self) -> List[List[int]]:
        """生成所有单张"""
        return self._singles.copy()

    def gen_pairs(self) -> List[List[int]]:
        """生成所有对子"""
        return self._pairs.copy()

    def gen_triples(self) -> List[List[int]]:
        """生成所有三张"""
        return self._triples.copy()

    def gen_bombs(self) -> List[List[int]]:
        """生成所有炸弹 (不含王炸)"""
        return self._bombs.copy()

    def gen_rocket(self) -> List[List[int]]:
        """生成王炸"""
        return [self._rocket] if self._rocket else []

    def gen_triple_single(self) -> List[List[int]]:
        """生成所有三带一"""
        result = []
        for triple in self._triples:
            for single in self._singles:
                if single[0] != triple[0]:
                    result.append(triple + single)
        return result

    def gen_triple_pair(self) -> List[List[int]]:
        """生成所有三带对"""
        result = []
        for triple in self._triples:
            for pair in self._pairs:
                if pair[0] != triple[0]:
                    result.append(triple + pair)
        return result

    def _gen_serial(self, base_cards: List[List[int]], min_len: int,
                    repeat: int, required_len: int = 0) -> List[List[int]]:
        """
        生成连续牌型的通用方法

        Args:
            base_cards: 基础牌 (单张/对子/三张列表)
            min_len: 最小连续长度
            repeat: 每张牌重复次数 (1=顺子, 2=连对, 3=飞机)
            required_len: 要求的精确长度，0 表示不限制
        """
        if not base_cards:
            return []

        # 获取可用的牌面值 (2和王不能参与顺子)
        valid_cards = sorted(set(
            c[0] for c in base_cards
            if c[0] < Card.TWO
        ))

        if len(valid_cards) < min_len:
            return []

        result = []

        # 找连续序列
        for start_idx in range(len(valid_cards)):
            for end_idx in range(start_idx + min_len - 1, len(valid_cards)):
                seq = valid_cards[start_idx:end_idx + 1]

                # 检查是否连续
                if not self._is_consecutive(seq):
                    break

                seq_len = len(seq)

                # 如果指定了长度，只生成该长度
                if required_len > 0 and seq_len != required_len:
                    if seq_len < required_len:
                        continue
                    else:
                        break

                # 生成牌组
                cards = []
                for card in seq:
                    cards.extend([card] * repeat)
                result.append(cards)

        return result

    def _is_consecutive(self, cards: List[int]) -> bool:
        """检查牌面值是否连续"""
        for i in range(len(cards) - 1):
            if cards[i + 1] - cards[i] != 1:
                return False
        return True

    def gen_straight(self, required_len: int = 0) -> List[List[int]]:
        """生成顺子"""
        return self._gen_serial(self._singles, MIN_STRAIGHT_LEN, 1, required_len)

    def gen_straight_pair(self, required_len: int = 0) -> List[List[int]]:
        """生成连对"""
        return self._gen_serial(self._pairs, MIN_STRAIGHT_PAIR_LEN, 2, required_len)

    def gen_airplane(self, required_len: int = 0) -> List[List[int]]:
        """生成飞机不带"""
        return self._gen_serial(self._triples, MIN_AIRPLANE_LEN, 3, required_len)

    def gen_airplane_single(self, required_len: int = 0) -> List[List[int]]:
        """生成飞机带单"""
        result = []
        airplanes = self.gen_airplane(required_len)

        for airplane in airplanes:
            airplane_set = set(airplane)
            airplane_len = len(airplane) // 3

            # 可选的单张 (不能是飞机本身的牌)
            available_singles = [s[0] for s in self._singles if s[0] not in airplane_set]

            # 需要带 airplane_len 张单
            if len(available_singles) < airplane_len:
                continue

            for combo in itertools.combinations(available_singles, airplane_len):
                result.append(sorted(airplane + list(combo)))

        return result

    def gen_airplane_pair(self, required_len: int = 0) -> List[List[int]]:
        """生成飞机带对"""
        result = []
        airplanes = self.gen_airplane(required_len)

        for airplane in airplanes:
            airplane_set = set(airplane)
            airplane_len = len(airplane) // 3

            # 可选的对子 (不能是飞机本身的牌)
            available_pairs = [p for p in self._pairs if p[0] not in airplane_set]

            if len(available_pairs) < airplane_len:
                continue

            for combo in itertools.combinations(available_pairs, airplane_len):
                pairs = []
                for p in combo:
                    pairs.extend(p)
                result.append(sorted(airplane + pairs))

        return result

    def gen_quad_single(self) -> List[List[int]]:
        """生成四带二单"""
        result = []

        for bomb in self._bombs:
            bomb_card = bomb[0]
            # 可选的单张 (不能是炸弹本身)
            available = [s[0] for s in self._singles if s[0] != bomb_card]

            if len(available) < 2:
                continue

            for combo in itertools.combinations(available, 2):
                result.append(sorted(bomb + list(combo)))

        return result

    def gen_quad_pair(self) -> List[List[int]]:
        """生成四带二对"""
        result = []

        for bomb in self._bombs:
            bomb_card = bomb[0]
            # 可选的对子 (不能是炸弹本身)
            available_pairs = [p for p in self._pairs if p[0] != bomb_card]

            if len(available_pairs) < 2:
                continue

            for combo in itertools.combinations(available_pairs, 2):
                pairs = []
                for p in combo:
                    pairs.extend(p)
                result.append(sorted(bomb + pairs))

        return result

    def generate_all(self) -> List[Action]:
        """
        生成所有可能的出牌动作 (主动出牌)

        Returns:
            所有合法动作列表
        """
        actions = []

        # 基础牌型
        for cards in self.gen_singles():
            actions.append(Action(tuple(cards), ActionType.SINGLE))
        for cards in self.gen_pairs():
            actions.append(Action(tuple(cards), ActionType.PAIR))
        for cards in self.gen_triples():
            actions.append(Action(tuple(cards), ActionType.TRIPLE))
        for cards in self.gen_bombs():
            actions.append(Action(tuple(cards), ActionType.BOMB))
        for cards in self.gen_rocket():
            actions.append(Action(tuple(cards), ActionType.ROCKET))

        # 带牌
        for cards in self.gen_triple_single():
            actions.append(Action(tuple(sorted(cards)), ActionType.TRIPLE_SINGLE))
        for cards in self.gen_triple_pair():
            actions.append(Action(tuple(sorted(cards)), ActionType.TRIPLE_PAIR))

        # 连续牌型
        for cards in self.gen_straight():
            actions.append(Action(tuple(cards), ActionType.STRAIGHT))
        for cards in self.gen_straight_pair():
            actions.append(Action(tuple(cards), ActionType.STRAIGHT_PAIR))
        for cards in self.gen_airplane():
            actions.append(Action(tuple(cards), ActionType.AIRPLANE))
        for cards in self.gen_airplane_single():
            actions.append(Action(tuple(cards), ActionType.AIRPLANE_SINGLE))
        for cards in self.gen_airplane_pair():
            actions.append(Action(tuple(cards), ActionType.AIRPLANE_PAIR))

        # 四带
        for cards in self.gen_quad_single():
            actions.append(Action(tuple(sorted(cards)), ActionType.QUAD_SINGLE))
        for cards in self.gen_quad_pair():
            actions.append(Action(tuple(sorted(cards)), ActionType.QUAD_PAIR))

        return actions

    def generate_responses(self, last_action: Action) -> List[Action]:
        """
        生成对上家出牌的合法响应

        Args:
            last_action: 上家的出牌

        Returns:
            所有合法响应动作列表
        """
        # PASS 表示主动出牌
        if last_action.is_pass:
            return self.generate_all()

        responses = [Action.pass_action()]

        # 王炸无法被打过
        if last_action.action_type == ActionType.ROCKET:
            return responses

        last_rank = last_action.rank
        last_type = last_action.action_type

        # 同类型更大的牌
        if last_type == ActionType.SINGLE:
            for cards in self.gen_singles():
                if cards[0] > last_rank:
                    responses.append(Action(tuple(cards), ActionType.SINGLE))

        elif last_type == ActionType.PAIR:
            for cards in self.gen_pairs():
                if cards[0] > last_rank:
                    responses.append(Action(tuple(cards), ActionType.PAIR))

        elif last_type == ActionType.TRIPLE:
            for cards in self.gen_triples():
                if cards[0] > last_rank:
                    responses.append(Action(tuple(cards), ActionType.TRIPLE))

        elif last_type == ActionType.BOMB:
            for cards in self.gen_bombs():
                if cards[0] > last_rank:
                    responses.append(Action(tuple(cards), ActionType.BOMB))

        elif last_type == ActionType.TRIPLE_SINGLE:
            for cards in self.gen_triple_single():
                # 三带一的 rank 是三张的牌面值
                triple_rank = sorted(cards)[1]  # 中间那张就是三张
                if triple_rank > last_rank:
                    responses.append(Action(tuple(sorted(cards)), ActionType.TRIPLE_SINGLE))

        elif last_type == ActionType.TRIPLE_PAIR:
            for cards in self.gen_triple_pair():
                triple_rank = sorted(cards)[2]  # 三张的位置
                if triple_rank > last_rank:
                    responses.append(Action(tuple(sorted(cards)), ActionType.TRIPLE_PAIR))

        elif last_type == ActionType.STRAIGHT:
            required_len = len(last_action)
            for cards in self.gen_straight(required_len):
                if cards[0] > last_rank:
                    responses.append(Action(tuple(cards), ActionType.STRAIGHT))

        elif last_type == ActionType.STRAIGHT_PAIR:
            required_len = len(last_action) // 2
            for cards in self.gen_straight_pair(required_len):
                if cards[0] > last_rank:
                    responses.append(Action(tuple(cards), ActionType.STRAIGHT_PAIR))

        elif last_type == ActionType.AIRPLANE:
            required_len = len(last_action) // 3
            for cards in self.gen_airplane(required_len):
                if cards[0] > last_rank:
                    responses.append(Action(tuple(cards), ActionType.AIRPLANE))

        elif last_type == ActionType.AIRPLANE_SINGLE:
            # 飞机带单的长度 = 三张数量 × 4
            airplane_count = len(last_action) // 4
            for cards in self.gen_airplane_single(airplane_count):
                # 飞机的 rank 是三张的起始牌面值
                # 需要从cards中找出三张的部分
                card_count = defaultdict(int)
                for c in cards:
                    card_count[c] += 1
                triple_cards = sorted([c for c, n in card_count.items() if n == 3])
                if triple_cards and triple_cards[0] > last_rank:
                    responses.append(Action(tuple(cards), ActionType.AIRPLANE_SINGLE))

        elif last_type == ActionType.AIRPLANE_PAIR:
            airplane_count = len(last_action) // 5
            for cards in self.gen_airplane_pair(airplane_count):
                card_count = defaultdict(int)
                for c in cards:
                    card_count[c] += 1
                triple_cards = sorted([c for c, n in card_count.items() if n == 3])
                if triple_cards and triple_cards[0] > last_rank:
                    responses.append(Action(tuple(cards), ActionType.AIRPLANE_PAIR))

        elif last_type == ActionType.QUAD_SINGLE:
            for cards in self.gen_quad_single():
                # 四带二的 rank 是四张的牌面值
                card_count = defaultdict(int)
                for c in cards:
                    card_count[c] += 1
                quad_card = [c for c, n in card_count.items() if n == 4][0]
                if quad_card > last_rank:
                    responses.append(Action(tuple(sorted(cards)), ActionType.QUAD_SINGLE))

        elif last_type == ActionType.QUAD_PAIR:
            for cards in self.gen_quad_pair():
                card_count = defaultdict(int)
                for c in cards:
                    card_count[c] += 1
                quad_card = [c for c, n in card_count.items() if n == 4][0]
                if quad_card > last_rank:
                    responses.append(Action(tuple(sorted(cards)), ActionType.QUAD_PAIR))

        # 炸弹可以打任何非炸弹牌型
        if last_type != ActionType.BOMB:
            for cards in self.gen_bombs():
                responses.append(Action(tuple(cards), ActionType.BOMB))

        # 王炸可以打任何牌
        for cards in self.gen_rocket():
            responses.append(Action(tuple(cards), ActionType.ROCKET))

        return responses
