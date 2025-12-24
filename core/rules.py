"""
规则引擎 - 牌型检测、大小比较、合法性验证

所有方法都是纯函数，无状态
"""
from typing import List, Optional, Dict, Tuple
from collections import Counter, defaultdict

from .cards import Card
from .actions import Action, ActionType, MIN_STRAIGHT_LEN, MIN_STRAIGHT_PAIR_LEN, MIN_AIRPLANE_LEN


class RuleEngine:
    """
    斗地主规则引擎

    提供牌型检测、大小比较、合法性验证等功能
    所有方法都是静态方法，无状态
    """

    @staticmethod
    def is_consecutive(cards: List[int]) -> bool:
        """
        检查牌面值列表是否连续

        Args:
            cards: 已排序的牌面值列表

        Returns:
            是否连续
        """
        for i in range(len(cards) - 1):
            if cards[i + 1] - cards[i] != 1:
                return False
        return True

    @staticmethod
    def detect_action_type(cards: List[int]) -> ActionType:
        """
        检测牌型

        Args:
            cards: 牌列表

        Returns:
            牌型枚举值
        """
        if not cards:
            return ActionType.PASS

        n = len(cards)
        cards = sorted(cards)
        counter = Counter(cards)
        count_values = list(counter.values())
        unique_cards = sorted(counter.keys())

        # 单张
        if n == 1:
            return ActionType.SINGLE

        # 对子 或 王炸
        if n == 2:
            if cards[0] == cards[1]:
                return ActionType.PAIR
            if set(cards) == {Card.BLACK_JOKER, Card.RED_JOKER}:
                return ActionType.ROCKET
            return ActionType.WRONG

        # 三张
        if n == 3:
            if len(counter) == 1:
                return ActionType.TRIPLE
            return ActionType.WRONG

        # 四张: 炸弹 或 三带一
        if n == 4:
            if len(counter) == 1:
                return ActionType.BOMB
            if len(counter) == 2:
                if 3 in count_values:
                    return ActionType.TRIPLE_SINGLE
            return ActionType.WRONG

        # 5张: 三带二 或 顺子
        if n == 5:
            if len(counter) == 2 and 3 in count_values:
                return ActionType.TRIPLE_PAIR
            # 检查顺子
            if len(counter) == 5 and max(unique_cards) < Card.TWO:
                if RuleEngine.is_consecutive(unique_cards):
                    return ActionType.STRAIGHT
            return ActionType.WRONG

        # 6张及以上: 需要更复杂的检测
        # 先检查顺子 (所有牌都是单张且连续)
        if len(counter) == n and max(unique_cards) < Card.TWO:
            if RuleEngine.is_consecutive(unique_cards):
                return ActionType.STRAIGHT

        # 统计各数量的牌有多少种
        count_stat: Dict[int, int] = defaultdict(int)
        for c, num in counter.items():
            count_stat[num] += 1

        # 四带二单
        if n == 6 and count_stat.get(4) == 1:
            if count_stat.get(1) == 2 or count_stat.get(2) == 1:
                return ActionType.QUAD_SINGLE

        # 四带二对
        if n == 8 and count_stat.get(4) == 1 and count_stat.get(2) == 2:
            return ActionType.QUAD_PAIR

        # 连对: 所有牌都是对子且连续
        if len(counter) == n // 2 and all(v == 2 for v in count_values):
            if len(unique_cards) >= MIN_STRAIGHT_PAIR_LEN and max(unique_cards) < Card.TWO:
                if RuleEngine.is_consecutive(unique_cards):
                    return ActionType.STRAIGHT_PAIR

        # 飞机不带: 所有牌都是三张且连续
        if len(counter) == n // 3 and all(v == 3 for v in count_values):
            if len(unique_cards) >= MIN_AIRPLANE_LEN and max(unique_cards) < Card.TWO:
                if RuleEngine.is_consecutive(unique_cards):
                    return ActionType.AIRPLANE

        # 飞机带翅膀 (带单或带对)
        if count_stat.get(3, 0) >= MIN_AIRPLANE_LEN:
            triple_cards = sorted([c for c, v in counter.items() if v == 3])
            single_cards = [c for c, v in counter.items() if v == 1]
            pair_cards = [c for c, v in counter.items() if v == 2]

            # 检查三张部分是否连续
            if max(triple_cards) < Card.TWO and RuleEngine.is_consecutive(triple_cards):
                triple_count = len(triple_cards)

                # 飞机带单: 单张数量 + 对子数量*2 == 三张数量
                if len(single_cards) + len(pair_cards) * 2 == triple_count:
                    return ActionType.AIRPLANE_SINGLE

                # 飞机带对: 对子数量 == 三张数量
                if len(pair_cards) == triple_count and len(single_cards) == 0:
                    return ActionType.AIRPLANE_PAIR

            # 特殊情况: 4个三张时，可能有一个三张被当作单张
            if len(triple_cards) == 4:
                # 尝试 3个三张 + 翅膀
                for i in range(4):
                    sub_triples = triple_cards[:i] + triple_cards[i+1:]
                    if RuleEngine.is_consecutive(sub_triples):
                        extra = [triple_cards[i]] * 3
                        remaining_singles = single_cards + extra
                        if len(remaining_singles) == 3:
                            return ActionType.AIRPLANE_SINGLE
                        remaining_pairs = pair_cards + [(triple_cards[i],)] * 3
                        # 这种情况较复杂，简化处理

        return ActionType.WRONG

    @staticmethod
    def compare_actions(a: Action, b: Action) -> int:
        """
        比较两个动作的大小

        Args:
            a: 动作 a
            b: 动作 b (通常是上家的牌)

        Returns:
            1 if a > b, -1 if a < b, 0 if 不可比较
        """
        # PASS 不参与比较
        if a.is_pass or b.is_pass:
            return 0

        # 王炸最大
        if a.action_type == ActionType.ROCKET:
            return 1
        if b.action_type == ActionType.ROCKET:
            return -1

        # 炸弹 vs 非炸弹
        if a.action_type == ActionType.BOMB and b.action_type != ActionType.BOMB:
            return 1
        if b.action_type == ActionType.BOMB and a.action_type != ActionType.BOMB:
            return -1

        # 炸弹 vs 炸弹
        if a.action_type == ActionType.BOMB and b.action_type == ActionType.BOMB:
            return 1 if a.cards[0] > b.cards[0] else (-1 if a.cards[0] < b.cards[0] else 0)

        # 不同类型不可比较 (炸弹情况已处理)
        if a.action_type != b.action_type:
            return 0

        # 长度不同不可比较 (针对顺子、连对、飞机等)
        if len(a) != len(b):
            return 0

        # 同类型比较 rank
        a_rank = RuleEngine.get_rank(a)
        b_rank = RuleEngine.get_rank(b)

        if a_rank > b_rank:
            return 1
        elif a_rank < b_rank:
            return -1
        return 0

    @staticmethod
    def get_rank(action: Action) -> int:
        """
        获取动作的主牌面值 (用于大小比较)

        Args:
            action: 动作

        Returns:
            主牌面值
        """
        if not action.cards:
            return 0

        cards = list(action.cards)
        counter = Counter(cards)

        action_type = action.action_type

        # 单张、对子、三张、炸弹、顺子: 直接取第一张
        if action_type in (ActionType.SINGLE, ActionType.PAIR, ActionType.TRIPLE,
                           ActionType.BOMB, ActionType.STRAIGHT, ActionType.STRAIGHT_PAIR,
                           ActionType.AIRPLANE):
            return min(cards)

        # 三带一/三带二: 取三张的牌面值
        if action_type in (ActionType.TRIPLE_SINGLE, ActionType.TRIPLE_PAIR):
            for card, count in counter.items():
                if count == 3:
                    return card

        # 飞机带翅膀: 取三张部分的最小牌面值
        if action_type in (ActionType.AIRPLANE_SINGLE, ActionType.AIRPLANE_PAIR):
            triples = [card for card, count in counter.items() if count == 3]
            return min(triples) if triples else 0

        # 四带: 取四张的牌面值
        if action_type in (ActionType.QUAD_SINGLE, ActionType.QUAD_PAIR):
            for card, count in counter.items():
                if count == 4:
                    return card

        # 王炸
        if action_type == ActionType.ROCKET:
            return 100  # 最大

        return cards[0] if cards else 0

    @staticmethod
    def is_valid_play(
        action: Action,
        last_action: Optional[Action],
        hand: List[int]
    ) -> bool:
        """
        验证出牌是否合法

        Args:
            action: 要出的牌
            last_action: 上家的牌 (None 或 PASS 表示主动出牌)
            hand: 当前手牌

        Returns:
            是否合法
        """
        # PASS 始终合法 (如果不是主动出牌)
        if action.is_pass:
            # 主动出牌时不能 PASS
            if last_action is None or last_action.is_pass:
                return False
            return True

        # 检查牌型是否正确
        if action.action_type == ActionType.WRONG:
            return False

        # 检查牌是否在手中
        hand_counter = Counter(hand)
        action_counter = Counter(action.cards)
        for card, count in action_counter.items():
            if hand_counter.get(card, 0) < count:
                return False

        # 主动出牌: 只要牌型正确且在手中就合法
        if last_action is None or last_action.is_pass:
            return True

        # 跟牌: 需要能打过上家
        return RuleEngine.compare_actions(action, last_action) == 1

    @staticmethod
    def can_beat(hand: List[int], last_action: Action) -> bool:
        """
        检查手牌是否能打过上家

        Args:
            hand: 当前手牌
            last_action: 上家的牌

        Returns:
            是否能打过
        """
        from .actions import ActionGenerator

        if last_action.is_pass:
            return True

        generator = ActionGenerator(hand)
        responses = generator.generate_responses(last_action)

        # 排除 PASS
        return any(not r.is_pass for r in responses)

    @staticmethod
    def get_winner(
        played_cards: Dict[str, List[int]],
        hands: Dict[str, List[int]]
    ) -> Optional[str]:
        """
        判断游戏是否结束及赢家

        Args:
            played_cards: 各玩家已出的牌
            hands: 各玩家剩余手牌

        Returns:
            赢家角色名，未结束返回 None
        """
        for role, hand in hands.items():
            if not hand:  # 手牌出完
                if role == "landlord":
                    return "landlord"
                else:
                    return "farmer"
        return None

    @staticmethod
    def calculate_score(
        winner: str,
        bid_count: int,
        bombs_count: int,
        is_spring: bool
    ) -> Dict[str, int]:
        """
        计算得分

        Args:
            winner: 赢家 ("landlord" 或 "farmer")
            bid_count: 叫牌倍数 (1-3)
            bombs_count: 炸弹数量
            is_spring: 是否春天

        Returns:
            各角色得分字典
        """
        base = bid_count

        # 炸弹翻倍
        multiplier = 2 ** bombs_count

        # 春天翻倍
        if is_spring:
            multiplier *= 2

        score = base * multiplier

        if winner == "landlord":
            return {
                "landlord": score * 2,
                "landlord_down": -score,
                "landlord_up": -score,
            }
        else:
            return {
                "landlord": -score * 2,
                "landlord_down": score,
                "landlord_up": score,
            }
