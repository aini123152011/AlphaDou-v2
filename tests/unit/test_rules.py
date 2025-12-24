"""规则引擎测试"""
import pytest

from core.cards import Card
from core.actions import Action, ActionType
from core.rules import RuleEngine


class TestDetectActionType:
    """牌型检测测试"""

    def test_pass(self):
        assert RuleEngine.detect_action_type([]) == ActionType.PASS

    def test_single(self):
        assert RuleEngine.detect_action_type([5]) == ActionType.SINGLE
        assert RuleEngine.detect_action_type([Card.RED_JOKER]) == ActionType.SINGLE

    def test_pair(self):
        assert RuleEngine.detect_action_type([5, 5]) == ActionType.PAIR

    def test_triple(self):
        assert RuleEngine.detect_action_type([5, 5, 5]) == ActionType.TRIPLE

    def test_bomb(self):
        assert RuleEngine.detect_action_type([5, 5, 5, 5]) == ActionType.BOMB

    def test_rocket(self):
        assert RuleEngine.detect_action_type([Card.BLACK_JOKER, Card.RED_JOKER]) == ActionType.ROCKET

    def test_triple_single(self):
        assert RuleEngine.detect_action_type([5, 5, 5, 3]) == ActionType.TRIPLE_SINGLE

    def test_triple_pair(self):
        assert RuleEngine.detect_action_type([5, 5, 5, 3, 3]) == ActionType.TRIPLE_PAIR

    def test_straight_5(self):
        assert RuleEngine.detect_action_type([3, 4, 5, 6, 7]) == ActionType.STRAIGHT

    def test_straight_12(self):
        # 3-A 顺子
        cards = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        assert RuleEngine.detect_action_type(cards) == ActionType.STRAIGHT

    def test_straight_no_2(self):
        # 2 不能参与顺子
        assert RuleEngine.detect_action_type([10, 11, 12, 13, 14, 17]) == ActionType.WRONG

    def test_straight_pair(self):
        assert RuleEngine.detect_action_type([3, 3, 4, 4, 5, 5]) == ActionType.STRAIGHT_PAIR

    def test_airplane(self):
        assert RuleEngine.detect_action_type([3, 3, 3, 4, 4, 4]) == ActionType.AIRPLANE

    def test_airplane_single(self):
        assert RuleEngine.detect_action_type([3, 3, 3, 4, 4, 4, 5, 6]) == ActionType.AIRPLANE_SINGLE

    def test_airplane_pair(self):
        assert RuleEngine.detect_action_type([3, 3, 3, 4, 4, 4, 5, 5, 6, 6]) == ActionType.AIRPLANE_PAIR

    def test_quad_single(self):
        # 四带二单
        assert RuleEngine.detect_action_type([5, 5, 5, 5, 3, 4]) == ActionType.QUAD_SINGLE
        # 四带一对 (也算四带二单)
        assert RuleEngine.detect_action_type([5, 5, 5, 5, 3, 3]) == ActionType.QUAD_SINGLE

    def test_quad_pair(self):
        assert RuleEngine.detect_action_type([5, 5, 5, 5, 3, 3, 4, 4]) == ActionType.QUAD_PAIR

    def test_wrong(self):
        assert RuleEngine.detect_action_type([3, 4]) == ActionType.WRONG  # 不是对子
        assert RuleEngine.detect_action_type([3, 3, 4]) == ActionType.WRONG  # 不成牌型


class TestCompareActions:
    """动作比较测试"""

    def test_compare_singles(self):
        a = Action.from_cards([5])
        b = Action.from_cards([3])
        assert RuleEngine.compare_actions(a, b) == 1  # 5 > 3

        a = Action.from_cards([3])
        b = Action.from_cards([5])
        assert RuleEngine.compare_actions(a, b) == -1  # 3 < 5

    def test_compare_pairs(self):
        a = Action.from_cards([7, 7])
        b = Action.from_cards([5, 5])
        assert RuleEngine.compare_actions(a, b) == 1

    def test_compare_bombs(self):
        a = Action.from_cards([8, 8, 8, 8])
        b = Action.from_cards([5, 5, 5, 5])
        assert RuleEngine.compare_actions(a, b) == 1

    def test_bomb_beats_non_bomb(self):
        bomb = Action.from_cards([3, 3, 3, 3])
        single = Action.from_cards([Card.RED_JOKER])
        assert RuleEngine.compare_actions(bomb, single) == 1
        assert RuleEngine.compare_actions(single, bomb) == -1

    def test_rocket_beats_bomb(self):
        rocket = Action.from_cards([Card.BLACK_JOKER, Card.RED_JOKER])
        bomb = Action.from_cards([17, 17, 17, 17])  # 2222
        assert RuleEngine.compare_actions(rocket, bomb) == 1
        assert RuleEngine.compare_actions(bomb, rocket) == -1

    def test_rocket_is_max(self):
        rocket = Action.from_cards([Card.BLACK_JOKER, Card.RED_JOKER])
        single = Action.from_cards([Card.RED_JOKER])
        assert RuleEngine.compare_actions(rocket, single) == 1

    def test_different_types_incomparable(self):
        single = Action.from_cards([5])
        pair = Action.from_cards([3, 3])
        assert RuleEngine.compare_actions(single, pair) == 0

    def test_different_lengths_incomparable(self):
        straight5 = Action.from_cards([3, 4, 5, 6, 7])
        straight6 = Action.from_cards([3, 4, 5, 6, 7, 8])
        assert RuleEngine.compare_actions(straight5, straight6) == 0

    def test_pass_not_compared(self):
        pass_action = Action.pass_action()
        single = Action.from_cards([5])
        assert RuleEngine.compare_actions(pass_action, single) == 0
        assert RuleEngine.compare_actions(single, pass_action) == 0

    def test_compare_triple_single(self):
        a = Action.from_cards([7, 7, 7, 3])
        b = Action.from_cards([5, 5, 5, 14])
        assert RuleEngine.compare_actions(a, b) == 1  # 777 > 555

    def test_compare_straight(self):
        a = Action.from_cards([4, 5, 6, 7, 8])
        b = Action.from_cards([3, 4, 5, 6, 7])
        assert RuleEngine.compare_actions(a, b) == 1


class TestGetRank:
    """主牌面值获取测试"""

    def test_single_rank(self):
        assert RuleEngine.get_rank(Action.from_cards([5])) == 5

    def test_pair_rank(self):
        assert RuleEngine.get_rank(Action.from_cards([7, 7])) == 7

    def test_triple_single_rank(self):
        action = Action.from_cards([5, 5, 5, 3])
        assert RuleEngine.get_rank(action) == 5  # 三张的牌面值

    def test_straight_rank(self):
        action = Action.from_cards([3, 4, 5, 6, 7])
        assert RuleEngine.get_rank(action) == 3  # 起始牌面值

    def test_rocket_rank(self):
        action = Action.from_cards([Card.BLACK_JOKER, Card.RED_JOKER])
        assert RuleEngine.get_rank(action) == 100


class TestIsValidPlay:
    """出牌合法性验证测试"""

    def test_pass_not_valid_for_lead(self):
        pass_action = Action.pass_action()
        hand = [3, 4, 5]
        # 主动出牌不能 PASS
        assert RuleEngine.is_valid_play(pass_action, None, hand) is False

    def test_pass_valid_for_follow(self):
        pass_action = Action.pass_action()
        last = Action.from_cards([5])
        hand = [3]
        assert RuleEngine.is_valid_play(pass_action, last, hand) is True

    def test_card_not_in_hand(self):
        action = Action.from_cards([5])
        hand = [3, 4]
        assert RuleEngine.is_valid_play(action, None, hand) is False

    def test_card_count_not_enough(self):
        action = Action.from_cards([5, 5])
        hand = [5, 6]  # 只有一张 5
        assert RuleEngine.is_valid_play(action, None, hand) is False

    def test_valid_lead(self):
        action = Action.from_cards([5])
        hand = [3, 5, 7]
        assert RuleEngine.is_valid_play(action, None, hand) is True

    def test_valid_follow(self):
        action = Action.from_cards([7])
        last = Action.from_cards([5])
        hand = [3, 7, 9]
        assert RuleEngine.is_valid_play(action, last, hand) is True

    def test_cannot_beat(self):
        action = Action.from_cards([3])
        last = Action.from_cards([5])
        hand = [3, 4]
        assert RuleEngine.is_valid_play(action, last, hand) is False

    def test_wrong_type(self):
        action = Action(cards=(3, 4), action_type=ActionType.WRONG)
        hand = [3, 4]
        assert RuleEngine.is_valid_play(action, None, hand) is False


class TestCanBeat:
    """能否打过测试"""

    def test_can_beat_single(self):
        hand = [3, 5, 7]
        last = Action.from_cards([4])
        assert RuleEngine.can_beat(hand, last) is True

    def test_cannot_beat_single(self):
        hand = [3, 4]
        last = Action.from_cards([5])
        assert RuleEngine.can_beat(hand, last) is False

    def test_can_beat_with_bomb(self):
        hand = [3, 3, 3, 3]  # 炸弹
        last = Action.from_cards([Card.RED_JOKER])  # 大王
        assert RuleEngine.can_beat(hand, last) is True

    def test_beat_pass(self):
        hand = [3]
        last = Action.pass_action()
        assert RuleEngine.can_beat(hand, last) is True


class TestCalculateScore:
    """得分计算测试"""

    def test_landlord_wins_base(self):
        scores = RuleEngine.calculate_score("landlord", 1, 0, False)
        assert scores["landlord"] == 2
        assert scores["landlord_down"] == -1
        assert scores["landlord_up"] == -1

    def test_farmer_wins_base(self):
        scores = RuleEngine.calculate_score("farmer", 1, 0, False)
        assert scores["landlord"] == -2
        assert scores["landlord_down"] == 1
        assert scores["landlord_up"] == 1

    def test_bid_multiplier(self):
        scores = RuleEngine.calculate_score("landlord", 3, 0, False)
        assert scores["landlord"] == 6  # 3 * 2
        assert scores["landlord_down"] == -3

    def test_bomb_multiplier(self):
        scores = RuleEngine.calculate_score("landlord", 1, 2, False)
        # 2 bombs = 2^2 = 4x multiplier
        assert scores["landlord"] == 8  # 1 * 4 * 2
        assert scores["landlord_down"] == -4

    def test_spring_multiplier(self):
        scores = RuleEngine.calculate_score("landlord", 1, 0, True)
        assert scores["landlord"] == 4  # 1 * 2 * 2
        assert scores["landlord_down"] == -2

    def test_combined_multipliers(self):
        # 叫3分 + 1炸弹 + 春天
        scores = RuleEngine.calculate_score("landlord", 3, 1, True)
        # 3 * 2 * 2 = 12
        assert scores["landlord"] == 24
        assert scores["landlord_down"] == -12


class TestIsConsecutive:
    """连续检测测试"""

    def test_consecutive(self):
        assert RuleEngine.is_consecutive([3, 4, 5, 6, 7]) is True
        assert RuleEngine.is_consecutive([10, 11, 12, 13, 14]) is True

    def test_not_consecutive(self):
        assert RuleEngine.is_consecutive([3, 4, 6, 7, 8]) is False
        assert RuleEngine.is_consecutive([3, 5, 7, 9, 11]) is False

    def test_single(self):
        assert RuleEngine.is_consecutive([5]) is True

    def test_empty(self):
        assert RuleEngine.is_consecutive([]) is True
