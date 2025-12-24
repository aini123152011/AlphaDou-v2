"""动作生成测试"""
import pytest
from collections import Counter

from core.cards import Card
from core.actions import (
    ActionType,
    Action,
    ActionGenerator,
    MIN_STRAIGHT_LEN,
    MIN_STRAIGHT_PAIR_LEN,
    MIN_AIRPLANE_LEN,
)


class TestActionType:
    """ActionType 枚举测试"""

    def test_action_types_count(self):
        # 15种牌型 + WRONG
        assert len(ActionType) == 16

    def test_pass_is_zero(self):
        assert ActionType.PASS == 0


class TestAction:
    """Action 数据类测试"""

    def test_pass_action(self):
        action = Action.pass_action()
        assert action.is_pass is True
        assert action.cards == ()
        assert action.action_type == ActionType.PASS

    def test_from_cards_single(self):
        action = Action.from_cards([5])
        assert action.cards == (5,)
        assert action.action_type == ActionType.SINGLE

    def test_from_cards_pair(self):
        action = Action.from_cards([7, 7])
        assert action.cards == (7, 7)
        assert action.action_type == ActionType.PAIR

    def test_from_cards_bomb(self):
        action = Action.from_cards([8, 8, 8, 8])
        assert action.action_type == ActionType.BOMB
        assert action.is_bomb is True

    def test_from_cards_rocket(self):
        action = Action.from_cards([Card.BLACK_JOKER, Card.RED_JOKER])
        assert action.action_type == ActionType.ROCKET
        assert action.is_bomb is True

    def test_immutability(self):
        action = Action.from_cards([5])
        with pytest.raises(Exception):
            action.cards = (6,)

    def test_len(self):
        assert len(Action.from_cards([5])) == 1
        assert len(Action.from_cards([5, 5])) == 2
        assert len(Action.pass_action()) == 0


class TestActionGeneratorBasic:
    """ActionGenerator 基础牌型测试"""

    def test_gen_singles(self):
        gen = ActionGenerator([3, 4, 5])
        singles = gen.gen_singles()
        assert len(singles) == 3
        assert [3] in singles
        assert [4] in singles
        assert [5] in singles

    def test_gen_singles_with_jokers(self):
        gen = ActionGenerator([3, Card.BLACK_JOKER, Card.RED_JOKER])
        singles = gen.gen_singles()
        assert [Card.BLACK_JOKER] in singles
        assert [Card.RED_JOKER] in singles

    def test_gen_pairs(self):
        gen = ActionGenerator([3, 3, 4, 4, 5])
        pairs = gen.gen_pairs()
        assert len(pairs) == 2
        assert [3, 3] in pairs
        assert [4, 4] in pairs

    def test_gen_triples(self):
        gen = ActionGenerator([3, 3, 3, 4, 4])
        triples = gen.gen_triples()
        assert len(triples) == 1
        assert [3, 3, 3] in triples

    def test_gen_bombs(self):
        gen = ActionGenerator([3, 3, 3, 3, 4, 4, 4, 4])
        bombs = gen.gen_bombs()
        assert len(bombs) == 2

    def test_gen_rocket(self):
        gen = ActionGenerator([3, Card.BLACK_JOKER, Card.RED_JOKER])
        rockets = gen.gen_rocket()
        assert len(rockets) == 1
        assert rockets[0] == [Card.BLACK_JOKER, Card.RED_JOKER]

    def test_gen_rocket_missing_joker(self):
        gen = ActionGenerator([3, Card.BLACK_JOKER])
        rockets = gen.gen_rocket()
        assert len(rockets) == 0


class TestActionGeneratorCombo:
    """ActionGenerator 组合牌型测试"""

    def test_gen_triple_single(self):
        gen = ActionGenerator([3, 3, 3, 4, 5])
        combos = gen.gen_triple_single()
        assert len(combos) == 2  # 333+4, 333+5
        for combo in combos:
            assert len(combo) == 4
            assert combo.count(3) == 3

    def test_gen_triple_pair(self):
        gen = ActionGenerator([3, 3, 3, 4, 4])
        combos = gen.gen_triple_pair()
        assert len(combos) == 1
        assert sorted(combos[0]) == [3, 3, 3, 4, 4]

    def test_gen_quad_single(self):
        gen = ActionGenerator([3, 3, 3, 3, 4, 5])
        combos = gen.gen_quad_single()
        assert len(combos) == 1  # 3333 + 4,5
        assert len(combos[0]) == 6

    def test_gen_quad_pair(self):
        gen = ActionGenerator([3, 3, 3, 3, 4, 4, 5, 5])
        combos = gen.gen_quad_pair()
        assert len(combos) == 1  # 3333 + 44 + 55
        assert len(combos[0]) == 8


class TestActionGeneratorSerial:
    """ActionGenerator 连续牌型测试"""

    def test_gen_straight_min(self):
        gen = ActionGenerator([3, 4, 5, 6, 7])
        straights = gen.gen_straight()
        assert len(straights) == 1
        assert straights[0] == [3, 4, 5, 6, 7]

    def test_gen_straight_longer(self):
        gen = ActionGenerator([3, 4, 5, 6, 7, 8, 9])
        straights = gen.gen_straight()
        # 长度5: 34567, 45678, 56789
        # 长度6: 345678, 456789
        # 长度7: 3456789
        assert len(straights) == 6

    def test_gen_straight_no_two(self):
        # 2 不能参与顺子
        gen = ActionGenerator([11, 12, 13, 14, 17])  # JQKA2
        straights = gen.gen_straight()
        assert len(straights) == 0  # 缺10，无法形成顺子

    def test_gen_straight_pair(self):
        gen = ActionGenerator([3, 3, 4, 4, 5, 5])
        pairs = gen.gen_straight_pair()
        assert len(pairs) == 1
        assert pairs[0] == [3, 3, 4, 4, 5, 5]

    def test_gen_airplane(self):
        gen = ActionGenerator([3, 3, 3, 4, 4, 4])
        airplanes = gen.gen_airplane()
        assert len(airplanes) == 1
        assert airplanes[0] == [3, 3, 3, 4, 4, 4]

    def test_gen_airplane_single(self):
        gen = ActionGenerator([3, 3, 3, 4, 4, 4, 5, 6])
        combos = gen.gen_airplane_single()
        assert len(combos) == 1  # 333444 + 5,6
        assert len(combos[0]) == 8

    def test_gen_airplane_pair(self):
        gen = ActionGenerator([3, 3, 3, 4, 4, 4, 5, 5, 6, 6])
        combos = gen.gen_airplane_pair()
        assert len(combos) == 1  # 333444 + 55,66
        assert len(combos[0]) == 10


class TestActionGeneratorGenerateAll:
    """generate_all 测试"""

    def test_simple_hand(self):
        gen = ActionGenerator([3, 4, 5])
        actions = gen.generate_all()
        # 3个单张
        singles = [a for a in actions if a.action_type == ActionType.SINGLE]
        assert len(singles) == 3

    def test_includes_bombs(self):
        gen = ActionGenerator([3, 3, 3, 3, 4])
        actions = gen.generate_all()
        bombs = [a for a in actions if a.action_type == ActionType.BOMB]
        assert len(bombs) == 1


class TestActionGeneratorResponses:
    """generate_responses 测试"""

    def test_response_to_pass(self):
        gen = ActionGenerator([3, 4, 5])
        responses = gen.generate_responses(Action.pass_action())
        # PASS 相当于主动出牌
        assert len(responses) == len(gen.generate_all())

    def test_response_to_single(self):
        gen = ActionGenerator([3, 4, 5, 6])
        last = Action.from_cards([4])
        responses = gen.generate_responses(last)
        # 可以 PASS，或出 5, 6
        assert Action.pass_action() in responses
        singles = [r for r in responses if r.action_type == ActionType.SINGLE]
        assert len(singles) == 2  # 5, 6

    def test_response_to_pair(self):
        gen = ActionGenerator([3, 3, 5, 5, 7, 7])
        last = Action.from_cards([5, 5])
        responses = gen.generate_responses(last)
        pairs = [r for r in responses if r.action_type == ActionType.PAIR]
        assert len(pairs) == 1  # 只有 77 能打过 55

    def test_response_to_bomb(self):
        gen = ActionGenerator([5, 5, 5, 5, 8, 8, 8, 8])
        last = Action.from_cards([5, 5, 5, 5])
        responses = gen.generate_responses(last)
        bombs = [r for r in responses if r.action_type == ActionType.BOMB]
        assert len(bombs) == 1  # 只有 8888 能打过 5555

    def test_bomb_beats_non_bomb(self):
        gen = ActionGenerator([3, 3, 3, 3])
        last = Action.from_cards([14])  # A
        responses = gen.generate_responses(last)
        # 没有更大的单张，但有炸弹
        bombs = [r for r in responses if r.action_type == ActionType.BOMB]
        assert len(bombs) == 1

    def test_rocket_beats_bomb(self):
        gen = ActionGenerator([3, Card.BLACK_JOKER, Card.RED_JOKER])
        last = Action.from_cards([14, 14, 14, 14])  # AAAA 炸弹
        responses = gen.generate_responses(last)
        rockets = [r for r in responses if r.action_type == ActionType.ROCKET]
        assert len(rockets) == 1

    def test_cannot_beat_rocket(self):
        gen = ActionGenerator([3, 3, 3, 3, 17, 17, 17, 17])  # 有两个炸弹
        last = Action.from_cards([Card.BLACK_JOKER, Card.RED_JOKER])  # 王炸
        responses = gen.generate_responses(last)
        # 只能 PASS
        assert len(responses) == 1
        assert responses[0].is_pass
