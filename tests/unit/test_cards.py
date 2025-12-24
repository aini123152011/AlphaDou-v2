"""牌编码/解码测试"""
import pytest
import numpy as np

from core.cards import (
    Card,
    FULL_DECK,
    CARD_TO_STR,
    STR_TO_CARD,
    cards_to_array,
    array_to_cards,
    cards_to_str,
    str_to_cards,
    cards_to_onehot_matrix,
    is_bomb,
    is_rocket,
)


class TestCardEnum:
    """Card 枚举测试"""

    def test_card_values(self):
        assert Card.THREE == 3
        assert Card.ACE == 14
        assert Card.TWO == 17
        assert Card.BLACK_JOKER == 20
        assert Card.RED_JOKER == 30

    def test_card_ordering(self):
        assert Card.THREE < Card.FOUR < Card.ACE < Card.TWO
        assert Card.TWO < Card.BLACK_JOKER < Card.RED_JOKER


class TestFullDeck:
    """完整牌组测试"""

    def test_deck_size(self):
        assert len(FULL_DECK) == 54

    def test_deck_composition(self):
        from collections import Counter
        counter = Counter(FULL_DECK)
        # 普通牌各4张
        for card in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17]:
            assert counter[card] == 4
        # 王各1张
        assert counter[Card.BLACK_JOKER] == 1
        assert counter[Card.RED_JOKER] == 1


class TestCardsToArray:
    """cards_to_array 测试"""

    def test_empty_cards(self):
        arr = cards_to_array([])
        assert arr.shape == (54,)
        assert arr.sum() == 0

    def test_single_card(self):
        arr = cards_to_array([3])
        assert arr.shape == (54,)
        assert arr.sum() == 1
        assert arr[0] == 1  # 3 的第一张

    def test_pair(self):
        arr = cards_to_array([3, 3])
        assert arr.sum() == 2
        assert arr[0] == 1
        assert arr[1] == 1

    def test_four_of_a_kind(self):
        arr = cards_to_array([5, 5, 5, 5])
        assert arr.sum() == 4
        # 5 对应列索引 2，四张应在 [8, 9, 10, 11]
        assert arr[8:12].sum() == 4

    def test_jokers(self):
        arr = cards_to_array([Card.BLACK_JOKER, Card.RED_JOKER])
        assert arr[52] == 1  # 小王
        assert arr[53] == 1  # 大王

    def test_mixed_hand(self):
        hand = [3, 3, 4, 5, 5, 5, Card.BLACK_JOKER]
        arr = cards_to_array(hand)
        assert arr.sum() == 7


class TestArrayToCards:
    """array_to_cards 测试"""

    def test_roundtrip_empty(self):
        original = []
        arr = cards_to_array(original)
        result = array_to_cards(arr)
        assert result == original

    def test_roundtrip_single(self):
        original = [5]
        arr = cards_to_array(original)
        result = array_to_cards(arr)
        assert result == original

    def test_roundtrip_hand(self):
        original = sorted([3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 17])
        arr = cards_to_array(original)
        result = array_to_cards(arr)
        assert result == original

    def test_roundtrip_with_jokers(self):
        original = sorted([3, 4, 5, Card.BLACK_JOKER, Card.RED_JOKER])
        arr = cards_to_array(original)
        result = array_to_cards(arr)
        assert result == original


class TestCardsToOnehotMatrix:
    """cards_to_onehot_matrix 测试"""

    def test_shape(self):
        matrix = cards_to_onehot_matrix([3, 4, 5])
        assert matrix.shape == (4, 15)

    def test_joker_columns(self):
        matrix = cards_to_onehot_matrix([Card.BLACK_JOKER, Card.RED_JOKER])
        assert matrix[0, 13] == 1  # 小王
        assert matrix[0, 14] == 1  # 大王


class TestCardsStrConversion:
    """字符串转换测试"""

    def test_cards_to_str(self):
        assert cards_to_str([3, 4, 5]) == "345"
        assert cards_to_str([11, 12, 13, 14]) == "JQKA"
        assert cards_to_str([17]) == "2"
        assert cards_to_str([Card.BLACK_JOKER, Card.RED_JOKER]) == "XD"

    def test_str_to_cards(self):
        assert str_to_cards("345") == [3, 4, 5]
        assert str_to_cards("JQKA") == [11, 12, 13, 14]
        assert str_to_cards("10") == [10]
        assert str_to_cards("XD") == [20, 30]

    def test_roundtrip(self):
        original = [3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 17]
        s = cards_to_str(original)
        result = str_to_cards(s)
        assert sorted(result) == sorted(original)


class TestIsBomb:
    """炸弹检测测试"""

    def test_four_of_a_kind(self):
        assert is_bomb([5, 5, 5, 5]) is True
        assert is_bomb([14, 14, 14, 14]) is True

    def test_rocket(self):
        assert is_bomb([Card.BLACK_JOKER, Card.RED_JOKER]) is True

    def test_not_bomb(self):
        assert is_bomb([3, 3, 3]) is False
        assert is_bomb([3, 4, 5, 6]) is False
        assert is_bomb([Card.BLACK_JOKER]) is False


class TestIsRocket:
    """王炸检测测试"""

    def test_rocket(self):
        assert is_rocket([Card.BLACK_JOKER, Card.RED_JOKER]) is True
        assert is_rocket([Card.RED_JOKER, Card.BLACK_JOKER]) is True

    def test_not_rocket(self):
        assert is_rocket([5, 5, 5, 5]) is False
        assert is_rocket([Card.BLACK_JOKER]) is False
