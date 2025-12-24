"""
牌的定义与编码

斗地主使用 54 张牌：
- 3-10, J, Q, K, A, 2 各 4 张
- 小王、大王各 1 张
"""
from enum import IntEnum
from typing import List, Tuple, Dict
from collections import Counter
import numpy as np


class Card(IntEnum):
    """牌面值定义 (与原项目保持兼容)"""
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14
    TWO = 17
    BLACK_JOKER = 20
    RED_JOKER = 30


# 牌面值到显示字符的映射
CARD_TO_STR: Dict[int, str] = {
    3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
    8: '8', 9: '9', 10: '10', 11: 'J', 12: 'Q',
    13: 'K', 14: 'A', 17: '2', 20: 'X', 30: 'D'
}

# 显示字符到牌面值的映射
STR_TO_CARD: Dict[str, int] = {v: k for k, v in CARD_TO_STR.items()}

# 牌面值到数组列索引的映射 (用于 one-hot 编码)
CARD_TO_COLUMN: Dict[int, int] = {
    3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6,
    10: 7, 11: 8, 12: 9, 13: 10, 14: 11, 17: 12
}

# 完整牌组 (54 张)
FULL_DECK: Tuple[int, ...] = (
    3, 3, 3, 3,
    4, 4, 4, 4,
    5, 5, 5, 5,
    6, 6, 6, 6,
    7, 7, 7, 7,
    8, 8, 8, 8,
    9, 9, 9, 9,
    10, 10, 10, 10,
    11, 11, 11, 11,
    12, 12, 12, 12,
    13, 13, 13, 13,
    14, 14, 14, 14,
    17, 17, 17, 17,
    20, 30
)

# 所有炸弹牌型
BOMBS: Tuple[Tuple[int, ...], ...] = (
    (3, 3, 3, 3), (4, 4, 4, 4), (5, 5, 5, 5), (6, 6, 6, 6),
    (7, 7, 7, 7), (8, 8, 8, 8), (9, 9, 9, 9), (10, 10, 10, 10),
    (11, 11, 11, 11), (12, 12, 12, 12), (13, 13, 13, 13),
    (14, 14, 14, 14), (17, 17, 17, 17),
    (20, 30),  # 王炸
)

# 数量到 one-hot 数组的映射
NUM_TO_ONEHOT: Dict[int, Tuple[int, ...]] = {
    0: (0, 0, 0, 0),
    1: (1, 0, 0, 0),
    2: (1, 1, 0, 0),
    3: (1, 1, 1, 0),
    4: (1, 1, 1, 1),
}


def cards_to_array(cards: List[int]) -> np.ndarray:
    """
    将牌列表转换为 54 维 one-hot 向量

    编码方式 (参考 DouZero 论文):
    - 前 52 维: 13 种牌面 × 4 张牌 (按列展开)
    - 后 2 维: [小王, 大王]

    Args:
        cards: 牌列表，元素为牌面值

    Returns:
        54 维 numpy 数组
    """
    if not cards:
        return np.zeros(54, dtype=np.float32)

    matrix = np.zeros((4, 13), dtype=np.float32)
    jokers = np.zeros(2, dtype=np.float32)

    counter = Counter(cards)
    for card, count in counter.items():
        if card < 20:
            col = CARD_TO_COLUMN[card]
            for i in range(count):
                matrix[i, col] = 1
        elif card == Card.BLACK_JOKER:
            jokers[0] = 1
        elif card == Card.RED_JOKER:
            jokers[1] = 1

    # 按列展开 (F order) 并拼接王
    return np.concatenate([matrix.flatten('F'), jokers])


def cards_to_onehot_matrix(cards: List[int]) -> np.ndarray:
    """
    将牌列表转换为 4×15 的 one-hot 矩阵

    适用于卷积网络输入

    Args:
        cards: 牌列表

    Returns:
        (4, 15) numpy 数组，其中列 0-12 为 3-2，列 13-14 为小王大王
    """
    matrix = np.zeros((4, 15), dtype=np.float32)
    counter = Counter(cards)

    for card, count in counter.items():
        if card < 20:
            col = CARD_TO_COLUMN[card]
            for i in range(count):
                matrix[i, col] = 1
        elif card == Card.BLACK_JOKER:
            matrix[0, 13] = 1
        elif card == Card.RED_JOKER:
            matrix[0, 14] = 1

    return matrix


def array_to_cards(array: np.ndarray) -> List[int]:
    """
    将 54 维数组转换回牌列表

    Args:
        array: 54 维 numpy 数组

    Returns:
        牌列表
    """
    cards = []

    # 解码前 52 维 (4×13 矩阵按列展开)
    matrix = array[:52].reshape((4, 13), order='F')
    for col in range(13):
        # 找到对应的牌面值
        card_value = [k for k, v in CARD_TO_COLUMN.items() if v == col][0]
        count = int(matrix[:, col].sum())
        cards.extend([card_value] * count)

    # 解码王
    if array[52] > 0:
        cards.append(Card.BLACK_JOKER)
    if array[53] > 0:
        cards.append(Card.RED_JOKER)

    return sorted(cards)


def cards_to_str(cards: List[int]) -> str:
    """
    将牌列表转换为可读字符串

    Args:
        cards: 牌列表

    Returns:
        如 "34567" 或 "JQKA2XD"
    """
    return ''.join(CARD_TO_STR.get(c, '?') for c in sorted(cards))


def str_to_cards(s: str) -> List[int]:
    """
    将字符串转换为牌列表

    Args:
        s: 牌字符串，如 "34567"

    Returns:
        牌列表
    """
    cards = []
    i = 0
    while i < len(s):
        if i + 1 < len(s) and s[i:i+2] == '10':
            cards.append(10)
            i += 2
        else:
            cards.append(STR_TO_CARD[s[i]])
            i += 1
    return cards


def is_bomb(cards: List[int]) -> bool:
    """检查是否为炸弹 (四张相同或王炸)"""
    if len(cards) == 4 and len(set(cards)) == 1:
        return True
    if len(cards) == 2 and set(cards) == {Card.BLACK_JOKER, Card.RED_JOKER}:
        return True
    return False


def is_rocket(cards: List[int]) -> bool:
    """检查是否为王炸"""
    return set(cards) == {Card.BLACK_JOKER, Card.RED_JOKER}
