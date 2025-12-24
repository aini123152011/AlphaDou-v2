"""
Core Layer - 纯游戏逻辑 (无 ML 依赖)

Modules:
    cards: 牌定义与编码
    actions: 动作类型与生成
    rules: 规则引擎
    state: 游戏状态
"""
from .cards import (
    Card,
    FULL_DECK,
    BOMBS,
    CARD_TO_STR,
    STR_TO_CARD,
    cards_to_array,
    array_to_cards,
    cards_to_str,
    str_to_cards,
    is_bomb,
    is_rocket,
)

from .actions import (
    ActionType,
    Action,
    ActionGenerator,
    MIN_STRAIGHT_LEN,
    MIN_STRAIGHT_PAIR_LEN,
    MIN_AIRPLANE_LEN,
)

from .rules import RuleEngine

from .state import (
    Phase,
    Role,
    GameState,
    BID_ORDER,
    PLAY_ORDER,
)

__all__ = [
    # cards
    "Card",
    "FULL_DECK",
    "BOMBS",
    "CARD_TO_STR",
    "STR_TO_CARD",
    "cards_to_array",
    "array_to_cards",
    "cards_to_str",
    "str_to_cards",
    "is_bomb",
    "is_rocket",
    # actions
    "ActionType",
    "Action",
    "ActionGenerator",
    "MIN_STRAIGHT_LEN",
    "MIN_STRAIGHT_PAIR_LEN",
    "MIN_AIRPLANE_LEN",
    # rules
    "RuleEngine",
    # state
    "Phase",
    "Role",
    "GameState",
    "BID_ORDER",
    "PLAY_ORDER",
]
