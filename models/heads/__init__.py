"""
Heads 输出头

提供策略、价值、叫牌等输出层
"""
from .policy import PolicyHead, DuelingPolicyHead, AttentionPolicyHead
from .value import ValueHead, DistributionalValueHead, MultiHeadValueHead
from .bid import BidHead, BidValueHead

__all__ = [
    "PolicyHead",
    "DuelingPolicyHead",
    "AttentionPolicyHead",
    "ValueHead",
    "DistributionalValueHead",
    "MultiHeadValueHead",
    "BidHead",
    "BidValueHead",
]
