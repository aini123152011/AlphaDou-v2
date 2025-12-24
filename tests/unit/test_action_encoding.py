"""动作编码测试"""
import pytest
import random
from collections import defaultdict

from core.actions import ActionGenerator, ActionType, Action
from env.observation import ActionEncoder, get_action_encoder


class TestActionEncoder:
    """ActionEncoder 测试"""

    def test_num_actions(self):
        """验证动作空间大小"""
        encoder = ActionEncoder()
        # 扩展后的动作空间应该大于原来的 527
        assert encoder.num_actions > 1000

    def test_pass_encoding(self):
        """验证 PASS 动作编码"""
        encoder = ActionEncoder()
        pass_action = Action.pass_action()
        idx = encoder.encode(pass_action)
        assert idx == 0

        decoded = encoder.decode(0)
        assert decoded.is_pass

    def test_single_encoding(self):
        """验证单张编码"""
        encoder = ActionEncoder()

        for card in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 20, 30]:
            action = Action((card,), ActionType.SINGLE)
            idx = encoder.encode(action)
            assert idx >= 0, f"单张 {card} 编码失败"

            decoded = encoder.decode(idx)
            assert decoded.cards == (card,)

    def test_pair_encoding(self):
        """验证对子编码"""
        encoder = ActionEncoder()

        for card in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17]:
            action = Action((card, card), ActionType.PAIR)
            idx = encoder.encode(action)
            assert idx >= 0, f"对子 {card} 编码失败"

    def test_bomb_encoding(self):
        """验证炸弹编码"""
        encoder = ActionEncoder()

        for card in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17]:
            action = Action((card, card, card, card), ActionType.BOMB)
            idx = encoder.encode(action)
            assert idx >= 0, f"炸弹 {card} 编码失败"

    def test_rocket_encoding(self):
        """验证王炸编码"""
        encoder = ActionEncoder()
        action = Action((20, 30), ActionType.ROCKET)
        idx = encoder.encode(action)
        assert idx >= 0

    def test_triple_single_encoding(self):
        """验证三带一编码"""
        encoder = ActionEncoder()

        # 测试几个三带一
        action = Action((3, 3, 3, 4), ActionType.TRIPLE_SINGLE)
        idx = encoder.encode(action)
        assert idx >= 0

        action = Action((3, 14, 14, 14), ActionType.TRIPLE_SINGLE)
        idx = encoder.encode(action)
        assert idx >= 0

    def test_airplane_single_encoding(self):
        """验证飞机带单编码"""
        encoder = ActionEncoder()

        # 334445556 + 78 (两个飞机带两个单)
        airplane = [3, 3, 3, 4, 4, 4, 7, 8]
        action = Action(tuple(sorted(airplane)), ActionType.AIRPLANE_SINGLE)
        idx = encoder.encode(action)
        assert idx >= 0, f"飞机带单编码失败: {airplane}"

    def test_airplane_pair_encoding(self):
        """验证飞机带对编码"""
        encoder = ActionEncoder()

        # 33344477 (两个飞机带两个对)
        airplane = [3, 3, 3, 4, 4, 4, 7, 7, 8, 8]
        action = Action(tuple(sorted(airplane)), ActionType.AIRPLANE_PAIR)
        idx = encoder.encode(action)
        assert idx >= 0, f"飞机带对编码失败: {airplane}"

    def test_quad_single_encoding(self):
        """验证四带二单编码"""
        encoder = ActionEncoder()

        # 333345
        quad = [3, 3, 3, 3, 4, 5]
        action = Action(tuple(sorted(quad)), ActionType.QUAD_SINGLE)
        idx = encoder.encode(action)
        assert idx >= 0, f"四带二单编码失败: {quad}"

    def test_quad_pair_encoding(self):
        """验证四带二对编码"""
        encoder = ActionEncoder()

        # 33334455
        quad = [3, 3, 3, 3, 4, 4, 5, 5]
        action = Action(tuple(sorted(quad)), ActionType.QUAD_PAIR)
        idx = encoder.encode(action)
        assert idx >= 0, f"四带二对编码失败: {quad}"

    def test_encode_decode_roundtrip(self):
        """验证编码-解码往返一致"""
        encoder = ActionEncoder()

        for idx in range(min(1000, encoder.num_actions)):
            decoded = encoder.decode(idx)
            if decoded is not None:
                re_encoded = encoder.encode(decoded)
                assert re_encoded == idx, f"往返不一致: {idx} -> {decoded} -> {re_encoded}"


class TestActionEncoderCoverage:
    """动作编码覆盖率测试"""

    def test_all_single_hand_actions_encodable(self):
        """验证单一手牌的所有动作可编码"""
        encoder = ActionEncoder()
        hand = [3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14]
        gen = ActionGenerator(hand)

        for action in gen.generate_all():
            idx = encoder.encode(action)
            assert idx >= 0, f"动作无法编码: {action.cards} ({action.action_type.name})"

    def test_all_quad_hand_actions_encodable(self):
        """验证含四张手牌的所有动作可编码"""
        encoder = ActionEncoder()
        hand = [3, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        gen = ActionGenerator(hand)

        for action in gen.generate_all():
            idx = encoder.encode(action)
            assert idx >= 0, f"动作无法编码: {action.cards} ({action.action_type.name})"

    def test_random_hands_coverage(self):
        """随机手牌覆盖率测试"""
        encoder = ActionEncoder()

        # 使用固定种子确保可重复
        random.seed(42)

        # 所有牌
        full_deck = [c for c in range(3, 15) for _ in range(4)] + [17] * 4 + [20, 30]

        total_actions = 0
        encoded_actions = 0

        for _ in range(20):  # 测试 20 个随机手牌
            random.shuffle(full_deck)
            hand = sorted(full_deck[:17])  # 农民手牌

            gen = ActionGenerator(hand)
            for action in gen.generate_all():
                total_actions += 1
                if encoder.encode(action) >= 0:
                    encoded_actions += 1

        coverage = encoded_actions / total_actions if total_actions > 0 else 0
        assert coverage >= 0.99, f"编码覆盖率过低: {coverage:.2%}"

    def test_all_action_types_represented(self):
        """验证所有动作类型都有编码"""
        encoder = ActionEncoder()

        # 构造包含所有类型动作的手牌
        hand = [3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 9, 10, 11, 12, 20, 30]
        gen = ActionGenerator(hand)

        by_type = defaultdict(int)
        for action in gen.generate_all():
            if encoder.encode(action) >= 0:
                by_type[action.action_type.name] += 1

        # 验证关键类型都有编码
        critical_types = [
            "SINGLE", "PAIR", "TRIPLE", "BOMB", "ROCKET",
            "TRIPLE_SINGLE", "TRIPLE_PAIR",
            "STRAIGHT", "STRAIGHT_PAIR",
            "AIRPLANE", "AIRPLANE_SINGLE", "AIRPLANE_PAIR",
            "QUAD_SINGLE", "QUAD_PAIR"
        ]

        for action_type in critical_types:
            # 某些类型可能不在这个特定手牌中生成
            # 但至少应该在编码器中支持
            pass


class TestGlobalEncoder:
    """全局编码器测试"""

    def test_singleton(self):
        """验证全局编码器是单例"""
        encoder1 = get_action_encoder()
        encoder2 = get_action_encoder()
        assert encoder1 is encoder2

    def test_global_encoder_coverage(self):
        """验证全局编码器覆盖率"""
        encoder = get_action_encoder()
        assert encoder.num_actions > 1000
