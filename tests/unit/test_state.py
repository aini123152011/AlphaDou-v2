"""游戏状态测试"""
import pytest

from core.cards import Card
from core.actions import Action, ActionType
from core.state import (
    Phase,
    Role,
    GameState,
    BID_ORDER,
    PLAY_ORDER,
    get_play_roles,
)


class TestPhaseEnum:
    """Phase 枚举测试"""

    def test_phases(self):
        assert Phase.BIDDING.value == "bidding"
        assert Phase.PLAYING.value == "playing"
        assert Phase.FINISHED.value == "finished"


class TestRoleEnum:
    """Role 枚举测试"""

    def test_bid_roles(self):
        assert Role.FIRST.value == "first"
        assert Role.SECOND.value == "second"
        assert Role.THIRD.value == "third"

    def test_play_roles(self):
        assert Role.LANDLORD.value == "landlord"
        assert Role.LANDLORD_DOWN.value == "landlord_down"
        assert Role.LANDLORD_UP.value == "landlord_up"


class TestBidOrder:
    """叫牌顺序测试"""

    def test_bid_order(self):
        assert BID_ORDER == (Role.FIRST, Role.SECOND, Role.THIRD)


class TestPlayOrder:
    """出牌顺序测试"""

    def test_play_order(self):
        assert PLAY_ORDER == (Role.LANDLORD, Role.LANDLORD_DOWN, Role.LANDLORD_UP)


class TestGetPlayRoles:
    """角色映射测试"""

    def test_first_becomes_landlord(self):
        mapping = get_play_roles(Role.FIRST)
        assert mapping[Role.FIRST] == Role.LANDLORD
        assert mapping[Role.SECOND] == Role.LANDLORD_DOWN
        assert mapping[Role.THIRD] == Role.LANDLORD_UP

    def test_second_becomes_landlord(self):
        mapping = get_play_roles(Role.SECOND)
        assert mapping[Role.SECOND] == Role.LANDLORD
        assert mapping[Role.THIRD] == Role.LANDLORD_DOWN
        assert mapping[Role.FIRST] == Role.LANDLORD_UP

    def test_third_becomes_landlord(self):
        mapping = get_play_roles(Role.THIRD)
        assert mapping[Role.THIRD] == Role.LANDLORD
        assert mapping[Role.FIRST] == Role.LANDLORD_DOWN
        assert mapping[Role.SECOND] == Role.LANDLORD_UP


class TestGameStateInitial:
    """GameState 初始化测试"""

    def test_initial_state(self):
        state = GameState.initial(seed=42)
        assert state.phase == Phase.BIDDING
        assert state.current_player == Role.FIRST

    def test_initial_hands(self):
        state = GameState.initial(seed=42)
        hands = state.get_hands_dict()
        assert len(hands[Role.FIRST.value]) == 17
        assert len(hands[Role.SECOND.value]) == 17
        assert len(hands[Role.THIRD.value]) == 17

    def test_initial_three_cards(self):
        state = GameState.initial(seed=42)
        assert len(state.three_cards) == 3

    def test_total_cards(self):
        state = GameState.initial(seed=42)
        hands = state.get_hands_dict()
        total = (
            len(hands[Role.FIRST.value]) +
            len(hands[Role.SECOND.value]) +
            len(hands[Role.THIRD.value]) +
            len(state.three_cards)
        )
        assert total == 54

    def test_deterministic_with_seed(self):
        state1 = GameState.initial(seed=123)
        state2 = GameState.initial(seed=123)
        assert state1.hands == state2.hands
        assert state1.three_cards == state2.three_cards


class TestGameStateBidding:
    """叫牌阶段测试"""

    def test_first_bid(self):
        state = GameState.initial(seed=42)
        new_state = state.with_bid(1)
        assert new_state.bid_info[0] == 1
        assert new_state.current_player == Role.SECOND

    def test_all_pass_draw(self):
        state = GameState.initial(seed=42)
        state = state.with_bid(0)  # FIRST 不叫
        state = state.with_bid(0)  # SECOND 不叫
        state = state.with_bid(0)  # THIRD 不叫
        assert state.phase == Phase.FINISHED
        assert state.winner == "draw"

    def test_first_wins_bid(self):
        state = GameState.initial(seed=42)
        state = state.with_bid(3)  # FIRST 叫3分
        assert state.phase == Phase.PLAYING
        assert state.landlord == Role.LANDLORD
        assert state.landlord_bid_position == Role.FIRST

    def test_second_wins_bid(self):
        state = GameState.initial(seed=42)
        state = state.with_bid(1)  # FIRST 叫1分
        state = state.with_bid(3)  # SECOND 叫3分
        assert state.phase == Phase.PLAYING
        assert state.landlord_bid_position == Role.SECOND

    def test_highest_bid_wins(self):
        state = GameState.initial(seed=42)
        state = state.with_bid(1)  # FIRST
        state = state.with_bid(2)  # SECOND
        state = state.with_bid(0)  # THIRD 不叫
        assert state.phase == Phase.PLAYING
        assert state.landlord_bid_position == Role.SECOND
        assert state.bid_multiplier == 2

    def test_landlord_gets_three_cards(self):
        state = GameState.initial(seed=42)
        three_cards = state.three_cards
        state = state.with_bid(3)
        landlord_hand = state.get_hand(Role.LANDLORD)
        assert len(landlord_hand) == 20
        for card in three_cards:
            assert card in landlord_hand

    def test_bid_history(self):
        state = GameState.initial(seed=42)
        state = state.with_bid(1)
        state = state.with_bid(2)
        assert len(state.bid_history) == 2
        assert state.bid_history[0] == (Role.FIRST, 1)
        assert state.bid_history[1] == (Role.SECOND, 2)


class TestGameStatePlaying:
    """出牌阶段测试"""

    @pytest.fixture
    def playing_state(self):
        """创建出牌阶段的状态"""
        state = GameState.initial(seed=42)
        return state.with_bid(3)

    def test_play_removes_cards(self, playing_state):
        from collections import Counter
        hand = list(playing_state.get_hand(Role.LANDLORD))
        card = hand[0]
        action = Action.from_cards([card])
        new_state = playing_state.with_play(action)
        new_hand = new_state.get_hand(Role.LANDLORD)
        assert len(new_hand) == len(hand) - 1
        # 检查该牌的数量减少了1
        old_count = Counter(hand)[card]
        new_count = Counter(new_hand)[card]
        assert new_count == old_count - 1

    def test_play_updates_current_player(self, playing_state):
        hand = list(playing_state.get_hand(Role.LANDLORD))
        action = Action.from_cards([hand[0]])
        new_state = playing_state.with_play(action)
        assert new_state.current_player == Role.LANDLORD_DOWN

    def test_pass_keeps_cards(self, playing_state):
        # 先出一张牌
        hand = list(playing_state.get_hand(Role.LANDLORD))
        action = Action.from_cards([hand[0]])
        state = playing_state.with_play(action)

        # LANDLORD_DOWN PASS
        down_hand = state.get_hand(Role.LANDLORD_DOWN)
        state = state.with_play(Action.pass_action())
        assert state.get_hand(Role.LANDLORD_DOWN) == down_hand

    def test_last_action_tracking(self, playing_state):
        hand = list(playing_state.get_hand(Role.LANDLORD))
        action = Action.from_cards([hand[0]])
        new_state = playing_state.with_play(action)
        assert new_state.last_action == action
        assert new_state.last_player == Role.LANDLORD

    def test_pass_preserves_last_action(self, playing_state):
        hand = list(playing_state.get_hand(Role.LANDLORD))
        action = Action.from_cards([hand[0]])
        state = playing_state.with_play(action)
        state = state.with_play(Action.pass_action())
        assert state.last_action == action
        assert state.last_player == Role.LANDLORD

    def test_bomb_count(self, playing_state):
        assert playing_state.bombs_count == 0
        # 假设有炸弹
        bomb_action = Action(cards=(5, 5, 5, 5), action_type=ActionType.BOMB)
        # 直接测试状态更新逻辑
        state = GameState(
            hands=(
                (Role.LANDLORD.value, (5, 5, 5, 5, 6)),
                (Role.LANDLORD_DOWN.value, (3, 4)),
                (Role.LANDLORD_UP.value, (7, 8)),
            ),
            three_cards=(),
            phase=Phase.PLAYING,
            current_player=Role.LANDLORD,
            landlord=Role.LANDLORD,
        )
        new_state = state.with_play(bomb_action)
        assert new_state.bombs_count == 1

    def test_play_history(self, playing_state):
        hand = list(playing_state.get_hand(Role.LANDLORD))
        action = Action.from_cards([hand[0]])
        new_state = playing_state.with_play(action)
        assert len(new_state.play_history) == 1
        assert new_state.play_history[0][0] == Role.LANDLORD

    def test_step_count(self, playing_state):
        assert playing_state.step_count == 0
        hand = list(playing_state.get_hand(Role.LANDLORD))
        action = Action.from_cards([hand[0]])
        new_state = playing_state.with_play(action)
        assert new_state.step_count == 1


class TestGameStateWinning:
    """游戏结束测试"""

    def test_landlord_wins(self):
        state = GameState(
            hands=(
                (Role.LANDLORD.value, (5,)),  # 只剩一张
                (Role.LANDLORD_DOWN.value, (3, 4)),
                (Role.LANDLORD_UP.value, (7, 8)),
            ),
            three_cards=(),
            phase=Phase.PLAYING,
            current_player=Role.LANDLORD,
            landlord=Role.LANDLORD,
        )
        action = Action.from_cards([5])
        new_state = state.with_play(action)
        assert new_state.phase == Phase.FINISHED
        assert new_state.winner == "landlord"

    def test_farmer_wins(self):
        state = GameState(
            hands=(
                (Role.LANDLORD.value, (3, 4)),
                (Role.LANDLORD_DOWN.value, (5,)),  # 只剩一张
                (Role.LANDLORD_UP.value, (7, 8)),
            ),
            three_cards=(),
            phase=Phase.PLAYING,
            current_player=Role.LANDLORD_DOWN,
            landlord=Role.LANDLORD,
        )
        action = Action.from_cards([5])
        new_state = state.with_play(action)
        assert new_state.phase == Phase.FINISHED
        assert new_state.winner == "farmer"


class TestGameStateLegalActions:
    """合法动作获取测试"""

    def test_bidding_legal_actions(self):
        state = GameState.initial(seed=42)
        actions = state.get_legal_actions()
        assert 0 in actions  # 不叫
        assert 1 in actions
        assert 2 in actions
        assert 3 in actions

    def test_bidding_after_bid(self):
        state = GameState.initial(seed=42)
        state = state.with_bid(2)
        actions = state.get_legal_actions()
        assert 0 in actions  # 不叫
        assert 1 not in actions  # 不能叫比 2 小的
        assert 2 not in actions
        assert 3 in actions

    def test_playing_legal_actions(self):
        state = GameState.initial(seed=42).with_bid(3)
        actions = state.get_legal_actions()
        # 主动出牌，应该有很多选择
        assert len(actions) > 0
        # 不应该有 PASS
        assert not any(a.is_pass for a in actions)


class TestGameStateImmutability:
    """不可变性测试"""

    def test_state_is_frozen(self):
        state = GameState.initial(seed=42)
        with pytest.raises(Exception):
            state.phase = Phase.PLAYING

    def test_with_bid_returns_new_state(self):
        state = GameState.initial(seed=42)
        new_state = state.with_bid(1)
        assert state is not new_state
        assert state.phase == Phase.BIDDING
        assert state.bid_info == (-1, -1, -1)


class TestGameStateSpring:
    """春天检测测试"""

    def test_not_spring_during_game(self):
        state = GameState.initial(seed=42).with_bid(3)
        assert state.is_spring() is False

    def test_landlord_spring(self):
        # 地主赢，农民未出过牌
        state = GameState(
            hands=(
                (Role.LANDLORD.value, ()),
                (Role.LANDLORD_DOWN.value, (3, 4)),
                (Role.LANDLORD_UP.value, (7, 8)),
            ),
            three_cards=(),
            phase=Phase.FINISHED,
            current_player=Role.LANDLORD,
            landlord=Role.LANDLORD,
            play_history=(
                (Role.LANDLORD, (5,)),
                (Role.LANDLORD, (6,)),
            ),
            winner="landlord",
        )
        assert state.is_spring() is True

    def test_farmer_spring(self):
        # 农民赢，地主未出过牌（理论上不可能，但测试逻辑）
        state = GameState(
            hands=(
                (Role.LANDLORD.value, (3, 4)),
                (Role.LANDLORD_DOWN.value, ()),
                (Role.LANDLORD_UP.value, (7, 8)),
            ),
            three_cards=(),
            phase=Phase.FINISHED,
            current_player=Role.LANDLORD_DOWN,
            landlord=Role.LANDLORD,
            play_history=(
                (Role.LANDLORD_DOWN, (5,)),
            ),
            winner="farmer",
        )
        assert state.is_spring() is True


class TestGameStateWithAction:
    """with_action 统一接口测试"""

    def test_with_action_bidding(self):
        state = GameState.initial(seed=42)
        new_state = state.with_action(1)
        assert new_state.bid_info[0] == 1

    def test_with_action_playing(self):
        state = GameState.initial(seed=42).with_bid(3)
        hand = list(state.get_hand(Role.LANDLORD))
        action = Action.from_cards([hand[0]])
        new_state = state.with_action(action)
        assert new_state.step_count == 1

    def test_with_action_wrong_type_bidding(self):
        state = GameState.initial(seed=42)
        with pytest.raises(ValueError):
            state.with_action(Action.pass_action())

    def test_with_action_wrong_type_playing(self):
        state = GameState.initial(seed=42).with_bid(3)
        with pytest.raises(ValueError):
            state.with_action(1)

    def test_with_action_finished(self):
        state = GameState(
            hands=(
                (Role.LANDLORD.value, ()),
                (Role.LANDLORD_DOWN.value, ()),
                (Role.LANDLORD_UP.value, ()),
            ),
            three_cards=(),
            phase=Phase.FINISHED,
            current_player=Role.LANDLORD,
            winner="landlord",
        )
        with pytest.raises(ValueError):
            state.with_action(1)
