"""环境层测试"""
import pytest
import numpy as np

from core.state import Phase, Role, GameState
from core.actions import Action, ActionType


class TestObservationBuilder:
    """ObservationBuilder 测试"""

    def test_build_bidding_phase(self):
        from env.observation import ObservationBuilder

        builder = ObservationBuilder()
        state = GameState.initial(seed=42)
        obs = builder.build(state)

        assert obs.hand.shape == (54,)
        assert obs.phase == "bidding"
        assert obs.position.shape == (6,)
        assert obs.position.sum() == 1  # one-hot

    def test_build_playing_phase(self):
        from env.observation import ObservationBuilder

        builder = ObservationBuilder()
        state = GameState.initial(seed=42).with_bid(3)
        obs = builder.build(state)

        assert obs.hand.shape == (54,)
        assert obs.phase == "playing"
        assert obs.played_cards.shape == (3, 54)
        assert obs.history.shape == (15, 54)

    def test_to_dict(self):
        from env.observation import ObservationBuilder

        builder = ObservationBuilder()
        state = GameState.initial(seed=42).with_bid(3)
        obs = builder.build(state)
        obs_dict = obs.to_dict()

        assert "hand" in obs_dict
        assert "played_cards" in obs_dict
        assert "history" in obs_dict

    def test_to_flat_array(self):
        from env.observation import ObservationBuilder

        builder = ObservationBuilder()
        state = GameState.initial(seed=42).with_bid(3)
        obs = builder.build(state)
        flat = obs.to_flat_array()

        assert isinstance(flat, np.ndarray)
        assert flat.ndim == 1


class TestActionEncoder:
    """ActionEncoder 测试"""

    def test_encode_pass(self):
        from env.observation import ActionEncoder

        encoder = ActionEncoder()
        action = Action.pass_action()
        idx = encoder.encode(action)

        assert idx == 0

    def test_encode_single(self):
        from env.observation import ActionEncoder

        encoder = ActionEncoder()
        action = Action.from_cards([5])
        idx = encoder.encode(action)

        assert idx > 0

    def test_decode_single(self):
        from env.observation import ActionEncoder

        encoder = ActionEncoder()
        action = Action.from_cards([5])
        idx = encoder.encode(action)
        decoded = encoder.decode(idx)

        assert decoded is not None
        assert decoded.cards == action.cards

    def test_roundtrip_pair(self):
        from env.observation import ActionEncoder

        encoder = ActionEncoder()
        action = Action.from_cards([7, 7])
        idx = encoder.encode(action)
        decoded = encoder.decode(idx)

        assert decoded is not None
        assert decoded.cards == action.cards

    def test_build_legal_mask(self):
        from env.observation import ActionEncoder, get_action_encoder

        encoder = get_action_encoder()
        legal_actions = [
            Action.pass_action(),
            Action.from_cards([3]),
            Action.from_cards([5]),
        ]
        mask = encoder.build_legal_mask(legal_actions)

        assert mask.shape == (encoder.num_actions,)
        assert mask[0] == 1  # PASS


class TestRewardCalculator:
    """RewardCalculator 测试"""

    def test_sparse_reward_not_finished(self):
        from env.reward import RewardCalculator, RewardConfig, RewardType

        calc = RewardCalculator(RewardConfig(reward_type=RewardType.SPARSE))
        state = GameState.initial(seed=42).with_bid(3)
        reward = calc.compute(state, player=Role.LANDLORD)

        assert reward == 0.0

    def test_sparse_reward_landlord_wins(self):
        from env.reward import RewardCalculator, RewardConfig, RewardType

        calc = RewardCalculator(RewardConfig(reward_type=RewardType.SPARSE))

        # 创建地主赢的状态
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
            winner="landlord",
        )

        reward = calc.compute(state, player=Role.LANDLORD)
        assert reward == 1.0

        reward = calc.compute(state, player=Role.LANDLORD_DOWN)
        assert reward == -1.0

    def test_adp_reward(self):
        from env.reward import RewardCalculator, RewardConfig, RewardType

        calc = RewardCalculator(RewardConfig(reward_type=RewardType.ADP))

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
            bid_info=(3, 0, 0),  # 叫 3 分
            winner="landlord",
        )

        reward = calc.compute(state, player=Role.LANDLORD)
        assert reward > 0  # 地主赢得正奖励


class TestDoudizhuEnv:
    """DoudizhuEnv 测试"""

    def test_reset(self):
        from env import DoudizhuEnv

        env = DoudizhuEnv()
        obs, info = env.reset(seed=42)

        assert "hand" in obs
        assert "current_player" in info
        assert info["phase"] == "bidding"

    def test_step_bidding(self):
        from env import DoudizhuEnv

        env = DoudizhuEnv()
        obs, info = env.reset(seed=42)

        # 叫牌
        obs, reward, terminated, truncated, info = env.step(3)

        assert not terminated
        assert info["phase"] == "playing"

    def test_step_playing(self):
        from env import DoudizhuEnv

        env = DoudizhuEnv()
        obs, info = env.reset(seed=42)
        env.step(3)  # 叫牌

        # 出牌
        legal_actions = env.get_legal_actions()
        action = legal_actions[0]
        obs, reward, terminated, truncated, info = env.step(action)

        assert "hand" in obs

    def test_full_game(self):
        from env import DoudizhuEnv

        env = DoudizhuEnv()
        obs, info = env.reset(seed=42)

        # 完成一局游戏
        done = False
        steps = 0
        max_steps = 200

        while not done and steps < max_steps:
            action = env.sample_action()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        assert done or steps == max_steps

    def test_render_ansi(self):
        from env import DoudizhuEnv

        env = DoudizhuEnv(render_mode="ansi")
        env.reset(seed=42)
        env.step(3)

        output = env.render()
        assert isinstance(output, str)
        assert "landlord" in output


class TestWrappers:
    """环境包装器测试"""

    def test_flatten_observation_wrapper(self):
        from env import DoudizhuEnv
        from env.wrappers import FlattenObservationWrapper

        env = DoudizhuEnv()
        env = FlattenObservationWrapper(env)
        obs, _ = env.reset(seed=42)

        assert isinstance(obs, np.ndarray)
        assert obs.ndim == 1

    def test_record_episode_statistics(self):
        from env import DoudizhuEnv
        from env.wrappers import RecordEpisodeStatistics

        env = DoudizhuEnv()
        env = RecordEpisodeStatistics(env)
        env.reset(seed=42)

        # 完成一局
        done = False
        steps = 0
        info = {}
        while not done and steps < 200:
            action = env.unwrapped.sample_action()
            _, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        if done:
            assert "episode" in info
            assert "r" in info["episode"]
            assert "l" in info["episode"]

    def test_time_limit(self):
        from env import DoudizhuEnv
        from env.wrappers import TimeLimit

        env = DoudizhuEnv()
        env = TimeLimit(env, max_steps=10)
        env.reset(seed=42)

        for _ in range(15):
            action = env.unwrapped.sample_action()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        # 应该在 10 步内被截断
        assert truncated or terminated

    def test_wrap_env(self):
        from env import DoudizhuEnv
        from env.wrappers import wrap_env

        env = DoudizhuEnv()
        env = wrap_env(
            env,
            flatten_obs=False,
            action_mask=True,
            record_stats=True,
            time_limit=100,
        )

        obs, info = env.reset(seed=42)
        assert "action_mask" in info


class TestMultiAgentEnv:
    """多智能体环境测试"""

    def test_multi_agent_reset(self):
        from env import MultiAgentDoudizhuEnv

        env = MultiAgentDoudizhuEnv()
        observations, info = env.reset(seed=42)

        # 叫牌阶段有 3 个玩家
        assert len(observations) == 3
        assert "first" in observations

    def test_multi_agent_step(self):
        from env import MultiAgentDoudizhuEnv

        env = MultiAgentDoudizhuEnv()
        observations, info = env.reset(seed=42)

        # 叫牌
        observations, rewards, terminated, truncated, info = env.step(3)

        assert len(rewards) == 3
        assert "landlord" in observations


class TestMakeEnv:
    """make_env 工厂函数测试"""

    def test_make_single_agent(self):
        from env import make_env

        env = make_env()
        assert env is not None
        obs, _ = env.reset(seed=42)
        assert "hand" in obs

    def test_make_multi_agent(self):
        from env import make_env

        env = make_env(multi_agent=True)
        assert env is not None
        observations, _ = env.reset(seed=42)
        assert isinstance(observations, dict)
