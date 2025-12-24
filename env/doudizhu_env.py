"""
斗地主 Gymnasium 环境

遵循标准 Gymnasium API
"""
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from core.state import GameState, Phase, Role, PLAY_ORDER, BID_ORDER
from core.actions import Action, ActionType
from core.cards import cards_to_str

from .observation import ObservationBuilder, Observation, ActionEncoder, get_action_encoder
from .reward import RewardCalculator, RewardConfig, RewardType


class DoudizhuEnv(gym.Env):
    """
    斗地主 Gymnasium 环境

    支持:
    - 单智能体模式 (控制一个玩家)
    - 多智能体模式 (所有玩家)

    API:
    - reset() -> observation, info
    - step(action) -> observation, reward, terminated, truncated, info
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "Doudizhu-v2",
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        reward_type: str = "adp",
        history_length: int = 15,
        agent_role: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            render_mode: 渲染模式 ("human", "ansi", None)
            reward_type: 奖励类型 ("sparse", "shaped", "adp")
            history_length: 历史长度
            agent_role: 智能体角色 (None=当前玩家, "landlord"等)
            seed: 随机种子
        """
        super().__init__()

        self.render_mode = render_mode
        self._seed = seed

        # 观测构建器
        self._obs_builder = ObservationBuilder(history_length=history_length)

        # 奖励计算器
        self._reward_calculator = RewardCalculator(
            RewardConfig(reward_type=RewardType(reward_type))
        )

        # 动作编码器
        self._action_encoder = get_action_encoder()

        # 智能体角色
        self._agent_role = Role(agent_role) if agent_role else None

        # 状态
        self._state: Optional[GameState] = None
        self._prev_state: Optional[GameState] = None

        # 定义空间
        self._define_spaces()

    def _define_spaces(self):
        """定义观测和动作空间"""
        # 动作空间: 所有可能动作的索引
        self.action_space = spaces.Discrete(self._action_encoder.num_actions)

        # 观测空间: 字典形式
        self.observation_space = spaces.Dict({
            "hand": spaces.Box(0, 1, shape=(54,), dtype=np.float32),
            "other_hands": spaces.Box(0, 1, shape=(2, 54), dtype=np.float32),
            "played_cards": spaces.Box(0, 1, shape=(3, 54), dtype=np.float32),
            "history": spaces.Box(0, 1, shape=(15, 54), dtype=np.float32),
            "last_action": spaces.Box(0, 1, shape=(54,), dtype=np.float32),
            "position": spaces.Box(0, 1, shape=(6,), dtype=np.float32),
            "bid_info": spaces.Box(-1, 3, shape=(3,), dtype=np.float32),
            "cards_left": spaces.Box(0, 1, shape=(3,), dtype=np.float32),
        })

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        重置环境

        Args:
            seed: 随机种子
            options: 额外选项

        Returns:
            (observation, info) 元组
        """
        super().reset(seed=seed)

        # 使用提供的种子或初始种子
        game_seed = seed if seed is not None else self._seed

        # 初始化新游戏
        self._state = GameState.initial(seed=game_seed)
        self._prev_state = None

        # 构建观测
        obs = self._build_observation()
        info = self._build_info()

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(
        self,
        action: Union[int, Action],
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        执行动作

        Args:
            action: 动作索引或 Action 对象

        Returns:
            (observation, reward, terminated, truncated, info) 元组
        """
        if self._state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        # 保存前一状态
        self._prev_state = self._state

        # 解码动作
        concrete_action = self._decode_action(action)

        # 验证动作合法性
        if not self._is_valid_action(concrete_action):
            # 非法动作：给予惩罚并保持状态
            obs = self._build_observation()
            info = self._build_info()
            info["error"] = "Invalid action"
            return obs, -1.0, False, False, info

        # 执行动作
        self._state = self._state.with_action(concrete_action)

        # 构建观测
        obs = self._build_observation()

        # 计算奖励
        reward = self._compute_reward()

        # 检查终止
        terminated = self._state.phase == Phase.FINISHED
        truncated = False

        # 构建 info
        info = self._build_info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _decode_action(self, action: Union[int, Action]) -> Union[int, Action]:
        """解码动作"""
        if isinstance(action, Action):
            return action
        elif isinstance(action, (int, np.integer)):
            if self._state.phase == Phase.BIDDING:
                # 叫牌阶段：动作是叫分值
                return action
            else:
                # 出牌阶段：解码为 Action
                decoded = self._action_encoder.decode(action)
                if decoded is None:
                    raise ValueError(
                        f"Invalid action index: {action}. "
                        f"Valid range: 0-{self._action_encoder.num_actions - 1}"
                    )
                return decoded
        else:
            raise ValueError(f"Invalid action type: {type(action)}")

    def _is_valid_action(self, action: Union[int, Action]) -> bool:
        """验证动作合法性"""
        legal_actions = self._state.get_legal_actions()

        if self._state.phase == Phase.BIDDING:
            return action in legal_actions
        else:
            return action in legal_actions

    def _build_observation(self) -> Dict[str, np.ndarray]:
        """构建观测"""
        perspective = self._agent_role if self._agent_role else None
        obs = self._obs_builder.build(self._state, perspective)
        return obs.to_dict()

    def _build_info(self) -> Dict[str, Any]:
        """构建 info 字典"""
        legal_actions = self._state.get_legal_actions()

        info = {
            "current_player": self._state.current_player.value,
            "phase": self._state.phase.value,
            "legal_actions": legal_actions,
            "step_count": self._state.step_count,
            "bombs_count": self._state.bombs_count,
        }

        if self._state.phase == Phase.PLAYING:
            # 添加动作掩码
            info["legal_action_mask"] = self._action_encoder.build_legal_mask(
                legal_actions
            )
            info["legal_action_indices"] = self._action_encoder.get_legal_action_indices(
                legal_actions
            )

        if self._state.phase == Phase.FINISHED:
            info["winner"] = self._state.winner
            info["is_spring"] = self._state.is_spring()

        return info

    def _compute_reward(self) -> float:
        """计算奖励"""
        player = self._agent_role if self._agent_role else self._state.current_player
        return self._reward_calculator.compute(
            self._state, self._prev_state, player
        )

    def render(self) -> Optional[str]:
        """渲染环境"""
        if self.render_mode == "ansi" or self.render_mode == "human":
            return self._render_text()
        return None

    def _render_text(self) -> str:
        """文本渲染"""
        lines = []
        lines.append("=" * 50)
        lines.append(f"Phase: {self._state.phase.value}")
        lines.append(f"Current Player: {self._state.current_player.value}")

        if self._state.phase == Phase.BIDDING:
            lines.append(f"Bid Info: {self._state.bid_info}")
        else:
            lines.append(f"Landlord: {self._state.landlord.value if self._state.landlord else 'None'}")

            # 显示手牌
            for role in PLAY_ORDER:
                hand = self._state.get_hand(role)
                hand_str = cards_to_str(list(hand))
                lines.append(f"{role.value}: {hand_str} ({len(hand)})")

            # 显示上一动作
            if self._state.last_action:
                last_str = cards_to_str(list(self._state.last_action.cards))
                lines.append(f"Last Action: {last_str} by {self._state.last_player.value}")

        if self._state.phase == Phase.FINISHED:
            lines.append(f"Winner: {self._state.winner}")
            lines.append(f"Spring: {self._state.is_spring()}")

        lines.append("=" * 50)

        output = "\n".join(lines)
        if self.render_mode == "human":
            print(output)
        return output

    def close(self):
        """关闭环境"""
        pass

    @property
    def state(self) -> Optional[GameState]:
        """获取当前状态 (用于调试)"""
        return self._state

    def get_legal_actions(self) -> List:
        """获取当前合法动作"""
        if self._state is None:
            return []
        return self._state.get_legal_actions()

    def sample_action(self) -> Union[int, Action]:
        """随机采样一个合法动作"""
        legal_actions = self.get_legal_actions()
        if not legal_actions:
            return 0
        idx = np.random.randint(len(legal_actions))
        return legal_actions[idx]


class MultiAgentDoudizhuEnv(DoudizhuEnv):
    """
    多智能体斗地主环境

    返回所有玩家的观测和奖励
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._obs_builder = ObservationBuilder(
            history_length=kwargs.get("history_length", 15),
            include_other_hands=False,  # 不完全信息
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Dict], Dict[str, Any]]:
        """重置并返回所有玩家观测"""
        super().reset(seed=seed, options=options)

        observations = {}
        if self._state.phase == Phase.BIDDING:
            for role in BID_ORDER:
                obs = self._obs_builder.build(self._state, role)
                observations[role.value] = obs.to_dict()
        else:
            for role in PLAY_ORDER:
                obs = self._obs_builder.build(self._state, role)
                observations[role.value] = obs.to_dict()

        info = self._build_info()
        return observations, info

    def step(
        self,
        action: Union[int, Action],
    ) -> Tuple[Dict[str, Dict], Dict[str, float], bool, bool, Dict[str, Any]]:
        """执行动作并返回所有玩家的结果"""
        _, _, terminated, truncated, info = super().step(action)

        # 构建所有玩家的观测
        observations = {}
        rewards = {}

        if self._state.phase == Phase.BIDDING:
            roles = BID_ORDER
        else:
            roles = PLAY_ORDER

        for role in roles:
            obs = self._obs_builder.build(self._state, role)
            observations[role.value] = obs.to_dict()
            rewards[role.value] = self._reward_calculator.compute(
                self._state, self._prev_state, role
            )

        return observations, rewards, terminated, truncated, info


def make_env(
    env_id: str = "Doudizhu-v2",
    **kwargs
) -> DoudizhuEnv:
    """
    工厂函数：创建环境

    Args:
        env_id: 环境 ID
        **kwargs: 环境参数

    Returns:
        DoudizhuEnv 实例
    """
    if "multi_agent" in kwargs and kwargs.pop("multi_agent"):
        return MultiAgentDoudizhuEnv(**kwargs)
    return DoudizhuEnv(**kwargs)
