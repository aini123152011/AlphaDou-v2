"""
环境包装器

提供常用的环境增强功能
"""
from typing import Dict, Any, Tuple, Optional, List, Callable
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import Wrapper
except ImportError:
    import gym
    from gym import Wrapper

from core.state import Phase


class FlattenObservationWrapper(Wrapper):
    """
    将字典观测展平为单一向量

    用于不支持字典观测的算法
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        # 计算展平后的维度
        sample_obs, _ = env.reset()
        flat_dim = sum(
            np.prod(v.shape) for v in sample_obs.values()
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(flat_dim),),
            dtype=np.float32,
        )

    def observation(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """展平观测"""
        return np.concatenate([
            v.flatten() for v in obs.values()
        ])

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info


class LegalActionMaskWrapper(Wrapper):
    """
    在 info 中添加合法动作掩码

    用于支持 action masking 的算法
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        obs, info = self.env.reset(**kwargs)
        info["action_mask"] = self._get_action_mask()
        return obs, info

    def step(self, action) -> Tuple[Dict, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not terminated:
            info["action_mask"] = self._get_action_mask()
        return obs, reward, terminated, truncated, info

    def _get_action_mask(self) -> np.ndarray:
        """获取动作掩码"""
        if hasattr(self.env, "_action_encoder"):
            legal_actions = self.env.get_legal_actions()
            return self.env._action_encoder.build_legal_mask(legal_actions)
        return np.ones(self.action_space.n, dtype=np.float32)


class SelfPlayWrapper(Wrapper):
    """
    自博弈包装器

    自动为对手玩家选择动作
    """

    def __init__(
        self,
        env: gym.Env,
        opponent_policy: Optional[Callable] = None,
        controlled_role: str = "landlord",
    ):
        """
        Args:
            env: 基础环境
            opponent_policy: 对手策略函数 (obs -> action)
            controlled_role: 控制的角色
        """
        super().__init__(env)
        self.opponent_policy = opponent_policy or self._random_policy
        self.controlled_role = controlled_role

    def _random_policy(self, obs: Dict, info: Dict) -> int:
        """随机策略"""
        legal_actions = info.get("legal_actions", [])
        if not legal_actions:
            return 0
        idx = np.random.randint(len(legal_actions))
        return legal_actions[idx]

    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        obs, info = self.env.reset(**kwargs)

        # 如果不是我方回合，让对手先行动
        while info.get("current_player") != self.controlled_role:
            if self.env.state.phase == Phase.FINISHED:
                break
            action = self.opponent_policy(obs, info)
            obs, _, terminated, _, info = self.env.step(action)
            if terminated:
                break

        return obs, info

    def step(self, action) -> Tuple[Dict, float, bool, bool, Dict]:
        # 执行我方动作
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated:
            return obs, reward, terminated, truncated, info

        # 对手回合
        while info.get("current_player") != self.controlled_role:
            if self.env.state.phase == Phase.FINISHED:
                break
            opponent_action = self.opponent_policy(obs, info)
            obs, opp_reward, terminated, truncated, info = self.env.step(opponent_action)
            if terminated:
                # 更新最终奖励
                reward = self._compute_final_reward()
                break

        return obs, reward, terminated, truncated, info

    def _compute_final_reward(self) -> float:
        """计算最终奖励"""
        if hasattr(self.env, "_reward_calculator"):
            from core.state import Role
            role = Role(self.controlled_role)
            return self.env._reward_calculator.compute(
                self.env.state, None, role
            )
        return 0.0


class FrameStackWrapper(Wrapper):
    """
    帧堆叠包装器

    将多个连续帧堆叠在一起
    """

    def __init__(self, env: gym.Env, num_stack: int = 4):
        super().__init__(env)
        self.num_stack = num_stack
        self._frames: List[Dict] = []

        # 更新观测空间
        sample_obs, _ = env.reset()
        new_spaces = {}
        for key, space in env.observation_space.spaces.items():
            new_shape = (num_stack,) + space.shape
            new_spaces[key] = gym.spaces.Box(
                low=space.low.min(),
                high=space.high.max(),
                shape=new_shape,
                dtype=space.dtype,
            )
        self.observation_space = gym.spaces.Dict(new_spaces)

    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        obs, info = self.env.reset(**kwargs)
        self._frames = [obs] * self.num_stack
        return self._get_stacked_obs(), info

    def step(self, action) -> Tuple[Dict, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.pop(0)
        self._frames.append(obs)
        return self._get_stacked_obs(), reward, terminated, truncated, info

    def _get_stacked_obs(self) -> Dict[str, np.ndarray]:
        """获取堆叠后的观测"""
        stacked = {}
        for key in self._frames[0].keys():
            stacked[key] = np.stack([f[key] for f in self._frames], axis=0)
        return stacked


class RewardScaleWrapper(Wrapper):
    """
    奖励缩放包装器
    """

    def __init__(self, env: gym.Env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale

    def step(self, action) -> Tuple[Dict, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward * self.scale, terminated, truncated, info


class TimeLimit(Wrapper):
    """
    时间限制包装器

    限制每局游戏的最大步数
    """

    def __init__(self, env: gym.Env, max_steps: int = 200):
        super().__init__(env)
        self.max_steps = max_steps
        self._step_count = 0

    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        self._step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action) -> Tuple[Dict, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1

        if self._step_count >= self.max_steps:
            truncated = True

        return obs, reward, terminated, truncated, info


class RecordEpisodeStatistics(Wrapper):
    """
    记录回合统计信息
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._episode_reward = 0.0
        self._episode_length = 0
        self._episode_bombs = 0

    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        obs, info = self.env.reset(**kwargs)
        self._episode_reward = 0.0
        self._episode_length = 0
        self._episode_bombs = 0
        return obs, info

    def step(self, action) -> Tuple[Dict, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._episode_reward += reward
        self._episode_length += 1
        self._episode_bombs = info.get("bombs_count", 0)

        if terminated or truncated:
            info["episode"] = {
                "r": self._episode_reward,
                "l": self._episode_length,
                "bombs": self._episode_bombs,
                "winner": info.get("winner"),
                "is_spring": info.get("is_spring", False),
            }

        return obs, reward, terminated, truncated, info


def wrap_env(
    env: gym.Env,
    flatten_obs: bool = False,
    action_mask: bool = True,
    record_stats: bool = True,
    time_limit: Optional[int] = None,
    reward_scale: float = 1.0,
) -> gym.Env:
    """
    应用常用包装器组合

    Args:
        env: 基础环境
        flatten_obs: 是否展平观测
        action_mask: 是否添加动作掩码
        record_stats: 是否记录统计
        time_limit: 时间限制
        reward_scale: 奖励缩放

    Returns:
        包装后的环境
    """
    if record_stats:
        env = RecordEpisodeStatistics(env)

    if time_limit is not None:
        env = TimeLimit(env, max_steps=time_limit)

    if reward_scale != 1.0:
        env = RewardScaleWrapper(env, scale=reward_scale)

    if action_mask:
        env = LegalActionMaskWrapper(env)

    if flatten_obs:
        env = FlattenObservationWrapper(env)

    return env
