"""
Rollout Worker

收集训练轨迹
"""
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from queue import Queue
import threading

from .buffer import Transition, Trajectory
from .config import RolloutConfig


@dataclass
class RolloutResult:
    """Rollout 结果"""
    trajectories: List[Trajectory]
    total_steps: int
    total_episodes: int
    avg_reward: float
    avg_length: float
    win_rate: float


class RolloutWorker:
    """
    单个 Rollout Worker

    收集一个环境的轨迹
    """

    def __init__(
        self,
        env_fn: Callable,
        policy_fn: Callable[[Dict[str, np.ndarray]], int],
        worker_id: int = 0,
        temperature: float = 1.0,
        epsilon: float = 0.0,
    ):
        self.env = env_fn()
        self.policy_fn = policy_fn
        self.worker_id = worker_id
        self.temperature = temperature
        self.epsilon = epsilon

        self._obs = None
        self._info = None

    def reset(self):
        """重置环境"""
        self._obs, self._info = self.env.reset()

    def collect_trajectory(self) -> Trajectory:
        """收集一条完整轨迹"""
        if self._obs is None:
            self.reset()

        trajectory = Trajectory()
        done = False

        while not done:
            # 选择动作
            if np.random.random() < self.epsilon:
                action = self.env.sample_action()
            else:
                action = self.policy_fn(self._obs)

            # 执行动作
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # 记录转移
            transition = Transition(
                obs=self._obs,
                action=action,
                reward=reward,
                next_obs=next_obs if not done else None,
                done=done,
                info=info,
            )
            trajectory.add(transition)

            if done:
                self._obs, self._info = self.env.reset()
                # 记录结果
                trajectory.player = info.get("player", "unknown")
                trajectory.winner = info.get("winner", None)
            else:
                self._obs = next_obs
                self._info = info

        return trajectory

    def collect_steps(self, n_steps: int) -> List[Trajectory]:
        """收集指定步数的数据"""
        if self._obs is None:
            self.reset()

        trajectories = []
        current_trajectory = Trajectory()
        steps = 0

        while steps < n_steps:
            # 选择动作
            if np.random.random() < self.epsilon:
                action = self.env.sample_action()
            else:
                action = self.policy_fn(self._obs)

            # 执行动作
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # 记录转移
            transition = Transition(
                obs=self._obs,
                action=action,
                reward=reward,
                next_obs=next_obs if not done else None,
                done=done,
                info=info,
            )
            current_trajectory.add(transition)
            steps += 1

            if done:
                current_trajectory.player = info.get("player", "unknown")
                current_trajectory.winner = info.get("winner", None)
                trajectories.append(current_trajectory)
                current_trajectory = Trajectory()
                self._obs, self._info = self.env.reset()
            else:
                self._obs = next_obs
                self._info = info

        # 添加未完成的轨迹
        if len(current_trajectory) > 0:
            trajectories.append(current_trajectory)

        return trajectories


class VectorRolloutWorker:
    """
    向量化 Rollout Worker

    并行收集多个环境的轨迹
    """

    def __init__(
        self,
        env_fn: Callable,
        policy_fn: Callable[[Dict[str, np.ndarray]], np.ndarray],
        num_envs: int = 8,
        temperature: float = 1.0,
        epsilon: float = 0.0,
    ):
        self.envs = [env_fn() for _ in range(num_envs)]
        self.policy_fn = policy_fn
        self.num_envs = num_envs
        self.temperature = temperature
        self.epsilon = epsilon

        self._obs_list = [None] * num_envs
        self._trajectories = [Trajectory() for _ in range(num_envs)]

    def reset_all(self):
        """重置所有环境"""
        for i, env in enumerate(self.envs):
            self._obs_list[i], _ = env.reset()
            self._trajectories[i] = Trajectory()

    def collect_steps(self, n_steps: int) -> RolloutResult:
        """收集指定步数"""
        if self._obs_list[0] is None:
            self.reset_all()

        completed_trajectories = []
        total_steps = 0
        total_rewards = []
        total_lengths = []
        wins = 0

        for _ in range(n_steps):
            # 批量获取动作
            actions = []
            for i in range(self.num_envs):
                if np.random.random() < self.epsilon:
                    actions.append(self.envs[i].sample_action())
                else:
                    actions.append(self.policy_fn(self._obs_list[i]))

            # 执行动作
            for i, (env, action) in enumerate(zip(self.envs, actions)):
                obs = self._obs_list[i]
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                transition = Transition(
                    obs=obs,
                    action=action,
                    reward=reward,
                    next_obs=next_obs if not done else None,
                    done=done,
                    info=info,
                )
                self._trajectories[i].add(transition)
                total_steps += 1

                if done:
                    traj = self._trajectories[i]
                    traj.player = info.get("player", "unknown")
                    traj.winner = info.get("winner", None)
                    completed_trajectories.append(traj)

                    total_rewards.append(traj.total_reward)
                    total_lengths.append(len(traj))
                    if info.get("won", False):
                        wins += 1

                    self._trajectories[i] = Trajectory()
                    self._obs_list[i], _ = env.reset()
                else:
                    self._obs_list[i] = next_obs

        # 添加未完成的轨迹
        for traj in self._trajectories:
            if len(traj) > 0:
                completed_trajectories.append(traj)

        n_episodes = len([t for t in completed_trajectories if t.transitions[-1].done])

        return RolloutResult(
            trajectories=completed_trajectories,
            total_steps=total_steps,
            total_episodes=n_episodes,
            avg_reward=np.mean(total_rewards) if total_rewards else 0.0,
            avg_length=np.mean(total_lengths) if total_lengths else 0.0,
            win_rate=wins / n_episodes if n_episodes > 0 else 0.0,
        )


class AsyncRolloutManager:
    """
    异步 Rollout 管理器

    在后台线程中持续收集轨迹
    """

    def __init__(
        self,
        env_fn: Callable,
        num_workers: int = 4,
        steps_per_worker: int = 256,
    ):
        self.env_fn = env_fn
        self.num_workers = num_workers
        self.steps_per_worker = steps_per_worker

        self._workers: List[RolloutWorker] = []
        self._trajectory_queue: Queue = Queue(maxsize=100)
        self._running = False
        self._threads: List[threading.Thread] = []
        self._policy_fn = None

    def set_policy(self, policy_fn: Callable):
        """设置策略函数"""
        self._policy_fn = policy_fn

    def start(self):
        """启动后台收集"""
        if self._running:
            return

        self._running = True
        for i in range(self.num_workers):
            worker = RolloutWorker(
                env_fn=self.env_fn,
                policy_fn=self._policy_fn,
                worker_id=i,
            )
            self._workers.append(worker)

            thread = threading.Thread(
                target=self._worker_loop,
                args=(worker,),
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def _worker_loop(self, worker: RolloutWorker):
        """Worker 主循环"""
        while self._running:
            trajectories = worker.collect_steps(self.steps_per_worker)
            for traj in trajectories:
                self._trajectory_queue.put(traj)

    def get_trajectories(self, n: int, timeout: float = 1.0) -> List[Trajectory]:
        """获取轨迹"""
        trajectories = []
        for _ in range(n):
            try:
                traj = self._trajectory_queue.get(timeout=timeout)
                trajectories.append(traj)
            except:
                break
        return trajectories

    def stop(self):
        """停止收集"""
        self._running = False
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads.clear()
        self._workers.clear()


def collect_rollout(
    env_fn: Callable,
    policy: "torch.nn.Module",
    n_steps: int = 2048,
    n_envs: int = 8,
    device: str = "cpu",
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Dict[str, torch.Tensor]:
    """
    收集 rollout 数据

    Args:
        env_fn: 环境构造函数
        policy: 策略网络
        n_steps: 步数
        n_envs: 环境数
        device: 设备
        gamma: 折扣因子
        gae_lambda: GAE lambda

    Returns:
        训练数据字典
    """
    envs = [env_fn() for _ in range(n_envs)]

    # 初始化存储 - 包含模型所需的全部观测键
    obs_keys = ["hand", "played_cards", "history", "last_action", "position", "bid_info", "cards_left"]
    observations = {key: [] for key in obs_keys}
    actions = []
    rewards = []
    dones = []
    values = []
    log_probs = []

    # 统计信息
    episode_count = 0
    win_count = 0
    episode_lengths = []
    current_episode_length = [0] * n_envs

    # 重置环境
    obs_list = [env.reset()[0] for env in envs]

    policy.eval()
    with torch.no_grad():
        for step in range(n_steps // n_envs):
            for i, (env, obs) in enumerate(zip(envs, obs_list)):
                # 转换观测 - 包含模型所需的全部键
                obs_tensor = {
                    k: torch.from_numpy(v).unsqueeze(0).to(device)
                    for k, v in obs.items()
                    if k in obs_keys
                }

                # 获取动作
                output = policy(obs_tensor)
                dist = torch.distributions.Categorical(logits=output.policy_logits)
                action = dist.sample()

                # 存储
                for key in observations:
                    if key in obs:
                        observations[key].append(obs[key])
                actions.append(action.item())
                log_probs.append(dist.log_prob(action).item())
                values.append(output.value.squeeze().item())

                # 执行
                next_obs, reward, terminated, truncated, info = env.step(action.item())
                done = terminated or truncated

                rewards.append(reward)
                dones.append(done)
                current_episode_length[i] += 1

                if done:
                    episode_count += 1
                    episode_lengths.append(current_episode_length[i])
                    current_episode_length[i] = 0
                    # 检查是否获胜
                    if info.get("winner") == "landlord" or reward > 0:
                        win_count += 1
                    obs_list[i], _ = env.reset()
                else:
                    obs_list[i] = next_obs

    # 转换为张量
    data = {}
    for key, vals in observations.items():
        if vals:
            data[key] = torch.from_numpy(np.stack(vals))

    data["actions"] = torch.tensor(actions, dtype=torch.long)
    data["rewards"] = torch.tensor(rewards, dtype=torch.float32)
    data["dones"] = torch.tensor(dones, dtype=torch.bool)
    data["values"] = torch.tensor(values, dtype=torch.float32)
    data["log_probs"] = torch.tensor(log_probs, dtype=torch.float32)

    # 计算 GAE
    advantages = []
    returns = []
    gae = 0.0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]

        next_non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])

    data["advantages"] = torch.tensor(advantages, dtype=torch.float32)
    data["returns"] = torch.tensor(returns, dtype=torch.float32)

    # 归一化优势
    data["advantages"] = (data["advantages"] - data["advantages"].mean()) / (
        data["advantages"].std() + 1e-8
    )

    # 添加统计信息
    data["episode_count"] = episode_count
    data["win_count"] = win_count
    data["win_rate"] = win_count / episode_count if episode_count > 0 else 0.0
    data["avg_episode_length"] = np.mean(episode_lengths) if episode_lengths else 0.0

    return data
