"""
训练器

主训练循环
"""
from typing import Dict, Optional, Callable, List, Any
from dataclasses import dataclass, field
from pathlib import Path
import time
import logging
import torch
import torch.nn as nn

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    SummaryWriter = None
    HAS_TENSORBOARD = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    wandb = None
    HAS_WANDB = False

from .config import TrainConfig, RolloutConfig, SelfPlayConfig
from .buffer import ReplayBuffer, RolloutBuffer, Trajectory
from .rollout import RolloutWorker, VectorRolloutWorker, RolloutResult, collect_rollout
from .learner import Learner

logger = logging.getLogger(__name__)


@dataclass
class TrainStats:
    """训练统计"""
    step: int = 0
    episode: int = 0
    total_frames: int = 0
    loss: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    avg_reward: float = 0.0
    avg_length: float = 0.0
    win_rate: float = 0.0
    lr: float = 0.0
    fps: float = 0.0


class Callback:
    """回调基类"""

    def on_train_start(self, trainer: "Trainer"):
        pass

    def on_train_end(self, trainer: "Trainer"):
        pass

    def on_step_start(self, trainer: "Trainer", step: int):
        pass

    def on_step_end(self, trainer: "Trainer", step: int, stats: TrainStats):
        pass

    def on_rollout_end(self, trainer: "Trainer", result: RolloutResult):
        pass


class TensorBoardCallback(Callback):
    """TensorBoard 日志回调"""

    def __init__(self, log_dir: str):
        if not HAS_TENSORBOARD:
            raise ImportError("tensorboard is required for TensorBoardCallback")
        self.writer = SummaryWriter(log_dir)

    def on_step_end(self, trainer: "Trainer", step: int, stats: TrainStats):
        self.writer.add_scalar("train/loss", stats.loss, step)
        self.writer.add_scalar("train/policy_loss", stats.policy_loss, step)
        self.writer.add_scalar("train/value_loss", stats.value_loss, step)
        self.writer.add_scalar("train/entropy", stats.entropy, step)
        self.writer.add_scalar("train/lr", stats.lr, step)
        self.writer.add_scalar("rollout/avg_reward", stats.avg_reward, step)
        self.writer.add_scalar("rollout/avg_length", stats.avg_length, step)
        self.writer.add_scalar("rollout/win_rate", stats.win_rate, step)
        self.writer.add_scalar("perf/fps", stats.fps, step)

    def on_train_end(self, trainer: "Trainer"):
        self.writer.close()


class WandbCallback(Callback):
    """Weights & Biases 日志回调"""

    def __init__(
        self,
        project: str = "alphadou-v2",
        name: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        if not HAS_WANDB:
            raise ImportError("wandb is required for WandbCallback. Run: pip install wandb")
        wandb.init(project=project, name=name, config=config)

    def on_step_end(self, trainer: "Trainer", step: int, stats: TrainStats):
        wandb.log({
            "train/loss": stats.loss,
            "train/policy_loss": stats.policy_loss,
            "train/value_loss": stats.value_loss,
            "train/entropy": stats.entropy,
            "train/lr": stats.lr,
            "rollout/avg_reward": stats.avg_reward,
            "rollout/avg_length": stats.avg_length,
            "rollout/win_rate": stats.win_rate,
            "perf/fps": stats.fps,
            "step": step,
        })

    def on_train_end(self, trainer: "Trainer"):
        wandb.finish()


class CheckpointCallback(Callback):
    """检查点回调"""

    def __init__(self, save_dir: str, save_freq: int = 1000):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_freq = save_freq

    def on_step_end(self, trainer: "Trainer", step: int, stats: TrainStats):
        if step % self.save_freq == 0:
            path = self.save_dir / f"checkpoint_{step}.pt"
            trainer.save(str(path))
            logger.info(f"Saved checkpoint to {path}")


class EarlyStoppingCallback(Callback):
    """早停回调"""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_reward = float('-inf')
        self.wait = 0

    def on_step_end(self, trainer: "Trainer", step: int, stats: TrainStats):
        if stats.avg_reward > self.best_reward + self.min_delta:
            self.best_reward = stats.avg_reward
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                logger.info("Early stopping triggered")
                trainer.should_stop = True


class Trainer:
    """
    PPO 训练器

    完整的训练流程
    """

    def __init__(
        self,
        model: nn.Module,
        env_fn: Callable,
        config: TrainConfig,
        rollout_config: Optional[RolloutConfig] = None,
        callbacks: Optional[List[Callback]] = None,
        device: str = "auto",
    ):
        self.config = config
        self.rollout_config = rollout_config or RolloutConfig()
        self.callbacks = callbacks or []

        # 设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 模型
        self.model = model.to(self.device)

        # 学习器
        self.learner = Learner(self.model, config, loss_type="ppo")

        # 环境
        self.env_fn = env_fn

        # 状态
        self.global_step = 0
        self.total_frames = 0
        self.should_stop = False

    def _policy_fn(self, obs: Dict[str, Any]) -> int:
        """策略函数"""
        self.model.eval()
        with torch.no_grad():
            obs_tensor = {
                k: torch.from_numpy(v).unsqueeze(0).to(self.device)
                for k, v in obs.items()
                if isinstance(v, (torch.Tensor, type(None))) is False
            }
            output = self.model(obs_tensor)
            dist = torch.distributions.Categorical(logits=output.policy_logits)
            action = dist.sample()
            return action.item()

    def train(
        self,
        total_steps: int,
        log_interval: int = 10,
    ) -> TrainStats:
        """
        训练

        Args:
            total_steps: 总训练步数
            log_interval: 日志间隔

        Returns:
            最终统计
        """
        # 回调
        for callback in self.callbacks:
            callback.on_train_start(self)

        stats = TrainStats()
        start_time = time.time()

        while self.global_step < total_steps and not self.should_stop:
            step_start = time.time()

            # 回调
            for callback in self.callbacks:
                callback.on_step_start(self, self.global_step)

            # 收集数据
            data = collect_rollout(
                env_fn=self.env_fn,
                policy=self.model,
                n_steps=self.config.n_steps * self.config.n_envs,
                n_envs=self.config.n_envs,
                device=self.device,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
            )

            # 移动到设备
            data = {k: v.to(self.device) for k, v in data.items()}
            n_samples = len(data["actions"])
            self.total_frames += n_samples

            # 训练多个 epoch
            epoch_losses = []
            for epoch in range(self.config.num_epochs):
                losses = self.learner.train_epoch(data, self.config.batch_size)
                epoch_losses.append(losses)

            # 平均损失
            avg_losses = {
                k: sum(e[k] for e in epoch_losses) / len(epoch_losses)
                for k in epoch_losses[0]
            }

            # 更新统计
            step_time = time.time() - step_start
            stats = TrainStats(
                step=self.global_step,
                total_frames=self.total_frames,
                loss=avg_losses["loss"],
                policy_loss=avg_losses["policy_loss"],
                value_loss=avg_losses["value_loss"],
                entropy=-avg_losses["entropy_loss"],
                avg_reward=data["rewards"].mean().item(),
                lr=avg_losses["lr"],
                fps=n_samples / step_time,
            )

            # 日志
            if self.global_step % log_interval == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"Step {self.global_step} | "
                    f"Frames {self.total_frames} | "
                    f"Loss {stats.loss:.4f} | "
                    f"Reward {stats.avg_reward:.2f} | "
                    f"FPS {stats.fps:.0f} | "
                    f"Time {elapsed:.0f}s"
                )

            # 回调
            for callback in self.callbacks:
                callback.on_step_end(self, self.global_step, stats)

            self.global_step += 1

        # 结束回调
        for callback in self.callbacks:
            callback.on_train_end(self)

        return stats

    def save(self, path: str):
        """保存模型"""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.learner.optimizer.state_dict(),
            "global_step": self.global_step,
            "total_frames": self.total_frames,
            "config": self.config,
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.learner.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.total_frames = checkpoint.get("total_frames", 0)


class DMCTrainer:
    """
    Deep Monte Carlo 训练器

    DouZero 风格的训练
    """

    def __init__(
        self,
        model: nn.Module,
        env_fn: Callable,
        config: TrainConfig,
        buffer_size: int = 100000,
        device: str = "auto",
    ):
        self.config = config

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.learner = Learner(self.model, config, loss_type="dmc")

        self.env_fn = env_fn
        self.buffer = ReplayBuffer(capacity=buffer_size, batch_size=config.batch_size)

        self.global_step = 0

    def collect_episode(self) -> Trajectory:
        """收集一局游戏"""
        env = self.env_fn()
        obs, info = env.reset()

        trajectory = Trajectory()

        while True:
            # 选择动作
            with torch.no_grad():
                obs_tensor = {
                    k: torch.from_numpy(v).unsqueeze(0).to(self.device)
                    for k, v in obs.items()
                }
                output = self.model(obs_tensor)
                dist = torch.distributions.Categorical(logits=output.policy_logits)
                action = dist.sample().item()

            # 执行
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            from .buffer import Transition
            transition = Transition(
                obs=obs,
                action=action,
                reward=0.0,  # 暂时设为 0
                next_obs=next_obs if not done else None,
                done=done,
            )
            trajectory.add(transition)

            if done:
                # 根据胜负设置奖励
                final_reward = 1.0 if info.get("won", False) else -1.0
                for t in trajectory.transitions:
                    t.reward = final_reward
                trajectory.winner = info.get("winner")
                break

            obs = next_obs

        return trajectory

    def train_step(self) -> Dict[str, float]:
        """训练一步"""
        if not self.buffer.is_ready():
            return {}

        batch = self.buffer.sample()
        batch = {k: v.to(self.device) for k, v in batch.items()}

        losses = self.learner.step(batch)
        return losses

    def train(self, total_episodes: int, train_freq: int = 4) -> TrainStats:
        """训练"""
        stats = TrainStats()

        for episode in range(total_episodes):
            # 收集
            trajectory = self.collect_episode()
            self.buffer.add_trajectory(trajectory)

            # 训练
            if episode % train_freq == 0 and self.buffer.is_ready():
                losses = self.train_step()
                if losses:
                    stats.loss = losses.get("loss", 0)
                    stats.policy_loss = losses.get("policy_loss", 0)

            stats.episode = episode
            self.global_step += 1

            if episode % 100 == 0:
                logger.info(f"Episode {episode} | Buffer {len(self.buffer)}")

        return stats


class SelfPlayTrainer:
    """
    自博弈训练器

    多智能体自博弈训练
    """

    def __init__(
        self,
        model: nn.Module,
        env_fn: Callable,
        config: TrainConfig,
        self_play_config: Optional[SelfPlayConfig] = None,
        device: str = "auto",
    ):
        self.config = config
        self.sp_config = self_play_config or SelfPlayConfig()

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 当前模型
        self.model = model.to(self.device)
        self.learner = Learner(self.model, config, loss_type="ppo")

        # 对手池
        self.opponent_pool: List[Dict] = []

        self.env_fn = env_fn
        self.global_step = 0

    def add_to_pool(self):
        """将当前模型添加到对手池"""
        state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        self.opponent_pool.append(state_dict)

        # 限制池大小
        if len(self.opponent_pool) > self.sp_config.pool_size:
            self.opponent_pool.pop(0)

    def sample_opponent(self) -> Optional[Dict]:
        """采样对手"""
        if not self.opponent_pool:
            return None

        if torch.rand(1).item() < self.sp_config.sample_latest_prob:
            return self.opponent_pool[-1]
        else:
            idx = torch.randint(len(self.opponent_pool), (1,)).item()
            return self.opponent_pool[idx]

    def train(self, total_steps: int) -> TrainStats:
        """训练"""
        stats = TrainStats()

        for step in range(total_steps):
            # 更新对手池
            if step % self.sp_config.update_freq == 0:
                self.add_to_pool()

            # 收集数据 (使用自博弈)
            data = collect_rollout(
                env_fn=self.env_fn,
                policy=self.model,
                n_steps=self.config.n_steps * self.config.n_envs,
                n_envs=self.config.n_envs,
                device=self.device,
            )
            data = {k: v.to(self.device) for k, v in data.items()}

            # 训练
            for _ in range(self.config.num_epochs):
                losses = self.learner.train_epoch(data)

            stats.step = step
            stats.loss = losses["loss"]
            self.global_step += 1

            # 保存
            if step % self.sp_config.save_freq == 0:
                logger.info(f"Step {step} | Loss {stats.loss:.4f}")

        return stats
