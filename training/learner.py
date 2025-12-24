"""
学习器

计算损失和更新模型
"""
from typing import Dict, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from .config import TrainConfig


class LossInfo(NamedTuple):
    """损失信息"""
    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy_loss: torch.Tensor
    bid_loss: Optional[torch.Tensor] = None
    clip_fraction: Optional[float] = None
    approx_kl: Optional[float] = None


class PPOLoss(nn.Module):
    """
    PPO 损失

    Proximal Policy Optimization 损失函数
    """

    def __init__(
        self,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        normalize_advantage: bool = True,
    ):
        super().__init__()
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.normalize_advantage = normalize_advantage

    def forward(
        self,
        policy_logits: torch.Tensor,
        values: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> LossInfo:
        """
        计算 PPO 损失

        Args:
            policy_logits: 策略 logits (batch, action_dim)
            values: 价值预测 (batch, 1)
            actions: 动作 (batch,)
            old_log_probs: 旧策略的 log prob (batch,)
            old_values: 旧价值 (batch,)
            advantages: 优势 (batch,)
            returns: 回报 (batch,)
            action_mask: 动作掩码 (batch, action_dim)

        Returns:
            LossInfo
        """
        # 应用动作掩码
        if action_mask is not None:
            policy_logits = policy_logits.masked_fill(action_mask == 0, float('-inf'))

        # 计算新策略的 log prob
        dist = torch.distributions.Categorical(logits=policy_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        # 归一化优势
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 策略损失 (PPO-clip)
        ratio = torch.exp(log_probs - old_log_probs)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(
            ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
        )
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        # 价值损失
        values = values.squeeze(-1)
        if self.clip_range_vf is not None:
            # 裁剪价值损失
            values_clipped = old_values + torch.clamp(
                values - old_values, -self.clip_range_vf, self.clip_range_vf
            )
            value_loss_1 = F.mse_loss(values, returns, reduction='none')
            value_loss_2 = F.mse_loss(values_clipped, returns, reduction='none')
            value_loss = torch.max(value_loss_1, value_loss_2).mean()
        else:
            value_loss = F.mse_loss(values, returns)

        # 熵损失
        entropy_loss = -entropy.mean()

        # 总损失
        total_loss = (
            policy_loss
            + self.value_loss_coef * value_loss
            + self.entropy_coef * entropy_loss
        )

        # 计算诊断信息
        with torch.no_grad():
            clip_fraction = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
            approx_kl = ((ratio - 1) - (ratio.log())).mean().item()

        return LossInfo(
            total_loss=total_loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy_loss=entropy_loss,
            clip_fraction=clip_fraction,
            approx_kl=approx_kl,
        )


class A2CLoss(nn.Module):
    """
    A2C 损失

    Advantage Actor-Critic 损失函数
    """

    def __init__(
        self,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ):
        super().__init__()
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

    def forward(
        self,
        policy_logits: torch.Tensor,
        values: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> LossInfo:
        """计算 A2C 损失"""
        if action_mask is not None:
            policy_logits = policy_logits.masked_fill(action_mask == 0, float('-inf'))

        dist = torch.distributions.Categorical(logits=policy_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 策略损失
        policy_loss = -(log_probs * advantages.detach()).mean()

        # 价值损失
        values = values.squeeze(-1)
        value_loss = F.mse_loss(values, returns)

        # 熵损失
        entropy_loss = -entropy.mean()

        total_loss = (
            policy_loss
            + self.value_loss_coef * value_loss
            + self.entropy_coef * entropy_loss
        )

        return LossInfo(
            total_loss=total_loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy_loss=entropy_loss,
        )


class DMCLoss(nn.Module):
    """
    Deep Monte Carlo 损失

    DouZero 风格的训练损失
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        policy_logits: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> LossInfo:
        """
        计算 DMC 损失

        Args:
            policy_logits: 策略 logits (batch, action_dim)
            actions: 实际动作 (batch,)
            rewards: 最终奖励 (batch,)
            action_mask: 动作掩码

        Returns:
            LossInfo
        """
        if action_mask is not None:
            policy_logits = policy_logits.masked_fill(action_mask == 0, float('-inf'))

        # 计算 log prob
        log_probs = F.log_softmax(policy_logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        # REINFORCE 损失
        policy_loss = -(action_log_probs * rewards).mean()

        return LossInfo(
            total_loss=policy_loss,
            policy_loss=policy_loss,
            value_loss=torch.tensor(0.0),
            entropy_loss=torch.tensor(0.0),
        )


class Learner:
    """
    学习器

    封装模型更新逻辑
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        loss_type: str = "ppo",
    ):
        self.model = model
        self.config = config

        # 设置优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # 设置损失函数
        if loss_type == "ppo":
            self.loss_fn = PPOLoss(
                clip_range=config.clip_range,
                clip_range_vf=config.clip_range_vf,
                value_loss_coef=config.value_loss_coef,
                entropy_coef=config.entropy_coef,
            )
        elif loss_type == "a2c":
            self.loss_fn = A2CLoss(
                value_loss_coef=config.value_loss_coef,
                entropy_coef=config.entropy_coef,
            )
        elif loss_type == "dmc":
            self.loss_fn = DMCLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        self.loss_type = loss_type

        # 学习率调度器
        self.scheduler = None
        self._setup_scheduler()

        # 梯度累积
        self.accumulate_steps = config.accumulate_grad_batches
        self._step_count = 0

    def _setup_scheduler(self):
        """设置学习率调度器"""
        if self.config.lr_scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=10000,
                eta_min=1e-6,
            )

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        action_mask: Optional[torch.Tensor] = None,
    ) -> LossInfo:
        """
        计算损失

        Args:
            batch: 训练批次
            action_mask: 动作掩码

        Returns:
            LossInfo
        """
        # 前向传播
        obs = {k: v for k, v in batch.items()
               if k in ["hand", "played_cards", "history", "last_action",
                       "position", "bid_info", "cards_left"]}

        output = self.model(obs, action_mask=action_mask)

        # 计算损失
        if self.loss_type == "ppo":
            loss_info = self.loss_fn(
                policy_logits=output.policy_logits,
                values=output.value,
                actions=batch["actions"],
                old_log_probs=batch["log_probs"],
                old_values=batch["values"],
                advantages=batch["advantages"],
                returns=batch["returns"],
                action_mask=action_mask,
            )
        elif self.loss_type == "a2c":
            loss_info = self.loss_fn(
                policy_logits=output.policy_logits,
                values=output.value,
                actions=batch["actions"],
                advantages=batch["advantages"],
                returns=batch["returns"],
                action_mask=action_mask,
            )
        elif self.loss_type == "dmc":
            loss_info = self.loss_fn(
                policy_logits=output.policy_logits,
                actions=batch["actions"],
                rewards=batch["rewards"],
                action_mask=action_mask,
            )

        return loss_info

    def step(
        self,
        batch: Dict[str, torch.Tensor],
        action_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        执行一步训练

        Args:
            batch: 训练批次
            action_mask: 动作掩码

        Returns:
            损失字典
        """
        self.model.train()

        # 计算损失
        loss_info = self.compute_loss(batch, action_mask)

        # 梯度累积
        loss = loss_info.total_loss / self.accumulate_steps
        loss.backward()

        self._step_count += 1

        # 更新参数
        if self._step_count % self.accumulate_steps == 0:
            # 梯度裁剪
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.scheduler is not None:
                self.scheduler.step()

        return {
            "loss": loss_info.total_loss.item(),
            "policy_loss": loss_info.policy_loss.item(),
            "value_loss": loss_info.value_loss.item(),
            "entropy_loss": loss_info.entropy_loss.item(),
            "clip_fraction": loss_info.clip_fraction or 0.0,
            "approx_kl": loss_info.approx_kl or 0.0,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def train_epoch(
        self,
        data: Dict[str, torch.Tensor],
        batch_size: Optional[int] = None,
        shuffle: bool = True,
    ) -> Dict[str, float]:
        """
        训练一个 epoch

        Args:
            data: 完整训练数据
            batch_size: 批次大小
            shuffle: 是否打乱

        Returns:
            平均损失字典
        """
        batch_size = batch_size or self.config.batch_size
        n_samples = len(data["actions"])

        # 创建索引
        indices = torch.randperm(n_samples) if shuffle else torch.arange(n_samples)

        # 累积损失
        total_losses = {}
        n_batches = 0

        # 批次训练
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_indices = indices[start:end]

            # 提取批次
            batch = {k: v[batch_indices] for k, v in data.items()}

            # 训练步骤
            losses = self.step(batch)

            # 累积
            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v
            n_batches += 1

        # 平均
        return {k: v / n_batches for k, v in total_losses.items()}

    def save_checkpoint(self, path: str):
        """保存检查点"""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "step_count": self._step_count,
        }, path)

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self._step_count = checkpoint.get("step_count", 0)
