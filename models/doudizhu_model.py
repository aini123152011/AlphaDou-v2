"""
斗地主统一模型

将骨干网络和各种头组合成完整模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

from .config import ModelSpec
from .backbone import ResNetBackbone, TransformerBackbone
from .heads import PolicyHead, ValueHead, BidHead


@dataclass
class ModelOutput:
    """
    模型输出

    Attributes:
        policy_logits: 策略 logits
        value: 状态价值
        bid_logits: 叫牌 logits (可选)
    """
    policy_logits: torch.Tensor
    value: torch.Tensor
    bid_logits: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, torch.Tensor]:
        d = {
            "policy_logits": self.policy_logits,
            "value": self.value,
        }
        if self.bid_logits is not None:
            d["bid_logits"] = self.bid_logits
        return d


class ObservationEncoder(nn.Module):
    """
    观测编码器

    将字典形式的观测转换为网络输入张量
    """

    def __init__(self, output_channels: int = 40):
        super().__init__()
        self.output_channels = output_channels

        # 各特征的编码层
        self.hand_enc = nn.Linear(54, 64)
        self.played_enc = nn.Linear(162, 64)  # 3 * 54
        self.history_enc = nn.Linear(810, 128)  # 15 * 54
        self.last_enc = nn.Linear(54, 32)
        self.position_enc = nn.Linear(6, 16)
        self.bid_enc = nn.Linear(3, 8)
        self.cards_left_enc = nn.Linear(3, 8)

        # 融合层
        total_dim = 64 + 64 + 128 + 32 + 16 + 8 + 8  # 320
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, output_channels * 15),
            nn.ReLU(inplace=True),
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            obs: 观测字典

        Returns:
            x: (batch, output_channels, seq_len) 用于 1D Conv
        """
        batch_size = obs["hand"].shape[0]

        # 编码各特征
        hand = self.hand_enc(obs["hand"])
        played = self.played_enc(obs["played_cards"].view(batch_size, -1))
        history = self.history_enc(obs["history"].view(batch_size, -1))
        last = self.last_enc(obs["last_action"])
        position = self.position_enc(obs["position"])
        bid = self.bid_enc(obs["bid_info"])
        cards_left = self.cards_left_enc(obs["cards_left"])

        # 拼接
        combined = torch.cat([
            hand, played, history, last, position, bid, cards_left
        ], dim=-1)

        # 融合并重塑
        x = self.fusion(combined)
        x = x.view(batch_size, self.output_channels, -1)

        return x


class DoudizhuModel(nn.Module):
    """
    斗地主统一模型

    结构:
        Observation -> Encoder -> Backbone -> [PolicyHead, ValueHead, BidHead]
    """

    def __init__(
        self,
        backbone: nn.Module,
        policy_head: nn.Module,
        value_head: nn.Module,
        bid_head: Optional[nn.Module] = None,
        obs_encoder: Optional[nn.Module] = None,
    ):
        """
        Args:
            backbone: 骨干网络
            policy_head: 策略头
            value_head: 价值头
            bid_head: 叫牌头 (可选)
            obs_encoder: 观测编码器 (可选)
        """
        super().__init__()

        self.obs_encoder = obs_encoder or ObservationEncoder()
        self.backbone = backbone
        self.policy_head = policy_head
        self.value_head = value_head
        self.bid_head = bid_head

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        action_mask: Optional[torch.Tensor] = None,
        compute_bid: bool = False,
    ) -> ModelOutput:
        """
        前向传播

        Args:
            obs: 观测字典
            action_mask: 合法动作掩码
            compute_bid: 是否计算叫牌头

        Returns:
            ModelOutput
        """
        # 编码观测
        x = self.obs_encoder(obs)

        # 骨干网络
        features = self.backbone(x)

        # 策略头
        policy_logits = self.policy_head(features, action_mask)

        # 价值头
        value = self.value_head(features)

        # 叫牌头
        bid_logits = None
        if compute_bid and self.bid_head is not None:
            bid_logits = self.bid_head(features)

        return ModelOutput(
            policy_logits=policy_logits,
            value=value,
            bid_logits=bid_logits,
        )

    def get_action(
        self,
        obs: Dict[str, torch.Tensor],
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        采样动作

        Args:
            obs: 观测
            action_mask: 合法动作掩码
            deterministic: 是否确定性选择

        Returns:
            (action, log_prob, value) 元组
        """
        output = self.forward(obs, action_mask)

        # 计算概率分布
        probs = F.softmax(output.policy_logits, dim=-1)

        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

        # 计算 log prob
        log_prob = F.log_softmax(output.policy_logits, dim=-1)
        log_prob = log_prob.gather(-1, action.unsqueeze(-1)).squeeze(-1)

        return action, log_prob, output.value.squeeze(-1)

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作

        Args:
            obs: 观测
            actions: 动作
            action_mask: 合法动作掩码

        Returns:
            (log_prob, entropy, value) 元组
        """
        output = self.forward(obs, action_mask)

        # 计算 log prob
        log_prob = F.log_softmax(output.policy_logits, dim=-1)
        log_prob = log_prob.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        # 计算熵
        probs = F.softmax(output.policy_logits, dim=-1)
        entropy = -(probs * log_prob).sum(dim=-1)

        return log_prob, entropy, output.value.squeeze(-1)


class DoudizhuModelSimple(nn.Module):
    """
    简化版斗地主模型

    使用 MLP 而非复杂骨干，适用于快速实验
    """

    def __init__(
        self,
        input_dim: int = 1026,  # 54 + 162 + 810 (默认)
        hidden_dim: int = 512,
        action_dim: int = 309,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.bid_head = nn.Linear(hidden_dim, 4)

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        action_mask: Optional[torch.Tensor] = None,
        compute_bid: bool = False,
    ) -> ModelOutput:
        # 展平观测
        batch_size = obs["hand"].shape[0]
        x = torch.cat([
            obs["hand"],
            obs["played_cards"].view(batch_size, -1),
            obs["history"].view(batch_size, -1),
        ], dim=-1)

        features = self.shared(x)

        policy_logits = self.policy_head(features)
        if action_mask is not None:
            policy_logits = policy_logits.masked_fill(action_mask == 0, float('-inf'))

        value = self.value_head(features)

        bid_logits = None
        if compute_bid:
            bid_logits = self.bid_head(features)

        return ModelOutput(
            policy_logits=policy_logits,
            value=value,
            bid_logits=bid_logits,
        )
