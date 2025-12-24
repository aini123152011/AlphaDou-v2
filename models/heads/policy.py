"""
策略头

输出动作概率分布
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PolicyHead(nn.Module):
    """
    策略头

    将骨干网络输出映射到动作空间的 logits
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 309,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 动作空间维度
            dropout: Dropout 概率
        """
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) 骨干网络输出
            mask: (batch, output_dim) 合法动作掩码

        Returns:
            logits: (batch, output_dim) 动作 logits
        """
        logits = self.fc(x)

        if mask is not None:
            # 将非法动作的 logits 设为负无穷
            logits = logits.masked_fill(mask == 0, float('-inf'))

        return logits


class DuelingPolicyHead(nn.Module):
    """
    Dueling 策略头

    分离状态价值和动作优势
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 309,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 状态价值流
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # 动作优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        value = self.value_stream(x)  # (batch, 1)
        advantage = self.advantage_stream(x)  # (batch, output_dim)

        # Q = V + (A - mean(A))
        logits = value + advantage - advantage.mean(dim=-1, keepdim=True)

        if mask is not None:
            logits = logits.masked_fill(mask == 0, float('-inf'))

        return logits


class AttentionPolicyHead(nn.Module):
    """
    注意力策略头

    使用注意力机制聚合特征
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 309,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads,
            dropout=dropout, batch_first=True
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (batch, input_dim)
        x = x.unsqueeze(1)  # (batch, 1, input_dim)

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_out, _ = self.attn(q, k, v)
        attn_out = attn_out.squeeze(1)

        logits = self.output(attn_out)

        if mask is not None:
            logits = logits.masked_fill(mask == 0, float('-inf'))

        return logits
