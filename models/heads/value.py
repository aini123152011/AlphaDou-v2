"""
价值头

输出状态价值估计
"""
import torch
import torch.nn as nn
from typing import Optional


class ValueHead(nn.Module):
    """
    价值头

    输出单一标量价值估计
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            dropout: Dropout 概率
        """
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) 骨干网络输出

        Returns:
            value: (batch, 1) 状态价值
        """
        return self.fc(x)


class DistributionalValueHead(nn.Module):
    """
    分布式价值头

    输出价值分布而非点估计 (C51 style)
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_atoms),
        )

        # 支撑点
        self.register_buffer(
            'support',
            torch.linspace(v_min, v_max, num_atoms)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)

        Returns:
            value: (batch, 1) 期望价值
        """
        logits = self.fc(x)  # (batch, num_atoms)
        probs = torch.softmax(logits, dim=-1)
        value = (probs * self.support).sum(dim=-1, keepdim=True)
        return value

    def get_distribution(self, x: torch.Tensor) -> torch.Tensor:
        """获取完整分布"""
        logits = self.fc(x)
        return torch.softmax(logits, dim=-1)


class MultiHeadValueHead(nn.Module):
    """
    多头价值头

    为不同角色输出独立的价值估计
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_heads: int = 3,  # landlord, down, up
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_heads = num_heads

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1)
            for _ in range(num_heads)
        ])

    def forward(
        self,
        x: torch.Tensor,
        head_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)
            head_idx: 指定的头索引，None 返回所有

        Returns:
            value: (batch, 1) 或 (batch, num_heads)
        """
        features = self.shared(x)

        if head_idx is not None:
            return self.heads[head_idx](features)

        values = torch.cat([
            head(features) for head in self.heads
        ], dim=-1)

        return values
