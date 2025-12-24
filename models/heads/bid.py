"""
叫牌头

输出叫牌动作概率
"""
import torch
import torch.nn as nn
from typing import Optional


class BidHead(nn.Module):
    """
    叫牌头

    输出叫牌分数的概率分布 (0, 1, 2, 3)
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        output_dim: int = 4,  # 0=不叫, 1/2/3=叫分
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度 (叫牌选项数)
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
            mask: (batch, output_dim) 合法叫牌掩码

        Returns:
            logits: (batch, output_dim) 叫牌 logits
        """
        logits = self.fc(x)

        if mask is not None:
            logits = logits.masked_fill(mask == 0, float('-inf'))

        return logits


class BidValueHead(nn.Module):
    """
    叫牌价值头

    同时输出叫牌策略和期望价值
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        output_dim: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.policy = nn.Linear(hidden_dim, output_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Args:
            x: (batch, input_dim)
            mask: (batch, output_dim) 合法叫牌掩码

        Returns:
            (bid_logits, bid_value) 元组
        """
        features = self.shared(x)

        logits = self.policy(features)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float('-inf'))

        value = self.value(features)

        return logits, value
