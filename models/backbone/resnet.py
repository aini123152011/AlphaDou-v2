"""
ResNet 骨干网络

基于 1D 卷积的残差网络，适用于序列特征处理
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block

    通道注意力机制，自适应地重新校准通道特征响应
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            (batch, channels, seq_len)
        """
        # Global average pooling
        y = x.mean(dim=-1)  # (batch, channels)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y.unsqueeze(-1)


class ResBlock(nn.Module):
    """
    残差块

    包含两个卷积层和可选的 SE 注意力
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        use_se: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(channels)

        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(channels)

        self.se = SEBlock(channels) if use_se else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.act = nn.Mish(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out = out + residual
        out = self.act(out)

        return out


class ResNetBackbone(nn.Module):
    """
    ResNet 骨干网络

    结构:
        Input -> Stem -> ResBlocks -> Pool -> Output

    输入: (batch, channels, seq_len)
    输出: (batch, hidden_dim)
    """

    def __init__(
        self,
        input_channels: int = 40,
        hidden_dim: int = 512,
        num_layers: int = 4,
        use_se: bool = True,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Stem: 初始卷积
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.Mish(inplace=True),
        )

        # 残差块
        self.layers = nn.ModuleList([
            ResBlock(hidden_dim, use_se=use_se, dropout=dropout)
            for _ in range(num_layers)
        ])

        # 全局池化
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 输出维度
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            features: (batch, hidden_dim)
        """
        x = self.stem(x)

        for layer in self.layers:
            x = layer(x)

        x = self.pool(x).squeeze(-1)

        return x


class ResNetBackboneV2(nn.Module):
    """
    ResNet V2 骨干网络

    改进版本，带有下采样和多尺度特征
    """

    def __init__(
        self,
        input_channels: int = 40,
        hidden_dim: int = 512,
        num_layers: int = 4,
        use_se: bool = True,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        # 阶段通道数
        channels = [hidden_dim // 4, hidden_dim // 2, hidden_dim, hidden_dim]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, channels[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.Mish(inplace=True),
        )

        # 构建各阶段
        self.stages = nn.ModuleList()
        layers_per_stage = max(1, num_layers // 4)

        for i in range(4):
            in_ch = channels[i-1] if i > 0 else channels[0]
            out_ch = channels[i]
            stride = 2 if i > 0 else 1

            stage = self._make_stage(
                in_ch, out_ch, layers_per_stage,
                stride=stride, use_se=use_se, dropout=dropout
            )
            self.stages.append(stage)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = hidden_dim

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
        use_se: bool = True,
        dropout: float = 0.1,
    ) -> nn.Sequential:
        """构建一个阶段"""
        layers = []

        # 下采样
        if stride > 1 or in_channels != out_channels:
            layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.Mish(inplace=True),
            ))

        # 残差块
        for _ in range(num_blocks):
            layers.append(ResBlock(out_channels, use_se=use_se, dropout=dropout))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        for stage in self.stages:
            x = stage(x)

        x = self.pool(x).squeeze(-1)

        return x
