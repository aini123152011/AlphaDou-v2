"""
Transformer 骨干网络

基于自注意力的编码器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        # Linear projections
        q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.w_o(context)


class FeedForward(nn.Module):
    """前馈网络"""

    def __init__(
        self,
        d_model: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer 块"""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attn(x, x, x, mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        # Feed-forward with residual
        ff_out = self.ff(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)

        return x


class TransformerBackbone(nn.Module):
    """
    Transformer 骨干网络

    结构:
        Input -> Embedding -> PositionalEncoding -> TransformerBlocks -> Pool -> Output

    输入: (batch, channels, seq_len) 或 (batch, seq_len, channels)
    输出: (batch, hidden_dim)
    """

    def __init__(
        self,
        input_channels: int = 40,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 100,
        **kwargs,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 输入投影
        self.input_proj = nn.Linear(input_channels, hidden_dim)

        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len, dropout)

        # Transformer 块
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # 输出归一化
        self.norm = nn.LayerNorm(hidden_dim)

        # CLS token (可选)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.output_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len) 或 (batch, seq_len, channels)
            mask: 可选的注意力掩码
        Returns:
            features: (batch, hidden_dim)
        """
        # 处理输入格式: 统一转为 (batch, seq_len, channels)
        if x.dim() == 3 and x.size(1) != x.size(2):
            # 如果 channels > seq_len，说明是 (batch, channels, seq_len)
            if x.size(1) > x.size(2):
                x = x.transpose(1, 2)  # -> (batch, seq_len, channels)

        batch_size = x.size(0)

        # 投影
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)

        # 添加 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 位置编码
        x = self.pos_encoding(x)

        # Transformer 块
        for layer in self.layers:
            x = layer(x, mask)

        # 归一化
        x = self.norm(x)

        # 使用 CLS token 作为输出
        return x[:, 0]


class TransformerBackboneV2(nn.Module):
    """
    Transformer V2 骨干网络

    使用平均池化而非 CLS token
    """

    def __init__(
        self,
        input_channels: int = 40,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 1D卷积输入投影 (更高效)
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        # Transformer 块
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        x = self.input_proj(x)  # (batch, hidden_dim, seq_len)
        x = x.transpose(1, 2)   # (batch, seq_len, hidden_dim)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # 平均池化
        return x.mean(dim=1)
