"""
模型配置

定义模型规格和超参数
"""
from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any


@dataclass
class ModelSpec:
    """
    模型规格配置

    Attributes:
        backbone_type: 骨干网络类型
        input_channels: 输入特征通道数
        hidden_dim: 隐藏层维度
        num_layers: 骨干网络层数
        action_dim: 动作空间维度
        num_heads: Transformer 注意力头数
        dropout: Dropout 概率
        use_se: 是否使用 SE 注意力
    """
    # 骨干网络类型
    backbone_type: Literal["resnet", "transformer", "lstm"] = "resnet"

    # 输入维度
    input_channels: int = 40

    # 隐藏层维度
    hidden_dim: int = 512

    # 骨干网络层数
    num_layers: int = 4

    # 动作空间维度 (应由环境动态提供)
    # 注意: 完整动作空间约 14636，需要与 env.action_space.n 一致
    action_dim: int = 0  # 0 表示需要动态设置

    # Transformer 参数
    num_heads: int = 8
    ff_dim: int = 2048

    # LSTM 参数
    lstm_hidden: int = 256
    lstm_layers: int = 2

    # 正则化
    dropout: float = 0.1

    # ResNet 参数
    use_se: bool = True  # Squeeze-and-Excitation

    # 是否共享骨干
    share_backbone: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModelSpec':
        """从字典创建配置"""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "backbone_type": self.backbone_type,
            "input_channels": self.input_channels,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "action_dim": self.action_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
            "use_se": self.use_se,
        }


@dataclass
class HeadSpec:
    """
    头部网络配置

    Attributes:
        input_dim: 输入维度
        hidden_dims: 隐藏层维度列表
        output_dim: 输出维度
        activation: 激活函数
        dropout: Dropout 概率
    """
    input_dim: int = 512
    hidden_dims: tuple = (256,)
    output_dim: int = 1
    activation: str = "relu"
    dropout: float = 0.1


# 预定义配置
RESNET_SMALL = ModelSpec(
    backbone_type="resnet",
    hidden_dim=256,
    num_layers=2,
)

RESNET_BASE = ModelSpec(
    backbone_type="resnet",
    hidden_dim=512,
    num_layers=4,
)

RESNET_LARGE = ModelSpec(
    backbone_type="resnet",
    hidden_dim=1024,
    num_layers=8,
)

TRANSFORMER_BASE = ModelSpec(
    backbone_type="transformer",
    hidden_dim=512,
    num_layers=4,
    num_heads=8,
)
