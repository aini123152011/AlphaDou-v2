"""
模型注册与工厂

提供模型创建和管理功能
"""
from typing import Dict, Type, Optional
import torch.nn as nn

from .config import ModelSpec
from .backbone import ResNetBackbone, ResNetBackboneV2, TransformerBackbone, TransformerBackboneV2
from .heads import PolicyHead, ValueHead, BidHead
from .doudizhu_model import DoudizhuModel, DoudizhuModelSimple, ObservationEncoder


# 骨干网络注册表
BACKBONES: Dict[str, Type[nn.Module]] = {
    "resnet": ResNetBackbone,
    "resnet_v2": ResNetBackboneV2,
    "transformer": TransformerBackbone,
    "transformer_v2": TransformerBackboneV2,
}


class ModelRegistry:
    """
    模型注册与工厂

    单例模式管理模型类的注册和创建
    """

    _instance: Optional['ModelRegistry'] = None

    def __init__(self):
        self._backbones: Dict[str, Type[nn.Module]] = {}
        self._models: Dict[str, Type[nn.Module]] = {}
        self._register_defaults()

    @classmethod
    def get_instance(cls) -> 'ModelRegistry':
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _register_defaults(self):
        """注册默认组件"""
        for name, backbone_cls in BACKBONES.items():
            self._backbones[name] = backbone_cls

        self._models["doudizhu"] = DoudizhuModel
        self._models["simple"] = DoudizhuModelSimple

    def register_backbone(self, name: str, backbone_cls: Type[nn.Module]):
        """注册骨干网络"""
        self._backbones[name] = backbone_cls

    def register_model(self, name: str, model_cls: Type[nn.Module]):
        """注册模型"""
        self._models[name] = model_cls

    def get_backbone(self, name: str) -> Type[nn.Module]:
        """获取骨干网络类"""
        if name not in self._backbones:
            raise ValueError(f"Unknown backbone: {name}")
        return self._backbones[name]

    def list_backbones(self) -> list:
        """列出所有注册的骨干网络"""
        return list(self._backbones.keys())

    def build_model(self, spec: ModelSpec) -> DoudizhuModel:
        """
        根据配置构建模型

        Args:
            spec: 模型规格配置

        Returns:
            DoudizhuModel 实例
        """
        # 获取骨干网络类
        backbone_cls = self._backbones.get(spec.backbone_type)
        if backbone_cls is None:
            raise ValueError(f"Unknown backbone type: {spec.backbone_type}")

        # 构建骨干网络
        backbone = backbone_cls(
            input_channels=spec.input_channels,
            hidden_dim=spec.hidden_dim,
            num_layers=spec.num_layers,
            dropout=spec.dropout,
            use_se=spec.use_se,
            num_heads=spec.num_heads,
            ff_dim=spec.ff_dim,
        )

        # 构建输出头
        policy_head = PolicyHead(
            input_dim=spec.hidden_dim,
            hidden_dim=spec.hidden_dim // 2,
            output_dim=spec.action_dim,
            dropout=spec.dropout,
        )

        value_head = ValueHead(
            input_dim=spec.hidden_dim,
            hidden_dim=spec.hidden_dim // 2,
            dropout=spec.dropout,
        )

        bid_head = BidHead(
            input_dim=spec.hidden_dim,
            hidden_dim=spec.hidden_dim // 4,
            output_dim=4,
            dropout=spec.dropout,
        )

        # 构建观测编码器
        obs_encoder = ObservationEncoder(
            output_channels=spec.input_channels
        )

        return DoudizhuModel(
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
            bid_head=bid_head,
            obs_encoder=obs_encoder,
        )

    def build_simple_model(
        self,
        input_dim: int = 1026,
        hidden_dim: int = 512,
        action_dim: int = 309,
    ) -> DoudizhuModelSimple:
        """构建简化模型"""
        return DoudizhuModelSimple(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
        )


def build_model(spec: ModelSpec) -> DoudizhuModel:
    """
    便捷函数：构建模型

    Args:
        spec: 模型规格配置

    Returns:
        DoudizhuModel 实例
    """
    return ModelRegistry.get_instance().build_model(spec)


def build_model_from_config(config: Dict) -> DoudizhuModel:
    """
    从字典配置构建模型

    Args:
        config: 配置字典

    Returns:
        DoudizhuModel 实例
    """
    spec = ModelSpec.from_dict(config)
    return build_model(spec)


def get_registry() -> ModelRegistry:
    """获取模型注册表"""
    return ModelRegistry.get_instance()
