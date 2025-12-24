"""
Model Layer - 神经网络模型

Modules:
    config: 模型配置
    registry: 模型注册与工厂
    backbone: 骨干网络 (ResNet, Transformer)
    heads: 输出头 (Policy, Value, Bid)
    doudizhu_model: 统一模型
"""
from .config import ModelSpec, HeadSpec, RESNET_BASE, RESNET_SMALL, TRANSFORMER_BASE
from .registry import (
    ModelRegistry,
    build_model,
    build_model_from_config,
    get_registry,
    BACKBONES,
)
from .doudizhu_model import (
    DoudizhuModel,
    DoudizhuModelSimple,
    ModelOutput,
    ObservationEncoder,
)
from .backbone import (
    ResNetBackbone,
    ResNetBackboneV2,
    TransformerBackbone,
    TransformerBackboneV2,
)
from .heads import (
    PolicyHead,
    ValueHead,
    BidHead,
)

__all__ = [
    # config
    "ModelSpec",
    "HeadSpec",
    "RESNET_BASE",
    "RESNET_SMALL",
    "TRANSFORMER_BASE",
    # registry
    "ModelRegistry",
    "build_model",
    "build_model_from_config",
    "get_registry",
    "BACKBONES",
    # model
    "DoudizhuModel",
    "DoudizhuModelSimple",
    "ModelOutput",
    "ObservationEncoder",
    # backbone
    "ResNetBackbone",
    "ResNetBackboneV2",
    "TransformerBackbone",
    "TransformerBackboneV2",
    # heads
    "PolicyHead",
    "ValueHead",
    "BidHead",
]
