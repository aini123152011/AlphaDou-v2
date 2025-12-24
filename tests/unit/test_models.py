"""模型层测试"""
import pytest
import torch
import numpy as np


class TestModelSpec:
    """ModelSpec 测试"""

    def test_default_spec(self):
        from models.config import ModelSpec

        spec = ModelSpec()
        assert spec.backbone_type == "resnet"
        assert spec.hidden_dim == 512
        assert spec.num_layers == 4

    def test_from_dict(self):
        from models.config import ModelSpec

        config = {
            "backbone_type": "transformer",
            "hidden_dim": 256,
            "num_layers": 2,
            "extra_key": "ignored",
        }
        spec = ModelSpec.from_dict(config)
        assert spec.backbone_type == "transformer"
        assert spec.hidden_dim == 256

    def test_to_dict(self):
        from models.config import ModelSpec

        spec = ModelSpec(hidden_dim=1024)
        d = spec.to_dict()
        assert d["hidden_dim"] == 1024


class TestResNetBackbone:
    """ResNet 骨干网络测试"""

    def test_forward_shape(self):
        from models.backbone import ResNetBackbone

        backbone = ResNetBackbone(
            input_channels=40,
            hidden_dim=128,
            num_layers=2,
        )

        x = torch.randn(4, 40, 15)  # (batch, channels, seq_len)
        out = backbone(x)

        assert out.shape == (4, 128)

    def test_output_dim(self):
        from models.backbone import ResNetBackbone

        backbone = ResNetBackbone(hidden_dim=256)
        assert backbone.output_dim == 256

    def test_se_block(self):
        from models.backbone.resnet import SEBlock

        se = SEBlock(channels=64, reduction=4)
        x = torch.randn(2, 64, 10)
        out = se(x)

        assert out.shape == x.shape

    def test_res_block(self):
        from models.backbone.resnet import ResBlock

        block = ResBlock(channels=128, use_se=True)
        x = torch.randn(2, 128, 10)
        out = block(x)

        assert out.shape == x.shape


class TestTransformerBackbone:
    """Transformer 骨干网络测试"""

    def test_forward_shape(self):
        from models.backbone import TransformerBackbone

        backbone = TransformerBackbone(
            input_channels=40,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
        )

        x = torch.randn(4, 40, 15)
        out = backbone(x)

        assert out.shape == (4, 128)

    def test_attention_block(self):
        from models.backbone.transformer import TransformerBlock

        block = TransformerBlock(d_model=128, num_heads=4)
        x = torch.randn(2, 10, 128)
        out = block(x)

        assert out.shape == x.shape


class TestPolicyHead:
    """策略头测试"""

    def test_forward_shape(self):
        from models.heads import PolicyHead

        head = PolicyHead(
            input_dim=128,
            hidden_dim=64,
            output_dim=309,
        )

        x = torch.randn(4, 128)
        out = head(x)

        assert out.shape == (4, 309)

    def test_with_mask(self):
        from models.heads import PolicyHead

        head = PolicyHead(input_dim=128, output_dim=309)
        x = torch.randn(4, 128)

        mask = torch.zeros(4, 309)
        mask[:, :10] = 1  # 只有前 10 个动作合法

        out = head(x, mask)

        # 非法动作应为负无穷
        assert torch.isinf(out[:, 10]).all()
        assert not torch.isinf(out[:, 0]).any()


class TestValueHead:
    """价值头测试"""

    def test_forward_shape(self):
        from models.heads import ValueHead

        head = ValueHead(input_dim=128, hidden_dim=64)
        x = torch.randn(4, 128)
        out = head(x)

        assert out.shape == (4, 1)


class TestBidHead:
    """叫牌头测试"""

    def test_forward_shape(self):
        from models.heads import BidHead

        head = BidHead(input_dim=128, output_dim=4)
        x = torch.randn(4, 128)
        out = head(x)

        assert out.shape == (4, 4)


class TestObservationEncoder:
    """观测编码器测试"""

    def test_forward_shape(self):
        from models.doudizhu_model import ObservationEncoder

        encoder = ObservationEncoder(output_channels=40)

        obs = {
            "hand": torch.randn(4, 54),
            "played_cards": torch.randn(4, 3, 54),
            "history": torch.randn(4, 15, 54),
            "last_action": torch.randn(4, 54),
            "position": torch.randn(4, 6),
            "bid_info": torch.randn(4, 3),
            "cards_left": torch.randn(4, 3),
        }

        out = encoder(obs)
        assert out.shape[0] == 4
        assert out.shape[1] == 40


class TestDoudizhuModel:
    """DoudizhuModel 测试"""

    @pytest.fixture
    def sample_obs(self):
        return {
            "hand": torch.randn(4, 54),
            "played_cards": torch.randn(4, 3, 54),
            "history": torch.randn(4, 15, 54),
            "last_action": torch.randn(4, 54),
            "position": torch.randn(4, 6),
            "bid_info": torch.randn(4, 3),
            "cards_left": torch.randn(4, 3),
        }

    def test_forward(self, sample_obs):
        from models import build_model, ModelSpec

        spec = ModelSpec(hidden_dim=128, num_layers=2, action_dim=309)
        model = build_model(spec)

        output = model(sample_obs)

        assert output.policy_logits.shape == (4, 309)
        assert output.value.shape == (4, 1)

    def test_forward_with_bid(self, sample_obs):
        from models import build_model, ModelSpec

        spec = ModelSpec(hidden_dim=128, num_layers=2)
        model = build_model(spec)

        output = model(sample_obs, compute_bid=True)

        assert output.bid_logits is not None
        assert output.bid_logits.shape == (4, 4)

    def test_get_action(self, sample_obs):
        from models import build_model, ModelSpec

        spec = ModelSpec(hidden_dim=128, num_layers=2, action_dim=309)
        model = build_model(spec)

        action, log_prob, value = model.get_action(sample_obs)

        assert action.shape == (4,)
        assert log_prob.shape == (4,)
        assert value.shape == (4,)

    def test_get_action_deterministic(self, sample_obs):
        from models import build_model, ModelSpec

        spec = ModelSpec(hidden_dim=128, num_layers=2, action_dim=309)
        model = build_model(spec)
        model.eval()

        # 确定性模式应该产生相同结果
        action1, _, _ = model.get_action(sample_obs, deterministic=True)
        action2, _, _ = model.get_action(sample_obs, deterministic=True)

        assert torch.equal(action1, action2)


class TestDoudizhuModelSimple:
    """简化模型测试"""

    def test_forward(self):
        from models import DoudizhuModelSimple

        model = DoudizhuModelSimple(
            input_dim=1026,
            hidden_dim=256,
            action_dim=309,
        )

        obs = {
            "hand": torch.randn(4, 54),
            "played_cards": torch.randn(4, 3, 54),
            "history": torch.randn(4, 15, 54),
        }

        output = model(obs)

        assert output.policy_logits.shape == (4, 309)
        assert output.value.shape == (4, 1)


class TestModelRegistry:
    """模型注册表测试"""

    def test_get_instance(self):
        from models import get_registry

        registry1 = get_registry()
        registry2 = get_registry()

        assert registry1 is registry2  # 单例

    def test_list_backbones(self):
        from models import get_registry

        registry = get_registry()
        backbones = registry.list_backbones()

        assert "resnet" in backbones
        assert "transformer" in backbones

    def test_build_model(self):
        from models import build_model, ModelSpec

        spec = ModelSpec(backbone_type="resnet", hidden_dim=128)
        model = build_model(spec)

        assert model is not None

    def test_build_model_from_config(self):
        from models import build_model_from_config

        config = {
            "backbone_type": "resnet",
            "hidden_dim": 128,
            "num_layers": 2,
        }
        model = build_model_from_config(config)

        assert model is not None


class TestModelGradients:
    """梯度测试"""

    def test_backward(self):
        from models import build_model, ModelSpec

        spec = ModelSpec(hidden_dim=128, num_layers=2, action_dim=309)
        model = build_model(spec)

        obs = {
            "hand": torch.randn(4, 54),
            "played_cards": torch.randn(4, 3, 54),
            "history": torch.randn(4, 15, 54),
            "last_action": torch.randn(4, 54),
            "position": torch.randn(4, 6),
            "bid_info": torch.randn(4, 3),
            "cards_left": torch.randn(4, 3),
        }

        # 使用 compute_bid=True 确保所有头都参与前向传播
        output = model(obs, compute_bid=True)

        # 计算损失
        loss = output.policy_logits.mean() + output.value.mean() + output.bid_logits.mean()
        loss.backward()

        # 检查梯度
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
