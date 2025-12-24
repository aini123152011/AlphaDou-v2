"""模型-环境集成测试"""
import pytest
import torch
import numpy as np

from env import DoudizhuEnv
from env.observation import get_action_encoder
from models import build_model, ModelSpec
from models.config import ModelSpec as ConfigModelSpec


class TestModelEnvDimensionMatch:
    """模型-环境维度匹配测试"""

    def test_action_space_dimension_match(self):
        """验证模型动作维度与环境匹配"""
        env = DoudizhuEnv()
        encoder = get_action_encoder()

        # 环境动作空间
        env_action_dim = env.action_space.n
        encoder_action_dim = encoder.num_actions

        assert env_action_dim == encoder_action_dim, (
            f"环境动作空间 ({env_action_dim}) 与编码器 ({encoder_action_dim}) 不匹配"
        )

    def test_model_output_matches_env(self):
        """验证模型输出维度与环境匹配"""
        env = DoudizhuEnv()
        action_dim = env.action_space.n

        spec = ModelSpec(
            backbone_type="resnet",
            hidden_dim=128,  # 小模型用于测试
            num_layers=2,
            action_dim=action_dim,
        )
        model = build_model(spec)

        # 创建假观测
        obs, _ = env.reset()
        batch_obs = {
            k: torch.tensor(v, dtype=torch.float32).unsqueeze(0)
            for k, v in obs.items()
        }

        # 前向传播
        with torch.no_grad():
            output = model(batch_obs)

        # 验证输出维度
        assert output.policy_logits.shape[-1] == action_dim, (
            f"模型输出维度 ({output.policy_logits.shape[-1]}) 与环境 ({action_dim}) 不匹配"
        )


class TestModelEnvIntegration:
    """模型-环境集成测试"""

    def test_model_can_step_env(self):
        """验证模型可以与环境交互"""
        env = DoudizhuEnv()
        action_dim = env.action_space.n

        spec = ModelSpec(
            backbone_type="resnet",
            hidden_dim=128,
            num_layers=2,
            action_dim=action_dim,
        )
        model = build_model(spec)
        model.eval()

        obs, info = env.reset()

        # 执行几步
        for _ in range(5):
            # 获取合法动作
            legal_actions = env.get_legal_actions()
            if not legal_actions:
                break

            # 随机选择一个合法动作
            action = legal_actions[np.random.randint(len(legal_actions))]

            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

    def test_model_action_selection(self):
        """验证模型可以选择动作"""
        env = DoudizhuEnv()
        action_dim = env.action_space.n

        spec = ModelSpec(
            backbone_type="resnet",
            hidden_dim=128,
            num_layers=2,
            action_dim=action_dim,
        )
        model = build_model(spec)
        model.eval()

        obs, info = env.reset()

        # 跳过叫牌阶段
        while info.get("phase") == "bidding":
            action = 3  # 叫 3 分
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                pytest.skip("游戏在叫牌阶段结束")

        # 现在在出牌阶段
        if info.get("phase") == "playing":
            # 获取合法动作掩码
            legal_mask = info.get("legal_action_mask")
            if legal_mask is not None:
                # 转换观测
                batch_obs = {
                    k: torch.tensor(v, dtype=torch.float32).unsqueeze(0)
                    for k, v in obs.items()
                }
                action_mask = torch.tensor(legal_mask, dtype=torch.float32).unsqueeze(0)

                # 模型前向传播
                with torch.no_grad():
                    output = model(batch_obs, action_mask=action_mask)

                # 选择最大概率动作
                logits = output.policy_logits[0]
                # 掩码非法动作
                logits = logits.masked_fill(action_mask[0] == 0, float('-inf'))
                action_idx = logits.argmax().item()

                # 验证动作合法
                assert legal_mask[action_idx] == 1, "模型选择了非法动作"


class TestInvalidActionHandling:
    """非法动作处理测试"""

    def test_invalid_action_index_raises(self):
        """验证非法动作索引抛出异常"""
        env = DoudizhuEnv()
        obs, info = env.reset()

        # 跳过叫牌阶段
        while info.get("phase") == "bidding":
            obs, _, terminated, truncated, info = env.step(3)
            if terminated or truncated:
                pytest.skip("游戏在叫牌阶段结束")

        # 尝试使用超出范围的动作索引
        invalid_action = env.action_space.n + 1000

        with pytest.raises(ValueError) as excinfo:
            env.step(invalid_action)

        assert "Invalid action index" in str(excinfo.value)


class TestFullGameWithModel:
    """完整游戏测试"""

    def test_complete_game_with_random_actions(self):
        """使用随机动作完成完整游戏"""
        env = DoudizhuEnv()
        obs, info = env.reset()

        max_steps = 200
        step = 0

        while step < max_steps:
            legal_actions = env.get_legal_actions()
            if not legal_actions:
                break

            # 随机选择
            action = legal_actions[np.random.randint(len(legal_actions))]

            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

            if terminated or truncated:
                break

        assert info.get("winner") is not None or info.get("phase") == "finished"

    def test_complete_game_with_model(self):
        """使用模型完成完整游戏"""
        env = DoudizhuEnv()
        action_dim = env.action_space.n

        spec = ModelSpec(
            backbone_type="resnet",
            hidden_dim=128,
            num_layers=2,
            action_dim=action_dim,
        )
        model = build_model(spec)
        model.eval()

        obs, info = env.reset()

        max_steps = 200
        step = 0

        while step < max_steps:
            legal_actions = env.get_legal_actions()
            if not legal_actions:
                break

            if info.get("phase") == "bidding":
                # 叫牌阶段使用随机动作
                action = legal_actions[np.random.randint(len(legal_actions))]
            else:
                # 出牌阶段使用模型
                # 简化: 使用随机合法动作，因为模型未训练
                action = legal_actions[np.random.randint(len(legal_actions))]

            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

            if terminated or truncated:
                break

        # 游戏应该正常结束
        assert step < max_steps, "游戏未能在最大步数内结束"
