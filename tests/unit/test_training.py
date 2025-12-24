"""训练层测试"""
import pytest
import torch
import numpy as np


class TestTrainConfig:
    """训练配置测试"""

    def test_default_config(self):
        from training.config import TrainConfig

        config = TrainConfig()
        assert config.learning_rate == 1e-4
        assert config.batch_size == 256
        assert config.gamma == 0.99

    def test_from_dict(self):
        from training.config import TrainConfig

        d = {"learning_rate": 1e-3, "batch_size": 128, "extra": "ignored"}
        config = TrainConfig.from_dict(d)
        assert config.learning_rate == 1e-3
        assert config.batch_size == 128


class TestTransition:
    """Transition 测试"""

    def test_create(self):
        from training.buffer import Transition

        obs = {"hand": np.zeros(54), "played_cards": np.zeros((3, 54))}
        t = Transition(
            obs=obs,
            action=0,
            reward=1.0,
            next_obs=None,
            done=True,
        )
        assert t.action == 0
        assert t.reward == 1.0
        assert t.done is True


class TestTrajectory:
    """Trajectory 测试"""

    def test_add_transitions(self):
        from training.buffer import Transition, Trajectory

        traj = Trajectory()
        obs = {"hand": np.zeros(54)}

        for i in range(5):
            t = Transition(
                obs=obs,
                action=i,
                reward=0.1,
                next_obs=obs,
                done=False,
            )
            traj.add(t)

        assert len(traj) == 5
        assert traj.total_reward == 0.5

    def test_compute_returns(self):
        from training.buffer import Transition, Trajectory

        traj = Trajectory()
        obs = {"hand": np.zeros(54)}

        rewards = [1.0, 0.0, 0.0, 0.0, 10.0]
        for i, r in enumerate(rewards):
            t = Transition(
                obs=obs,
                action=0,
                reward=r,
                next_obs=None if i == len(rewards) - 1 else obs,
                done=i == len(rewards) - 1,
            )
            traj.add(t)

        traj.compute_returns(gamma=0.99)

        # 最后一步的 return 应该等于 reward
        assert abs(traj.transitions[-1].returns - 10.0) < 0.01

    def test_to_batch(self):
        from training.buffer import Transition, Trajectory

        traj = Trajectory()
        obs = {"hand": np.random.randn(54).astype(np.float32)}

        for i in range(3):
            t = Transition(
                obs=obs.copy(),
                action=i,
                reward=float(i),
                next_obs=obs.copy(),
                done=i == 2,
            )
            traj.add(t)

        batch = traj.to_batch()

        assert "hand" in batch
        assert batch["hand"].shape == (3, 54)
        assert batch["actions"].shape == (3,)
        assert batch["rewards"].shape == (3,)


class TestReplayBuffer:
    """ReplayBuffer 测试"""

    def test_add_and_sample(self):
        from training.buffer import Transition, ReplayBuffer

        buffer = ReplayBuffer(capacity=100, batch_size=8)
        obs = {"hand": np.zeros(54, dtype=np.float32)}

        for i in range(20):
            t = Transition(obs=obs, action=i % 10, reward=0.1, next_obs=obs, done=False)
            buffer.add(t)

        assert len(buffer) == 20
        assert buffer.is_ready()

        batch = buffer.sample()
        assert "actions" in batch
        assert batch["actions"].shape[0] == 8

    def test_capacity(self):
        from training.buffer import Transition, ReplayBuffer

        buffer = ReplayBuffer(capacity=10, batch_size=4)
        obs = {"hand": np.zeros(54, dtype=np.float32)}

        for i in range(20):
            t = Transition(obs=obs, action=i, reward=0.1, next_obs=obs, done=False)
            buffer.add(t)

        # 超出容量后应该保持在 capacity
        assert len(buffer) == 10


class TestRolloutBuffer:
    """RolloutBuffer 测试"""

    def test_add_and_get(self):
        from training.buffer import RolloutBuffer

        obs_shape = {"hand": (54,), "played_cards": (3, 54)}
        buffer = RolloutBuffer(buffer_size=100, obs_shape=obs_shape)

        for i in range(50):
            obs = {
                "hand": np.random.randn(54).astype(np.float32),
                "played_cards": np.random.randn(3, 54).astype(np.float32),
            }
            buffer.add(
                obs=obs,
                action=i % 10,
                reward=0.1,
                done=i % 10 == 9,
                value=0.5,
                log_prob=-0.1,
            )

        buffer.compute_returns_and_advantages(last_value=0.0)
        data = buffer.get()

        assert "hand" in data
        assert data["hand"].shape == (50, 54)
        assert data["advantages"].shape == (50,)
        assert data["returns"].shape == (50,)


class TestPrioritizedReplayBuffer:
    """PrioritizedReplayBuffer 测试"""

    def test_priority_sampling(self):
        from training.buffer import Transition, PrioritizedReplayBuffer

        buffer = PrioritizedReplayBuffer(capacity=100, batch_size=8)
        obs = {"hand": np.zeros(54, dtype=np.float32)}

        for i in range(20):
            t = Transition(obs=obs, action=i % 10, reward=0.1, next_obs=obs, done=False)
            buffer.add(t)

        batch, indices, weights = buffer.sample()

        assert len(indices) == 8
        assert len(weights) == 8
        assert np.all(weights > 0)

    def test_update_priorities(self):
        from training.buffer import Transition, PrioritizedReplayBuffer

        buffer = PrioritizedReplayBuffer(capacity=100, batch_size=4)
        obs = {"hand": np.zeros(54, dtype=np.float32)}

        for i in range(10):
            t = Transition(obs=obs, action=i, reward=0.1, next_obs=obs, done=False)
            buffer.add(t)

        _, indices, _ = buffer.sample()
        new_priorities = np.ones(4) * 10.0
        buffer.update_priorities(indices, new_priorities)

        assert buffer.max_priority == 10.0


class TestPPOLoss:
    """PPO 损失测试"""

    def test_forward(self):
        from training.learner import PPOLoss

        loss_fn = PPOLoss(clip_range=0.2)

        batch_size = 16
        action_dim = 309

        policy_logits = torch.randn(batch_size, action_dim, requires_grad=True)
        values = torch.randn(batch_size, 1, requires_grad=True)
        actions = torch.randint(0, action_dim, (batch_size,))
        old_log_probs = torch.randn(batch_size)
        old_values = torch.randn(batch_size)
        advantages = torch.randn(batch_size)
        returns = torch.randn(batch_size)

        loss_info = loss_fn(
            policy_logits=policy_logits,
            values=values,
            actions=actions,
            old_log_probs=old_log_probs,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
        )

        assert loss_info.total_loss.requires_grad
        assert loss_info.clip_fraction is not None

    def test_with_mask(self):
        from training.learner import PPOLoss

        loss_fn = PPOLoss()

        batch_size = 4
        action_dim = 10

        policy_logits = torch.randn(batch_size, action_dim, requires_grad=True)
        values = torch.randn(batch_size, 1, requires_grad=True)
        actions = torch.zeros(batch_size, dtype=torch.long)
        old_log_probs = torch.zeros(batch_size)
        old_values = torch.zeros(batch_size)
        advantages = torch.ones(batch_size)
        returns = torch.ones(batch_size)

        # 只允许动作 0
        mask = torch.zeros(batch_size, action_dim)
        mask[:, 0] = 1

        loss_info = loss_fn(
            policy_logits=policy_logits,
            values=values,
            actions=actions,
            old_log_probs=old_log_probs,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            action_mask=mask,
        )

        assert not torch.isnan(loss_info.total_loss)


class TestA2CLoss:
    """A2C 损失测试"""

    def test_forward(self):
        from training.learner import A2CLoss

        loss_fn = A2CLoss()

        batch_size = 16
        action_dim = 100

        policy_logits = torch.randn(batch_size, action_dim, requires_grad=True)
        values = torch.randn(batch_size, 1, requires_grad=True)
        actions = torch.randint(0, action_dim, (batch_size,))
        advantages = torch.randn(batch_size)
        returns = torch.randn(batch_size)

        loss_info = loss_fn(
            policy_logits=policy_logits,
            values=values,
            actions=actions,
            advantages=advantages,
            returns=returns,
        )

        assert loss_info.total_loss.requires_grad


class TestDMCLoss:
    """DMC 损失测试"""

    def test_forward(self):
        from training.learner import DMCLoss

        loss_fn = DMCLoss()

        batch_size = 16
        action_dim = 309

        policy_logits = torch.randn(batch_size, action_dim, requires_grad=True)
        actions = torch.randint(0, action_dim, (batch_size,))
        rewards = torch.randn(batch_size)

        loss_info = loss_fn(
            policy_logits=policy_logits,
            actions=actions,
            rewards=rewards,
        )

        assert loss_info.total_loss.requires_grad


class TestLearner:
    """Learner 测试"""

    @pytest.fixture
    def simple_model(self):
        """简单模型"""
        from models import DoudizhuModelSimple

        return DoudizhuModelSimple(
            input_dim=54 + 3 * 54 + 15 * 54,
            hidden_dim=64,
            action_dim=309,
        )

    @pytest.fixture
    def config(self):
        from training.config import TrainConfig

        return TrainConfig(learning_rate=1e-3, batch_size=8)

    def test_compute_loss(self, simple_model, config):
        from training.learner import Learner

        learner = Learner(simple_model, config, loss_type="ppo")

        batch = {
            "hand": torch.randn(8, 54),
            "played_cards": torch.randn(8, 3, 54),
            "history": torch.randn(8, 15, 54),
            "actions": torch.randint(0, 309, (8,)),
            "log_probs": torch.randn(8),
            "values": torch.randn(8),
            "advantages": torch.randn(8),
            "returns": torch.randn(8),
        }

        loss_info = learner.compute_loss(batch)
        assert loss_info.total_loss.requires_grad

    def test_step(self, simple_model, config):
        from training.learner import Learner

        learner = Learner(simple_model, config, loss_type="ppo")

        batch = {
            "hand": torch.randn(8, 54),
            "played_cards": torch.randn(8, 3, 54),
            "history": torch.randn(8, 15, 54),
            "actions": torch.randint(0, 309, (8,)),
            "log_probs": torch.randn(8),
            "values": torch.randn(8),
            "advantages": torch.randn(8),
            "returns": torch.randn(8),
        }

        losses = learner.step(batch)
        assert "loss" in losses
        assert "policy_loss" in losses


class TestRolloutWorker:
    """RolloutWorker 测试"""

    def test_collect_trajectory(self):
        from training.rollout import RolloutWorker
        from env import DoudizhuEnv

        def env_fn():
            return DoudizhuEnv()

        def policy_fn(obs):
            return 0  # 总是选择 pass

        worker = RolloutWorker(env_fn, policy_fn)
        trajectory = worker.collect_trajectory()

        assert len(trajectory) > 0
        assert trajectory.transitions[-1].done

    def test_collect_steps(self):
        from training.rollout import RolloutWorker
        from env import DoudizhuEnv

        def env_fn():
            return DoudizhuEnv()

        def policy_fn(obs):
            return 0

        worker = RolloutWorker(env_fn, policy_fn)
        trajectories = worker.collect_steps(n_steps=20)

        total_steps = sum(len(t) for t in trajectories)
        assert total_steps >= 20


class TestVectorRolloutWorker:
    """VectorRolloutWorker 测试"""

    def test_collect_steps(self):
        from training.rollout import VectorRolloutWorker
        from env import DoudizhuEnv

        def env_fn():
            return DoudizhuEnv()

        def policy_fn(obs):
            return 0

        worker = VectorRolloutWorker(env_fn, policy_fn, num_envs=2)
        result = worker.collect_steps(n_steps=10)

        assert result.total_steps == 20  # 2 envs * 10 steps


class TestTrainStats:
    """TrainStats 测试"""

    def test_create(self):
        from training.trainer import TrainStats

        stats = TrainStats(
            step=100,
            loss=0.5,
            avg_reward=1.0,
        )
        assert stats.step == 100
        assert stats.loss == 0.5


class TestCallbacks:
    """回调测试"""

    def test_checkpoint_callback(self, tmp_path):
        from training.trainer import CheckpointCallback

        callback = CheckpointCallback(str(tmp_path), save_freq=1)
        assert callback.save_dir.exists()

    def test_early_stopping(self):
        from training.trainer import EarlyStoppingCallback, TrainStats

        callback = EarlyStoppingCallback(patience=3)

        class MockTrainer:
            should_stop = False

        trainer = MockTrainer()

        # 没有改进的情况
        for i in range(5):
            stats = TrainStats(avg_reward=0.0)
            callback.on_step_end(trainer, i, stats)

        assert trainer.should_stop  # 应该触发早停
