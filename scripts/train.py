#!/usr/bin/env python3
"""
训练脚本

Usage:
    python scripts/train.py --config configs/training/ppo.yaml
    python scripts/train.py --algorithm ppo --steps 100000
    python scripts/train.py --algorithm dmc --num-actors 5
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
import json

# 添加项目根目录到路径
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch

from env import DoudizhuEnv
from models import build_model, ModelSpec
from training import (
    TrainConfig,
    RolloutConfig,
    Trainer,
    DMCTrainer,
    SelfPlayTrainer,
    TensorBoardCallback,
    WandbCallback,
    CheckpointCallback,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaDou Training")

    # 算法选择
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "dmc", "self-play"],
        help="Training algorithm",
    )

    # 训练参数
    parser.add_argument("--steps", type=int, default=100000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")

    # PPO 参数
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--epochs", type=int, default=4, help="PPO epochs per update")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel envs")
    parser.add_argument("--n-steps", type=int, default=128, help="Steps per rollout")

    # DMC 参数
    parser.add_argument("--num-actors", type=int, default=5, help="Number of actors (DMC)")
    parser.add_argument("--buffer-size", type=int, default=100000, help="Replay buffer size")

    # 模型参数
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet",
        choices=["resnet", "transformer"],
        help="Backbone type",
    )
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers")

    # 保存和日志
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Save directory")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--save-freq", type=int, default=10000, help="Save frequency")
    parser.add_argument("--log-interval", type=int, default=10, help="Log interval")

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="alphadou-v2", help="W&B project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name")

    # 其他
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    return parser.parse_args()


def setup_seed(seed: Optional[int]):
    """设置随机种子"""
    if seed is not None:
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def get_action_dim() -> int:
    """从环境获取动作空间维度"""
    from env.observation import get_action_encoder
    encoder = get_action_encoder()
    return encoder.num_actions


def create_model(args) -> torch.nn.Module:
    """创建模型"""
    action_dim = get_action_dim()
    spec = ModelSpec(
        backbone_type=args.backbone,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        action_dim=action_dim,
    )
    return build_model(spec)


def train_ppo(args):
    """PPO 训练"""
    logger.info("Starting PPO training...")

    # 创建模型
    model = create_model(args)

    # 训练配置
    config = TrainConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
    )

    rollout_config = RolloutConfig(
        num_workers=args.n_envs,
        steps_per_worker=args.n_steps,
    )

    # 回调
    callbacks = []

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    callbacks.append(CheckpointCallback(str(save_dir), args.save_freq))

    try:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(TensorBoardCallback(str(log_dir)))
    except ImportError:
        logger.warning("TensorBoard not available, skipping logging")

    # W&B 日志
    if args.wandb:
        try:
            config_dict = {
                "algorithm": args.algorithm,
                "backbone": args.backbone,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "steps": args.steps,
            }
            callbacks.append(WandbCallback(
                project=args.wandb_project,
                name=args.wandb_name,
                config=config_dict,
            ))
            logger.info(f"W&B logging enabled: {args.wandb_project}")
        except ImportError:
            logger.warning("wandb not available. Run: pip install wandb")

    # 创建训练器
    trainer = Trainer(
        model=model,
        env_fn=DoudizhuEnv,
        config=config,
        rollout_config=rollout_config,
        callbacks=callbacks,
        device=args.device,
    )

    # 加载检查点
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        trainer.load(args.resume)

    # 训练
    stats = trainer.train(
        total_steps=args.steps,
        log_interval=args.log_interval,
    )

    # 保存最终模型
    final_path = save_dir / "final_model.pt"
    trainer.save(str(final_path))
    logger.info(f"Final model saved to {final_path}")

    return stats


def train_dmc(args):
    """DMC 训练"""
    logger.info("Starting DMC training...")

    # 创建模型
    model = create_model(args)

    # 训练配置
    config = TrainConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )

    # 创建训练器
    trainer = DMCTrainer(
        model=model,
        env_fn=DoudizhuEnv,
        config=config,
        buffer_size=args.buffer_size,
        device=args.device,
    )

    # 训练
    stats = trainer.train(
        total_episodes=args.steps,
        train_freq=4,
    )

    # 保存
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "dmc_model.pt")
    logger.info(f"Model saved to {save_dir / 'dmc_model.pt'}")

    return stats


def train_self_play(args):
    """自博弈训练"""
    logger.info("Starting Self-Play training...")

    # 创建模型
    model = create_model(args)

    # 训练配置
    config = TrainConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
    )

    # 创建训练器
    trainer = SelfPlayTrainer(
        model=model,
        env_fn=DoudizhuEnv,
        config=config,
        device=args.device,
    )

    # 训练
    stats = trainer.train(total_steps=args.steps)

    # 保存
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "self_play_model.pt")
    logger.info(f"Model saved to {save_dir / 'self_play_model.pt'}")

    return stats


def main():
    args = parse_args()

    logger.info("=" * 50)
    logger.info("AlphaDou-v2 Training")
    logger.info("=" * 50)
    logger.info(f"Algorithm: {args.algorithm}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Steps: {args.steps}")
    logger.info("=" * 50)

    # 设置随机种子
    setup_seed(args.seed)

    # 选择训练算法
    if args.algorithm == "ppo":
        stats = train_ppo(args)
    elif args.algorithm == "dmc":
        stats = train_dmc(args)
    elif args.algorithm == "self-play":
        stats = train_self_play(args)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    logger.info("=" * 50)
    logger.info("Training completed!")
    logger.info(f"Final stats: {stats}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
