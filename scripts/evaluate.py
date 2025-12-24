#!/usr/bin/env python3
"""
评估脚本

Usage:
    python scripts/evaluate.py --model checkpoints/final_model.pt --games 100
    python scripts/evaluate.py --model1 model_a.pt --model2 model_b.pt --compare
    python scripts/evaluate.py --tournament --models model1.pt model2.pt model3.pt
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List
import json

# 添加项目根目录到路径
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch

from env import DoudizhuEnv
from models import build_model, ModelSpec
from evaluation import (
    Evaluator,
    RandomAgent,
    RuleBasedAgent,
    ModelAgent,
    Arena,
    EloSystem,
    MultiPlayerElo,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaDou Evaluation")

    # 模式
    parser.add_argument("--compare", action="store_true", help="Compare two models")
    parser.add_argument("--tournament", action="store_true", help="Run tournament")

    # 模型
    parser.add_argument("--model", type=str, help="Model path for evaluation")
    parser.add_argument("--model1", type=str, help="First model for comparison")
    parser.add_argument("--model2", type=str, help="Second model for comparison")
    parser.add_argument("--models", nargs="+", type=str, help="Models for tournament")

    # 评估参数
    parser.add_argument("--games", type=int, default=100, help="Number of games")
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        choices=["random", "rule"],
        help="Opponent type",
    )

    # 模型参数
    parser.add_argument("--backbone", type=str, default="resnet")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)

    # 其他
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic mode")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    return parser.parse_args()


def load_model(path: str, args) -> torch.nn.Module:
    """加载模型"""
    spec = ModelSpec(
        backbone_type=args.backbone,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        action_dim=309,
    )
    model = build_model(spec)

    checkpoint = torch.load(path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def evaluate_single(args):
    """评估单个模型"""
    logger.info(f"Evaluating model: {args.model}")

    # 加载模型
    model = load_model(args.model, args)
    agent = ModelAgent(
        model=model,
        device=args.device,
        deterministic=args.deterministic,
        name="model",
    )

    # 创建对手
    if args.opponent == "random":
        opponents = [RandomAgent("random1"), RandomAgent("random2")]
    else:
        opponents = [RuleBasedAgent("rule1"), RuleBasedAgent("rule2")]

    # 评估
    evaluator = Evaluator(env_fn=DoudizhuEnv, device=args.device)
    result = evaluator.evaluate(
        agent=agent,
        n_games=args.games,
        opponents=opponents,
        verbose=args.verbose,
    )

    logger.info("=" * 50)
    logger.info("Evaluation Results")
    logger.info("=" * 50)
    logger.info(f"Win Rate: {result.win_rate:.2%}")
    logger.info(f"Landlord Win Rate: {result.landlord_win_rate:.2%}")
    logger.info(f"Farmer Win Rate: {result.farmer_win_rate:.2%}")
    logger.info(f"Average Reward: {result.avg_reward:.2f}")
    logger.info(f"Average Length: {result.avg_length:.1f}")
    logger.info(f"Spring Rate: {result.spring_rate:.2%}")
    logger.info(f"Bomb Rate: {result.bomb_rate:.2f}")
    logger.info("=" * 50)

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "win_rate": result.win_rate,
                "landlord_win_rate": result.landlord_win_rate,
                "farmer_win_rate": result.farmer_win_rate,
                "avg_reward": result.avg_reward,
                "avg_length": result.avg_length,
                "spring_rate": result.spring_rate,
                "bomb_rate": result.bomb_rate,
                "games_played": result.games_played,
            }, f, indent=2)
        logger.info(f"Results saved to {args.output}")

    return result


def compare_models(args):
    """比较两个模型"""
    logger.info(f"Comparing models: {args.model1} vs {args.model2}")

    # 加载模型
    model1 = load_model(args.model1, args)
    model2 = load_model(args.model2, args)

    agent1 = ModelAgent(model1, args.device, args.deterministic, "model1")
    agent2 = ModelAgent(model2, args.device, args.deterministic, "model2")

    # 比较
    evaluator = Evaluator(env_fn=DoudizhuEnv, device=args.device)
    result = evaluator.compare(agent1, agent2, n_games=args.games)

    logger.info("=" * 50)
    logger.info("Comparison Results")
    logger.info("=" * 50)
    logger.info(f"Model 1 wins: {result['agent1_wins']} ({result['agent1_win_rate']:.2%})")
    logger.info(f"Model 2 wins: {result['agent2_wins']} ({result['agent2_win_rate']:.2%})")
    logger.info("=" * 50)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)

    return result


def run_tournament(args):
    """运行锦标赛"""
    logger.info(f"Running tournament with {len(args.models)} models")

    # 加载模型
    agents = []
    for i, path in enumerate(args.models):
        model = load_model(path, args)
        agent = ModelAgent(model, args.device, args.deterministic, f"model_{i}")
        agents.append(agent)

    # 添加基线智能体
    agents.append(RandomAgent("random"))
    agents.append(RuleBasedAgent("rule"))

    # 运行锦标赛
    arena = Arena(env_fn=DoudizhuEnv)
    result = arena.round_robin(agents, games_per_match=args.games // len(agents))

    logger.info("=" * 50)
    logger.info("Tournament Results")
    logger.info("=" * 50)

    ranking = result.get_ranking()
    for i, (name, win_rate) in enumerate(ranking):
        logger.info(f"{i+1}. {name}: {win_rate:.2%}")

    logger.info("=" * 50)

    # ELO 评分
    elo = MultiPlayerElo()
    for match in result.matches:
        elo.record_game(
            landlord=match.landlord_agent,
            farmers=match.farmer_agents,
            winner=match.winner,
        )

    logger.info("ELO Ratings:")
    for player in elo.get_ranking():
        logger.info(f"  {player.name}: {player.rating:.0f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "rankings": ranking,
                "total_games": result.total_games,
                "elo": {p.name: p.rating for p in elo.get_ranking()},
            }, f, indent=2)

    return result


def main():
    args = parse_args()

    if args.tournament and args.models:
        run_tournament(args)
    elif args.compare and args.model1 and args.model2:
        compare_models(args)
    elif args.model:
        evaluate_single(args)
    else:
        logger.error("Please specify --model, --compare with --model1/--model2, or --tournament with --models")
        sys.exit(1)


if __name__ == "__main__":
    main()
