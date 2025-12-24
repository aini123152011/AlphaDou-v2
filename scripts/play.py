#!/usr/bin/env python3
"""
对战脚本

Usage:
    python scripts/play.py --mode watch  # 观看 AI 对战
    python scripts/play.py --mode play   # 与 AI 对战
    python scripts/play.py --model checkpoints/model.pt --mode watch
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional
import time

# 添加项目根目录到路径
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from core.cards import Card, array_to_cards
from core.actions import Action, ActionType
from core.state import Role
from env import DoudizhuEnv
from models import build_model, ModelSpec
from evaluation import RandomAgent, RuleBasedAgent, ModelAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

# 牌面显示
CARD_DISPLAY = {
    3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "10",
    11: "J", 12: "Q", 13: "K", 14: "A", 17: "2", 20: "小王", 30: "大王",
}


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaDou Play")

    parser.add_argument(
        "--mode",
        type=str,
        default="watch",
        choices=["watch", "play"],
        help="Mode: watch AI or play against AI",
    )
    parser.add_argument("--model", type=str, help="Model path")
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        choices=["random", "rule", "model"],
        help="Opponent type",
    )
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between moves")
    parser.add_argument("--games", type=int, default=1, help="Number of games")

    # 模型参数
    parser.add_argument("--backbone", type=str, default="resnet")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def cards_to_str(cards: List[int]) -> str:
    """牌列表转字符串"""
    if not cards:
        return "Pass"
    sorted_cards = sorted(cards)
    return " ".join(CARD_DISPLAY.get(c, str(c)) for c in sorted_cards)


def action_to_str(action: Action) -> str:
    """动作转字符串"""
    if action.action_type == ActionType.PASS:
        return "Pass"
    return cards_to_str(action.cards)


def print_game_state(env: DoudizhuEnv, obs: dict, info: dict):
    """打印游戏状态"""
    state = env._state

    print("\n" + "=" * 60)
    print(f"当前玩家: {info.get('current_player', 'unknown')}")
    print("-" * 60)

    # 显示各玩家手牌数
    for role in [Role.LANDLORD, Role.FARMER_DOWN, Role.FARMER_UP]:
        hand = state.hands.get(role, [])
        if role.value == info.get("current_player"):
            print(f"[{role.value}] 手牌 ({len(hand)}): {cards_to_str(hand)}")
        else:
            print(f" {role.value}  手牌数: {len(hand)}")

    # 显示上一个动作
    if state.last_action:
        print(f"\n上一个动作: {action_to_str(state.last_action)}")

    print("=" * 60)


def get_action_dim() -> int:
    """从环境获取动作空间维度"""
    from env.observation import get_action_encoder
    encoder = get_action_encoder()
    return encoder.num_actions


def load_model(path: str, args) -> torch.nn.Module:
    """加载模型"""
    action_dim = get_action_dim()
    spec = ModelSpec(
        backbone_type=args.backbone,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        action_dim=action_dim,
    )
    model = build_model(spec)

    checkpoint = torch.load(path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def create_agents(args):
    """创建智能体"""
    agents = []

    for i in range(3):
        if args.model:
            model = load_model(args.model, args)
            agent = ModelAgent(model, args.device, True, f"AI_{i}")
        elif args.opponent == "rule":
            agent = RuleBasedAgent(f"Rule_{i}")
        else:
            agent = RandomAgent(f"Random_{i}")
        agents.append(agent)

    return agents


def watch_game(args):
    """观看 AI 对战"""
    env = DoudizhuEnv()
    agents = create_agents(args)

    role_to_idx = {"landlord": 0, "farmer_down": 1, "farmer_up": 2}

    for game_idx in range(args.games):
        print(f"\n{'='*60}")
        print(f"Game {game_idx + 1}/{args.games}")
        print("=" * 60)

        obs, info = env.reset()
        done = False
        step = 0

        while not done:
            print_game_state(env, obs, info)

            current_player = info.get("current_player", "landlord")
            current_idx = role_to_idx.get(current_player, 0)
            legal_actions = env.get_legal_actions()

            # AI 选择动作
            action = agents[current_idx].act(obs, legal_actions)

            print(f"\n{agents[current_idx].name} 出牌: {action_to_str(action)}")

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            time.sleep(args.delay)

        # 游戏结束
        winner = info.get("winner", "unknown")
        print("\n" + "=" * 60)
        print(f"游戏结束! 胜者: {winner}")
        print(f"总步数: {step}")
        print("=" * 60)


def play_game(args):
    """与 AI 对战"""
    env = DoudizhuEnv()

    role_to_idx = {"landlord": 0, "farmer_down": 1, "farmer_up": 2}

    # 玩家是地主
    player_idx = 0

    # 创建 AI 对手
    if args.model:
        model = load_model(args.model, args)
        ai_agents = [
            ModelAgent(model, args.device, True, "AI_1"),
            ModelAgent(model, args.device, True, "AI_2"),
        ]
    else:
        ai_agents = [RandomAgent("AI_1"), RandomAgent("AI_2")]

    for game_idx in range(args.games):
        print(f"\n{'='*60}")
        print(f"Game {game_idx + 1}/{args.games}")
        print("你是地主!")
        print("=" * 60)

        obs, info = env.reset()
        done = False

        while not done:
            print_game_state(env, obs, info)

            current_player = info.get("current_player", "landlord")
            current_idx = role_to_idx.get(current_player, 0)
            legal_actions = env.get_legal_actions()

            if current_idx == player_idx:
                # 玩家回合
                print("\n可选动作:")
                for i, action in enumerate(legal_actions[:20]):  # 只显示前20个
                    print(f"  {i}: {action_to_str(action)}")
                if len(legal_actions) > 20:
                    print(f"  ... 还有 {len(legal_actions) - 20} 个动作")

                while True:
                    try:
                        choice = input("\n请选择动作编号 (或输入 'q' 退出): ")
                        if choice.lower() == 'q':
                            print("退出游戏")
                            return
                        idx = int(choice)
                        if 0 <= idx < len(legal_actions):
                            action = legal_actions[idx]
                            break
                        else:
                            print("无效选择，请重试")
                    except ValueError:
                        print("请输入数字")

                print(f"\n你出牌: {action_to_str(action)}")
            else:
                # AI 回合
                ai_idx = current_idx - 1 if current_idx > player_idx else current_idx
                action = ai_agents[ai_idx].act(obs, legal_actions)
                print(f"\n{ai_agents[ai_idx].name} 出牌: {action_to_str(action)}")
                time.sleep(args.delay)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # 游戏结束
        winner = info.get("winner", "unknown")
        print("\n" + "=" * 60)
        if (winner == "landlord" and player_idx == 0) or \
           (winner == "farmer" and player_idx != 0):
            print("恭喜你赢了!")
        else:
            print("你输了!")
        print(f"胜者: {winner}")
        print("=" * 60)


def main():
    args = parse_args()

    print("=" * 60)
    print("AlphaDou-v2 斗地主")
    print("=" * 60)

    if args.mode == "watch":
        watch_game(args)
    elif args.mode == "play":
        play_game(args)


if __name__ == "__main__":
    main()
