# AlphaDou v2 架构设计方案

> 基于 AlphaDou 项目的完整重写设计，整合 Codex 与 Gemini 的分析建议

---

## 目录

1. [项目概述](#1-项目概述)
2. [现有问题分析](#2-现有问题分析)
3. [新架构设计](#3-新架构设计)
4. [目录结构](#4-目录结构)
5. [核心组件设计](#5-核心组件设计)
6. [数据流设计](#6-数据流设计)
7. [技术选型](#7-技术选型)
8. [API 规范](#8-api-规范)
9. [迁移计划](#9-迁移计划)
10. [测试策略](#10-测试策略)

---

## 1. 项目概述

### 1.1 项目定位

AlphaDou-v2 是一个用于斗地主 (Doudizhu) 游戏的强化学习 AI 框架，基于原 AlphaDou 项目进行现代化重写。

### 1.2 核心目标

| 目标 | 描述 |
|------|------|
| **模块化** | 游戏逻辑、环境接口、模型、训练完全解耦 |
| **可扩展** | 配置驱动，支持快速实验不同网络架构 |
| **高性能** | 向量化环境、高效数据管道、混合精度训练 |
| **可维护** | 类型安全、完善测试、清晰文档 |

### 1.3 核心特性

- **完整游戏支持**: 叫牌阶段 + 出牌阶段端到端训练
- **多种网络架构**: ResNet / Transformer / LSTM 可选
- **现代训练框架**: PyTorch Lightning + Hydra 配置
- **自博弈训练**: League 系统管理历史对手池

---

## 2. 现有问题分析

### 2.1 架构层面问题

#### 2.1.1 训练管线耦合 (`douzero/dmc/dmc.py`)

```
问题位置: dmc.py:86-287

当前实现将以下职责混合在单一函数中:
├── 进程创建 (Actor spawning)
├── 线程管理 (Learner threads)
├── 数据队列 (Batch queues)
├── 模型同步 (Weight broadcasting)
├── 日志记录 (Logging)
└── 检查点保存 (Checkpointing)
```

**具体问题**:

| 问题 | 位置 | 影响 |
|------|------|------|
| 多重锁嵌套 | dmc.py:221-227 | 死锁风险、调试困难 |
| 每步同步Actor | dmc.py:181-182 | 性能瓶颈 |
| 全局可变状态 | dmc.py:18-21 | 难以测试、状态污染 |

#### 2.1.2 模型定义重复 (`models.py` vs `models_res.py`)

```python
# models.py:266-312 定义的 Model 类
class Model:
    models = {
        'first': GeneralModelBid,
        'second': GeneralModelBid,
        'third': GeneralModelBid,
        'landlord': GeneralModelResnet,
        'landlord_down': GeneralModelResnet,
        'landlord_up': GeneralModelResnet,
    }

# models_res.py:488-510 定义的另一个 Model 类
class Model:
    models = {
        'landlord': ResnetModel,
        'landlord_up': ResnetModel,
        'landlord_down': ResnetModel,
        'bidding': BidModel,  # 角色映射不一致
    }
```

**问题**:
- 两套 Model 包装类，角色映射不一致
- 切换架构需要修改多处代码
- 输入维度硬编码 (72, 54, 162 等魔法数字)

#### 2.1.3 游戏环境单体类 (`env/game.py`)

```
GameEnv 类 (~700行) 混合了:
├── 叫牌阶段逻辑 (bid_init, bid_step, bid_done)
├── 出牌阶段逻辑 (step, game_done)
├── 计分系统 (compute_player_utility, update_num_wins_scores)
├── 状态管理 (reset, get_infoset)
└── 合法动作生成 (get_legal_card_play_actions)
```

**问题**:
- 职责过多，难以单独测试
- `InfoSet` 可变且通过 deepcopy 传递，性能差
- 叫牌/出牌通过布尔标志切换，逻辑复杂

### 2.2 代码质量问题

| 类别 | 问题 | 位置 |
|------|------|------|
| 类型安全 | 无类型注解 | 全局 |
| 魔法数字 | 维度硬编码 | models.py, env_res.py |
| 逻辑泄漏 | 游戏规则嵌入 forward | models.py:81-87 |
| 特征工程 | 手工拼接，脆弱 | env_res.py:909-937 |

### 2.3 性能问题

| 问题 | 原因 | 影响 |
|------|------|------|
| 同步瓶颈 | 每个训练步同步所有 Actor 模型 | GPU 利用率低 |
| Python 循环 | 特征编码使用纯 Python | 数据吞吐受限 |
| 串行环境 | 单进程单环境 | 采样效率低 |

---

## 3. 新架构设计

### 3.1 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Scripts Layer                          │
│                  (train.py, evaluate.py)                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Training Layer                           │
│     ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│     │ Trainer  │  │ Learner  │  │  Buffer  │              │
│     │(Lightning)│  │          │  │(TorchRL) │              │
│     └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Model Layer                             │
│     ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│     │ Registry │  │ Backbone │  │  Heads   │              │
│     │          │  │(ResNet/  │  │(Policy/  │              │
│     │          │  │Transform)│  │Value/Bid)│              │
│     └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Environment Layer                         │
│     ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│     │DoudizhuEnv│ │Observation│  │ Wrappers │              │
│     │(Gymnasium)│  │ Builder  │  │(VecEnv)  │              │
│     └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Core Layer                             │
│     ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│     │  Rules   │  │  State   │  │ Actions  │              │
│     │(纯逻辑)   │  │(不可变)   │  │(枚举/生成)│              │
│     └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 核心设计原则

| 原则 | 实现方式 |
|------|---------|
| **单一职责** | 每个模块只负责一件事 |
| **依赖倒置** | 上层依赖抽象接口，不依赖具体实现 |
| **配置驱动** | 通过 YAML 配置切换行为，无需改代码 |
| **不可变状态** | GameState 使用 frozen dataclass |
| **类型安全** | 全面使用类型注解 + Pydantic 验证 |

---

## 4. 目录结构

```
AlphaDou-v2/
├── configs/                        # Hydra 配置文件
│   ├── default.yaml               # 默认配置入口
│   ├── model/
│   │   ├── resnet.yaml            # ResNet 配置
│   │   ├── transformer.yaml       # Transformer 配置
│   │   └── lstm.yaml              # LSTM 配置
│   ├── training/
│   │   ├── dmc.yaml               # DMC 训练配置
│   │   └── ppo.yaml               # PPO 训练配置 (可选)
│   └── env/
│       └── doudizhu.yaml          # 环境配置
│
├── core/                           # 纯游戏逻辑 (无 ML 依赖)
│   ├── __init__.py
│   ├── cards.py                   # 牌的定义与操作
│   ├── actions.py                 # 动作类型与生成
│   ├── rules.py                   # 规则判断 (合法性、牌型)
│   └── state.py                   # GameState 数据类
│
├── env/                            # Gymnasium 兼容环境
│   ├── __init__.py
│   ├── doudizhu_env.py            # 主环境类
│   ├── observation.py             # Observation 构建
│   ├── reward.py                  # 奖励函数
│   └── wrappers.py                # 环境包装器
│
├── models/                         # 神经网络模型
│   ├── __init__.py
│   ├── registry.py                # 模型注册与工厂
│   ├── config.py                  # ModelSpec 配置类
│   ├── backbone/
│   │   ├── __init__.py
│   │   ├── resnet.py              # ResNet 编码器
│   │   ├── transformer.py         # Transformer 编码器
│   │   └── lstm.py                # LSTM 编码器
│   ├── heads/
│   │   ├── __init__.py
│   │   ├── policy.py              # 策略头
│   │   ├── value.py               # 价值头
│   │   └── bid.py                 # 叫牌头
│   └── doudizhu_model.py          # 组合模型
│
├── training/                       # 训练组件
│   ├── __init__.py
│   ├── trainer.py                 # LightningModule
│   ├── learner.py                 # 学习器 (损失计算)
│   ├── rollout.py                 # Rollout Worker
│   ├── buffer.py                  # 经验缓冲
│   ├── self_play.py               # 自博弈管理
│   └── callbacks.py               # 训练回调
│
├── evaluation/                     # 评估模块
│   ├── __init__.py
│   ├── evaluator.py               # 评估器
│   ├── agents.py                  # 评估用 Agent
│   └── metrics.py                 # 评估指标
│
├── scripts/                        # 入口脚本
│   ├── train.py                   # 训练入口
│   ├── evaluate.py                # 评估入口
│   └── play.py                    # 人机对战
│
├── tests/                          # 测试
│   ├── unit/
│   │   ├── test_rules.py
│   │   ├── test_state.py
│   │   ├── test_actions.py
│   │   └── test_models.py
│   └── integration/
│       ├── test_env.py
│       └── test_training.py
│
├── pyproject.toml                  # 项目配置
├── requirements.txt                # 依赖
├── DESIGN.md                       # 本文档
└── README.md                       # 项目说明
```

---

## 5. 核心组件设计

### 5.1 Core Layer

#### 5.1.1 Cards (`core/cards.py`)

```python
from enum import IntEnum
from typing import Tuple, List
import numpy as np

class Card(IntEnum):
    """牌面值定义"""
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14
    TWO = 17
    BLACK_JOKER = 20
    RED_JOKER = 30

# 完整牌组
FULL_DECK: Tuple[int, ...] = (
    3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6,
    7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10,
    11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13,
    14, 14, 14, 14, 17, 17, 17, 17, 20, 30
)

def cards_to_array(cards: List[int]) -> np.ndarray:
    """将牌列表转换为 54 维 one-hot 向量"""
    array = np.zeros(54, dtype=np.float32)
    # ... 编码逻辑
    return array
```

#### 5.1.2 State (`core/state.py`)

```python
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, FrozenSet
from enum import Enum

class Phase(Enum):
    BIDDING = "bidding"
    PLAYING = "playing"
    FINISHED = "finished"

class Role(Enum):
    FIRST = "first"      # 叫牌阶段位置
    SECOND = "second"
    THIRD = "third"
    LANDLORD = "landlord"
    LANDLORD_DOWN = "landlord_down"
    LANDLORD_UP = "landlord_up"

@dataclass(frozen=True)
class GameState:
    """
    不可变游戏状态

    使用 frozen=True 保证:
    - 可哈希 (用于 MCTS)
    - 线程安全
    - 易于序列化
    """
    # 各玩家手牌 (frozen set 保证不可变)
    hands: Dict[Role, FrozenSet[int]]

    # 底牌
    three_cards: Tuple[int, ...]

    # 游戏阶段
    phase: Phase

    # 当前行动玩家
    current_player: Role

    # 地主角色 (叫牌结束后确定)
    landlord: Optional[Role]

    # 叫牌信息 [first_bid, second_bid, third_bid]
    bid_info: Tuple[int, int, int]

    # 出牌历史 ((player, action), ...)
    history: Tuple[Tuple[Role, Tuple[int, ...]], ...]

    # 已打出的炸弹数
    bombs_count: int

    # 回合数
    step_count: int

    def with_action(self, action: Tuple[int, ...]) -> 'GameState':
        """返回执行动作后的新状态 (不修改原状态)"""
        # ... 状态转移逻辑
        pass
```

#### 5.1.3 Actions (`core/actions.py`)

```python
from enum import IntEnum
from typing import List, Tuple
from dataclasses import dataclass

class ActionType(IntEnum):
    """动作类型枚举"""
    PASS = 0
    SINGLE = 1
    PAIR = 2
    TRIPLE = 3
    BOMB = 4
    ROCKET = 5
    TRIPLE_SINGLE = 6
    TRIPLE_PAIR = 7
    STRAIGHT = 8
    STRAIGHT_PAIR = 9
    AIRPLANE = 10
    AIRPLANE_SINGLE = 11
    AIRPLANE_PAIR = 12
    QUAD_SINGLE = 13
    QUAD_PAIR = 14

@dataclass(frozen=True)
class Action:
    """动作表示"""
    cards: Tuple[int, ...]
    action_type: ActionType

    @classmethod
    def pass_action(cls) -> 'Action':
        return cls(cards=(), action_type=ActionType.PASS)

class ActionGenerator:
    """合法动作生成器"""

    def __init__(self, hand_cards: List[int]):
        self.hand = sorted(hand_cards)
        self._analyze_hand()

    def _analyze_hand(self):
        """分析手牌结构"""
        # 统计各牌数量
        pass

    def generate_all(self) -> List[Action]:
        """生成所有可能动作"""
        actions = []
        actions.extend(self._gen_singles())
        actions.extend(self._gen_pairs())
        actions.extend(self._gen_triples())
        actions.extend(self._gen_bombs())
        # ... 其他牌型
        return actions

    def generate_responses(self, last_action: Action) -> List[Action]:
        """生成对上家出牌的合法响应"""
        if last_action.action_type == ActionType.PASS:
            return self.generate_all()

        responses = [Action.pass_action()]
        # 同类型更大的牌
        responses.extend(self._gen_greater(last_action))
        # 炸弹
        responses.extend(self._gen_bombs())
        return responses
```

#### 5.1.4 Rules (`core/rules.py`)

```python
from typing import List, Tuple, Optional
from .actions import Action, ActionType

class RuleEngine:
    """规则引擎 - 纯函数，无状态"""

    @staticmethod
    def detect_action_type(cards: List[int]) -> ActionType:
        """检测牌型"""
        n = len(cards)
        if n == 0:
            return ActionType.PASS
        if n == 1:
            return ActionType.SINGLE
        # ... 其他牌型检测

    @staticmethod
    def compare_actions(a: Action, b: Action) -> int:
        """比较两个动作大小: 1 if a > b, -1 if a < b, 0 if incomparable"""
        # 火箭最大
        if a.action_type == ActionType.ROCKET:
            return 1
        if b.action_type == ActionType.ROCKET:
            return -1

        # 炸弹 > 非炸弹
        if a.action_type == ActionType.BOMB and b.action_type != ActionType.BOMB:
            return 1
        # ... 其他比较逻辑

    @staticmethod
    def is_valid_play(
        action: Action,
        last_action: Optional[Action],
        hand: List[int]
    ) -> bool:
        """验证出牌合法性"""
        # 检查牌是否在手中
        # 检查是否能打过上家
        pass

    @staticmethod
    def calculate_score(
        winner: str,
        bid_count: int,
        bombs: int,
        is_spring: bool
    ) -> Dict[str, float]:
        """计算得分"""
        base = bid_count
        multiplier = 2 ** bombs
        if is_spring:
            multiplier *= 2
        # ...
```

### 5.2 Environment Layer

#### 5.2.1 DoudizhuEnv (`env/doudizhu_env.py`)

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional

from core.state import GameState, Phase, Role
from core.rules import RuleEngine
from .observation import ObservationBuilder

class DoudizhuEnv(gym.Env):
    """
    斗地主 Gymnasium 环境

    遵循标准 Gymnasium API:
    - reset() -> observation, info
    - step(action) -> observation, reward, terminated, truncated, info
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        reward_config: Optional[Dict] = None,
        render_mode: Optional[str] = None
    ):
        super().__init__()

        self.render_mode = render_mode
        self.reward_config = reward_config or {}

        # 观测空间定义
        self.observation_space = spaces.Dict({
            "hand": spaces.Box(0, 1, shape=(54,), dtype=np.float32),
            "played": spaces.Box(0, 1, shape=(3, 54), dtype=np.float32),
            "history": spaces.Box(0, 1, shape=(15, 54), dtype=np.float32),
            "legal_mask": spaces.Box(0, 1, shape=(27472,), dtype=np.float32),
            "position": spaces.Discrete(6),
            "bid_info": spaces.Box(-1, 3, shape=(3,), dtype=np.float32),
        })

        # 动作空间 (所有可能动作的索引)
        self.action_space = spaces.Discrete(27472)  # 斗地主所有合法组合数

        self._state: Optional[GameState] = None
        self._obs_builder = ObservationBuilder()
        self._rule_engine = RuleEngine()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)

        # 初始化新游戏
        self._state = self._init_game()

        obs = self._obs_builder.build(self._state)
        info = {"legal_actions": self._get_legal_actions()}

        return obs, info

    def step(
        self,
        action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        # 转换动作索引为具体动作
        concrete_action = self._decode_action(action)

        # 执行动作，获得新状态
        self._state = self._state.with_action(concrete_action)

        # 构建观测
        obs = self._obs_builder.build(self._state)

        # 计算奖励
        reward = self._compute_reward()

        # 检查终止
        terminated = self._state.phase == Phase.FINISHED
        truncated = False

        info = {
            "legal_actions": self._get_legal_actions(),
            "current_player": self._state.current_player.value,
        }

        return obs, reward, terminated, truncated, info

    def _init_game(self) -> GameState:
        """初始化游戏状态"""
        # 洗牌发牌
        pass

    def _get_legal_actions(self) -> List[int]:
        """获取当前合法动作索引"""
        pass

    def _compute_reward(self) -> float:
        """计算奖励"""
        pass
```

#### 5.2.2 Observation Builder (`env/observation.py`)

```python
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

from core.state import GameState, Role
from core.cards import cards_to_array

@dataclass
class Observation:
    """结构化观测"""
    # 自己的手牌 (54,)
    hand: np.ndarray

    # 其他玩家已出的牌 (3, 54)
    played_cards: np.ndarray

    # 最近 N 步历史 (15, 54)
    history: np.ndarray

    # 合法动作掩码
    legal_mask: np.ndarray

    # 位置编码 (6,) one-hot
    position: np.ndarray

    # 叫牌信息 (3,)
    bid_info: np.ndarray

    # 剩余牌数 (3,)
    cards_left: np.ndarray

    # 炸弹数
    bombs_count: int

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            "hand": self.hand,
            "played_cards": self.played_cards,
            "history": self.history,
            "legal_mask": self.legal_mask,
            "position": self.position,
            "bid_info": self.bid_info,
        }

    def to_tensor_dict(self) -> Dict[str, 'torch.Tensor']:
        import torch
        return {k: torch.from_numpy(v) for k, v in self.to_dict().items()}

class ObservationBuilder:
    """观测构建器"""

    def __init__(self, history_length: int = 15):
        self.history_length = history_length

    def build(self, state: GameState) -> Observation:
        """从游戏状态构建观测"""
        current = state.current_player

        # 编码手牌
        hand = cards_to_array(list(state.hands[current]))

        # 编码已出的牌
        played_cards = self._encode_played_cards(state)

        # 编码历史
        history = self._encode_history(state)

        # 构建合法动作掩码
        legal_mask = self._build_legal_mask(state)

        # 位置编码
        position = self._encode_position(current)

        # 叫牌信息
        bid_info = np.array(state.bid_info, dtype=np.float32)

        return Observation(
            hand=hand,
            played_cards=played_cards,
            history=history,
            legal_mask=legal_mask,
            position=position,
            bid_info=bid_info,
            cards_left=self._encode_cards_left(state),
            bombs_count=state.bombs_count,
        )

    def _encode_played_cards(self, state: GameState) -> np.ndarray:
        """编码三个玩家已出的牌"""
        result = np.zeros((3, 54), dtype=np.float32)
        # ... 编码逻辑
        return result

    def _encode_history(self, state: GameState) -> np.ndarray:
        """编码最近 N 步历史"""
        result = np.zeros((self.history_length, 54), dtype=np.float32)
        # 取最近 N 步
        recent = state.history[-self.history_length:]
        for i, (player, cards) in enumerate(recent):
            result[i] = cards_to_array(list(cards))
        return result
```

### 5.3 Model Layer

#### 5.3.1 Model Registry (`models/registry.py`)

```python
from typing import Dict, Type, Optional
import torch.nn as nn

from .config import ModelSpec
from .backbone import ResNetBackbone, TransformerBackbone, LSTMBackbone
from .heads import PolicyHead, ValueHead, BidHead
from .doudizhu_model import DoudizhuModel

BACKBONES: Dict[str, Type[nn.Module]] = {
    "resnet": ResNetBackbone,
    "transformer": TransformerBackbone,
    "lstm": LSTMBackbone,
}

class ModelRegistry:
    """模型注册与工厂"""

    _instance: Optional['ModelRegistry'] = None

    def __init__(self):
        self._models: Dict[str, Type[nn.Module]] = {}
        self._register_defaults()

    @classmethod
    def get_instance(cls) -> 'ModelRegistry':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _register_defaults(self):
        for name, cls in BACKBONES.items():
            self._models[name] = cls

    def register(self, name: str, model_cls: Type[nn.Module]):
        self._models[name] = model_cls

    def build_model(self, spec: ModelSpec) -> DoudizhuModel:
        """根据配置构建模型"""
        # 构建 backbone
        backbone_cls = BACKBONES[spec.backbone_type]
        backbone = backbone_cls(
            input_channels=spec.input_channels,
            hidden_dim=spec.hidden_dim,
            num_layers=spec.num_layers,
        )

        # 构建 heads
        policy_head = PolicyHead(
            input_dim=spec.hidden_dim,
            output_dim=spec.action_dim,
        )
        value_head = ValueHead(
            input_dim=spec.hidden_dim,
        )
        bid_head = BidHead(
            input_dim=spec.hidden_dim,
            output_dim=4,  # 0, 1, 2, 3 分
        )

        return DoudizhuModel(
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
            bid_head=bid_head,
        )

def build_model(spec: ModelSpec) -> DoudizhuModel:
    """便捷函数"""
    return ModelRegistry.get_instance().build_model(spec)
```

#### 5.3.2 Model Config (`models/config.py`)

```python
from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass
class ModelSpec:
    """模型规格配置"""

    # Backbone 类型
    backbone_type: Literal["resnet", "transformer", "lstm"] = "resnet"

    # 输入通道数
    input_channels: int = 40

    # 隐藏层维度
    hidden_dim: int = 512

    # Backbone 层数
    num_layers: int = 4

    # 动作空间维度
    action_dim: int = 27472

    # Transformer 专用参数
    num_heads: int = 8

    # LSTM 专用参数
    lstm_hidden: int = 256

    # Dropout
    dropout: float = 0.1

    # 是否使用注意力
    use_attention: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> 'ModelSpec':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
```

#### 5.3.3 Backbone - ResNet (`models/backbone/resnet.py`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = x.mean(dim=-1)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y.unsqueeze(-1)

class ResBlock(nn.Module):
    """残差块"""

    def __init__(self, channels: int, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.se = SEBlock(channels) if use_se else nn.Identity()
        self.act = nn.Mish(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.act(out + residual)
        return out

class ResNetBackbone(nn.Module):
    """ResNet Backbone"""

    def __init__(
        self,
        input_channels: int = 40,
        hidden_dim: int = 512,
        num_layers: int = 4,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.Mish(inplace=True),
        )

        self.layers = nn.ModuleList([
            ResBlock(hidden_dim) for _ in range(num_layers)
        ])

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            features: (batch, hidden_dim)
        """
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x).squeeze(-1)
        return x
```

#### 5.3.4 DoudizhuModel (`models/doudizhu_model.py`)

```python
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict, Any

from env.observation import Observation

@dataclass
class ModelOutput:
    """模型输出"""
    policy_logits: torch.Tensor      # (batch, action_dim)
    value: torch.Tensor              # (batch, 1)
    bid_logits: Optional[torch.Tensor] = None  # (batch, 4)

    def to_dict(self) -> Dict[str, torch.Tensor]:
        d = {"policy_logits": self.policy_logits, "value": self.value}
        if self.bid_logits is not None:
            d["bid_logits"] = self.bid_logits
        return d

class DoudizhuModel(nn.Module):
    """
    斗地主统一模型

    结构:
        Observation -> Backbone -> [PolicyHead, ValueHead, BidHead]
    """

    def __init__(
        self,
        backbone: nn.Module,
        policy_head: nn.Module,
        value_head: nn.Module,
        bid_head: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.policy_head = policy_head
        self.value_head = value_head
        self.bid_head = bid_head

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        compute_bid: bool = False,
    ) -> ModelOutput:
        """
        前向传播

        Args:
            obs: 观测字典
            compute_bid: 是否计算叫牌头
        """
        # 编码观测
        x = self._encode_observation(obs)

        # Backbone
        features = self.backbone(x)

        # Heads
        policy_logits = self.policy_head(features)
        value = self.value_head(features)

        bid_logits = None
        if compute_bid:
            bid_logits = self.bid_head(features)

        return ModelOutput(
            policy_logits=policy_logits,
            value=value,
            bid_logits=bid_logits,
        )

    def _encode_observation(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """将观测字典编码为 backbone 输入"""
        # 拼接各特征
        hand = obs["hand"]           # (batch, 54)
        played = obs["played_cards"] # (batch, 3, 54)
        history = obs["history"]     # (batch, 15, 54)

        # 展平并拼接
        batch_size = hand.shape[0]
        played_flat = played.view(batch_size, -1)  # (batch, 162)
        history_flat = history.view(batch_size, -1)  # (batch, 810)

        combined = torch.cat([hand, played_flat, history_flat], dim=-1)

        # 重塑为 (batch, channels, seq_len) 供 1D Conv
        # 这里采用简单策略，后续可优化
        return combined.unsqueeze(1).repeat(1, 40, 1)

    @torch.no_grad()
    def act(
        self,
        obs: Dict[str, torch.Tensor],
        legal_mask: torch.Tensor,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        采样动作

        Args:
            obs: 观测
            legal_mask: 合法动作掩码
            deterministic: 是否确定性选择
            temperature: 采样温度
        """
        output = self.forward(obs)
        logits = output.policy_logits

        # 应用合法动作掩码
        logits = logits.masked_fill(~legal_mask.bool(), float('-inf'))

        if deterministic:
            return logits.argmax(dim=-1)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)
```

### 5.4 Training Layer

#### 5.4.1 Trainer (`training/trainer.py`)

```python
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from dataclasses import dataclass

from models.doudizhu_model import DoudizhuModel, ModelOutput

@dataclass
class TrainConfig:
    """训练配置"""
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    batch_size: int = 256

class DoudizhuTrainer(pl.LightningModule):
    """
    斗地主训练器 (Lightning Module)

    处理:
    - 前向传播
    - 损失计算
    - 优化器配置
    - 日志记录
    """

    def __init__(
        self,
        model: DoudizhuModel,
        config: TrainConfig,
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters(ignore=['model'])

    def forward(self, obs: Dict[str, torch.Tensor]) -> ModelOutput:
        return self.model(obs)

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """训练步骤"""
        # 解包 batch
        obs = {
            "hand": batch["hand"],
            "played_cards": batch["played_cards"],
            "history": batch["history"],
        }
        actions = batch["actions"]
        returns = batch["returns"]
        advantages = batch["advantages"]
        old_log_probs = batch["log_probs"]

        # 前向
        output = self.model(obs)

        # 策略损失 (PPO style)
        log_probs = F.log_softmax(output.policy_logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        ratio = torch.exp(action_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 价值损失
        value_loss = F.mse_loss(output.value.squeeze(-1), returns)

        # 熵正则
        entropy = -(F.softmax(output.policy_logits, dim=-1) * log_probs).sum(-1).mean()

        # 总损失
        loss = (
            policy_loss
            + self.config.value_loss_coef * value_loss
            - self.config.entropy_coef * entropy
        )

        # 日志
        self.log_dict({
            "train/loss": loss,
            "train/policy_loss": policy_loss,
            "train/value_loss": value_loss,
            "train/entropy": entropy,
        }, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100000,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }
```

#### 5.4.2 Rollout Worker (`training/rollout.py`)

```python
import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import multiprocessing as mp

from env.doudizhu_env import DoudizhuEnv
from models.doudizhu_model import DoudizhuModel

@dataclass
class Trajectory:
    """单条轨迹"""
    observations: List[Dict[str, np.ndarray]] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

    def __len__(self):
        return len(self.actions)

    def to_batch(self) -> Dict[str, torch.Tensor]:
        """转换为训练 batch"""
        # 计算 returns 和 advantages
        returns = self._compute_returns()
        advantages = self._compute_advantages(returns)

        return {
            "hand": torch.stack([torch.from_numpy(o["hand"]) for o in self.observations]),
            "played_cards": torch.stack([torch.from_numpy(o["played_cards"]) for o in self.observations]),
            "history": torch.stack([torch.from_numpy(o["history"]) for o in self.observations]),
            "actions": torch.tensor(self.actions),
            "returns": torch.tensor(returns),
            "advantages": torch.tensor(advantages),
            "log_probs": torch.tensor(self.log_probs),
        }

    def _compute_returns(self, gamma: float = 0.99) -> List[float]:
        """计算折扣回报"""
        returns = []
        R = 0
        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            R = r + gamma * R * (1 - done)
            returns.insert(0, R)
        return returns

    def _compute_advantages(self, returns: List[float]) -> List[float]:
        """计算优势函数"""
        advantages = [r - v for r, v in zip(returns, self.values)]
        # 标准化
        adv_tensor = torch.tensor(advantages)
        return ((adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)).tolist()

class RolloutWorker:
    """采样工作进程"""

    def __init__(
        self,
        env_fn,
        model: DoudizhuModel,
        device: str = "cpu",
    ):
        self.env = env_fn()
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def collect(self, num_steps: int) -> Trajectory:
        """采集轨迹"""
        trajectory = Trajectory()

        obs, info = self.env.reset()

        for _ in range(num_steps):
            # 转换观测
            obs_tensor = {k: torch.from_numpy(v).unsqueeze(0).to(self.device) for k, v in obs.items()}
            legal_mask = torch.zeros(27472, dtype=torch.bool, device=self.device)
            legal_mask[info["legal_actions"]] = True

            # 采样动作
            with torch.no_grad():
                output = self.model(obs_tensor)
                action = self.model.act(obs_tensor, legal_mask.unsqueeze(0))
                log_prob = F.log_softmax(output.policy_logits, dim=-1)[0, action]
                value = output.value[0, 0]

            # 记录
            trajectory.observations.append(obs)
            trajectory.actions.append(action.item())
            trajectory.log_probs.append(log_prob.item())
            trajectory.values.append(value.item())

            # 环境步进
            obs, reward, terminated, truncated, info = self.env.step(action.item())
            trajectory.rewards.append(reward)
            trajectory.dones.append(terminated or truncated)

            if terminated or truncated:
                obs, info = self.env.reset()

        return trajectory
```

#### 5.4.3 Experience Buffer (`training/buffer.py`)

```python
from typing import Dict, List, Optional
import torch
import numpy as np
from collections import deque
import random

class ExperienceBuffer:
    """经验回放缓冲区"""

    def __init__(
        self,
        capacity: int = 100000,
        batch_size: int = 256,
    ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, trajectory: 'Trajectory'):
        """添加轨迹"""
        batch = trajectory.to_batch()
        for i in range(len(trajectory)):
            experience = {k: v[i] for k, v in batch.items()}
            self.buffer.append(experience)

    def sample(self) -> Dict[str, torch.Tensor]:
        """采样 batch"""
        experiences = random.sample(self.buffer, min(self.batch_size, len(self.buffer)))

        batch = {}
        for key in experiences[0].keys():
            batch[key] = torch.stack([e[key] for e in experiences])

        return batch

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def is_ready(self) -> bool:
        return len(self.buffer) >= self.batch_size
```

---

## 6. 数据流设计

### 6.1 训练数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Training Data Flow                           │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Actor 1    │     │   Actor 2    │     │   Actor N    │
│  (Process)   │     │  (Process)   │     │  (Process)   │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       │  Trajectory        │  Trajectory        │  Trajectory
       │                    │                    │
       └────────────────────┼────────────────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Experience Buffer  │
                 │   (Shared Memory)    │
                 └──────────┬───────────┘
                            │
                            │  Sampled Batch
                            ▼
                 ┌──────────────────────┐
                 │      Learner         │
                 │  (GPU, Main Thread)  │
                 └──────────┬───────────┘
                            │
                            │  Updated Weights
                            │  (Periodic Sync)
                            ▼
              ┌─────────────────────────────┐
              │     Parameter Server        │
              │  (Broadcast to all Actors)  │
              └─────────────────────────────┘
```

### 6.2 推理数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Inference Data Flow                           │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│  GameState   │
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌──────────────────────────────────────────────┐
│ Observation  │     │  Observation Builder                         │
│   Builder    │────▶│  - Encode hand cards                         │
└──────────────┘     │  - Encode played cards                       │
                     │  - Encode history                            │
                     │  - Build legal action mask                   │
                     └──────────────────────────────────────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────────────────────┐
                     │  Model Forward                               │
                     │  ┌─────────────────────────────────────────┐ │
                     │  │ Backbone (ResNet/Transformer)           │ │
                     │  └─────────────────────────────────────────┘ │
                     │                    │                         │
                     │        ┌───────────┼───────────┐             │
                     │        ▼           ▼           ▼             │
                     │  ┌──────────┐ ┌──────────┐ ┌──────────┐     │
                     │  │ Policy   │ │  Value   │ │   Bid    │     │
                     │  │  Head    │ │  Head    │ │  Head    │     │
                     │  └──────────┘ └──────────┘ └──────────┘     │
                     └──────────────────────────────────────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────────────────────┐
                     │  Action Selection                            │
                     │  - Apply legal mask                          │
                     │  - Sample / Argmax                           │
                     └──────────────────────────────────────────────┘
                                    │
                                    ▼
                            ┌──────────────┐
                            │   Action     │
                            └──────────────┘
```

---

## 7. 技术选型

### 7.1 核心依赖

| 组件 | 库 | 版本 | 用途 |
|------|-----|------|------|
| 深度学习 | PyTorch | ≥2.0 | 模型训练 |
| 训练框架 | PyTorch Lightning | ≥2.0 | 训练循环管理 |
| 配置管理 | Hydra + OmegaConf | ≥1.3 | 层级配置 |
| 实验追踪 | Weights & Biases | ≥0.15 | 日志可视化 |
| 环境接口 | Gymnasium | ≥0.29 | 标准化环境 |
| 数据验证 | Pydantic | ≥2.0 | 配置验证 |
| 测试 | pytest | ≥7.0 | 单元测试 |

### 7.2 可选依赖

| 组件 | 库 | 用途 |
|------|-----|------|
| 高性能缓冲 | TorchRL | 经验回放 |
| 向量化环境 | stable-baselines3 | VecEnv |
| 分布式训练 | Ray | 多机训练 |

### 7.3 Requirements

```txt
# requirements.txt
torch>=2.0.0
pytorch-lightning>=2.0.0
hydra-core>=1.3.0
omegaconf>=2.3.0
wandb>=0.15.0
gymnasium>=0.29.0
pydantic>=2.0.0
numpy>=1.24.0
pytest>=7.0.0
```

---

## 8. API 规范

### 8.1 Core API

```python
# 状态创建
state = GameState.initial()

# 状态转移 (不可变)
new_state = state.with_action(action)

# 规则检查
is_valid = RuleEngine.is_valid_play(action, last_action, hand)
action_type = RuleEngine.detect_action_type(cards)
```

### 8.2 Environment API

```python
# 遵循 Gymnasium 标准
env = DoudizhuEnv()
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

### 8.3 Model API

```python
# 构建模型
spec = ModelSpec(backbone_type="resnet", hidden_dim=512)
model = build_model(spec)

# 前向传播
output = model(obs_dict)  # -> ModelOutput

# 采样动作
action = model.act(obs_dict, legal_mask, deterministic=False)
```

### 8.4 Training API

```python
# 使用 Hydra 配置
@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig):
    model = build_model(ModelSpec(**cfg.model))
    trainer = pl.Trainer(**cfg.trainer)
    trainer.fit(model, datamodule)
```

---

## 9. 迁移计划

### Phase 1: Core Layer (Week 1-2)

```
目标: 建立纯游戏逻辑层

任务:
├── [ ] 实现 core/cards.py (牌定义与编码)
├── [ ] 实现 core/state.py (不可变 GameState)
├── [ ] 实现 core/actions.py (动作类型与生成)
├── [ ] 实现 core/rules.py (规则引擎)
└── [ ] 编写 core 模块单元测试

验收标准:
- 所有牌型检测正确
- 状态转移确定性
- 测试覆盖率 > 90%
```

### Phase 2: Environment Layer (Week 2-3)

```
目标: 实现 Gymnasium 兼容环境

任务:
├── [ ] 实现 env/doudizhu_env.py
├── [ ] 实现 env/observation.py
├── [ ] 实现 env/reward.py
├── [ ] 实现 env/wrappers.py (向量化)
└── [ ] 环境集成测试

验收标准:
- 通过 Gymnasium 环境检查器
- 支持向量化 (VecEnv)
- 性能 > 1000 steps/sec (单核)
```

### Phase 3: Model Layer (Week 3-4)

```
目标: 统一模型架构

任务:
├── [ ] 实现 models/registry.py
├── [ ] 实现 models/backbone/ (ResNet, Transformer)
├── [ ] 实现 models/heads/ (Policy, Value, Bid)
├── [ ] 实现 models/doudizhu_model.py
└── [ ] 模型单元测试

验收标准:
- 配置驱动模型构建
- 前向传播正确
- 支持 JIT 编译
```

### Phase 4: Training Layer (Week 4-5)

```
目标: 现代化训练流程

任务:
├── [ ] 实现 training/trainer.py (LightningModule)
├── [ ] 实现 training/rollout.py
├── [ ] 实现 training/buffer.py
├── [ ] 集成 Hydra 配置
├── [ ] 集成 W&B 日志
└── [ ] 训练集成测试

验收标准:
- 端到端训练可运行
- 指标可在 W&B 查看
- 支持断点续训
```

### Phase 5: Evaluation & Polish (Week 5-6)

```
目标: 完善评估与文档

任务:
├── [ ] 实现 evaluation/ 模块
├── [ ] 对比原版性能
├── [ ] 编写 README
├── [ ] 补充类型注解
└── [ ] 代码审查与重构

验收标准:
- 训练后 AI 胜率 > 基线
- 文档完整
- 代码通过 lint 检查
```

---

## 10. 测试策略

### 10.1 测试金字塔

```
                    ┌───────────────┐
                    │   E2E Tests   │  <- 少量，验证完整流程
                    │   (10%)       │
                    └───────┬───────┘
                            │
               ┌────────────┴────────────┐
               │   Integration Tests     │  <- 模块间交互
               │        (30%)            │
               └────────────┬────────────┘
                            │
          ┌─────────────────┴─────────────────┐
          │          Unit Tests               │  <- 大量，验证单个函数
          │            (60%)                  │
          └───────────────────────────────────┘
```

### 10.2 测试示例

```python
# tests/unit/test_rules.py
import pytest
from core.rules import RuleEngine
from core.actions import ActionType

class TestRuleEngine:
    def test_detect_single(self):
        assert RuleEngine.detect_action_type([3]) == ActionType.SINGLE

    def test_detect_pair(self):
        assert RuleEngine.detect_action_type([3, 3]) == ActionType.PAIR

    def test_detect_bomb(self):
        assert RuleEngine.detect_action_type([3, 3, 3, 3]) == ActionType.BOMB

    def test_detect_rocket(self):
        assert RuleEngine.detect_action_type([20, 30]) == ActionType.ROCKET

    def test_compare_bomb_beats_triple(self):
        bomb = Action(cards=(3, 3, 3, 3), action_type=ActionType.BOMB)
        triple = Action(cards=(14, 14, 14), action_type=ActionType.TRIPLE)
        assert RuleEngine.compare_actions(bomb, triple) == 1

# tests/integration/test_env.py
import gymnasium as gym
from env.doudizhu_env import DoudizhuEnv

class TestDoudizhuEnv:
    def test_reset(self):
        env = DoudizhuEnv()
        obs, info = env.reset()
        assert "hand" in obs
        assert "legal_actions" in info

    def test_step(self):
        env = DoudizhuEnv()
        obs, info = env.reset()
        action = info["legal_actions"][0]
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)
```

### 10.3 CI/CD 配置

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -e ".[dev]"
      - run: pytest tests/ --cov=alphadou_v2 --cov-report=xml
      - uses: codecov/codecov-action@v3
```

---

## 附录

### A. 配置文件示例

```yaml
# configs/default.yaml
defaults:
  - model: resnet
  - training: dmc
  - env: doudizhu

seed: 42
experiment_name: alphadou_v2_baseline
```

```yaml
# configs/model/resnet.yaml
backbone_type: resnet
input_channels: 40
hidden_dim: 512
num_layers: 4
use_attention: true
dropout: 0.1
```

```yaml
# configs/training/dmc.yaml
learning_rate: 1e-4
batch_size: 256
num_actors: 8
unroll_length: 100
total_frames: 1_000_000_000
save_interval: 1_000_000
value_loss_coef: 0.5
entropy_coef: 0.01
```

### B. 术语表

| 术语 | 解释 |
|------|------|
| DMC | Deep Monte Carlo，基于蒙特卡洛回报的训练方法 |
| Actor | 采样进程，与环境交互收集经验 |
| Learner | 学习进程，消费经验更新模型 |
| Backbone | 特征提取网络主干 |
| Head | 任务特定输出层 |
| InfoSet | 信息集，玩家可观测的游戏状态 |

---

*文档版本: 1.0*
*最后更新: 2024*
