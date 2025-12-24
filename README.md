# AlphaDou v2

高性能斗地主 (Doudizhu) 强化学习 AI 框架

## 特性

- **完整游戏支持**: 叫牌阶段 + 出牌阶段端到端训练
- **模块化架构**: 游戏逻辑、环境接口、模型、训练完全解耦
- **多种网络架构**: ResNet / Transformer 可选
- **多种训练算法**: PPO / DMC / Self-Play
- **Gymnasium 兼容**: 标准 RL 环境接口
- **完整动作空间**: 14,636 种合法动作全覆盖

## 安装

### 依赖

- Python >= 3.10
- PyTorch >= 2.0.0
- Gymnasium >= 0.29.0

### 安装步骤

```bash
# 克隆项目
git clone https://github.com/your-repo/AlphaDou-v2.git
cd AlphaDou-v2

# 安装 (开发模式)
pip install -e .

# 安装开发依赖 (可选)
pip install -e ".[dev]"
```

## 快速开始

### 训练

```bash
# PPO 训练 (默认)
python scripts/train.py

# DMC 训练
python scripts/train.py --algorithm dmc --num-actors 5

# 自博弈训练
python scripts/train.py --algorithm self-play --steps 500000

# 自定义参数
python scripts/train.py \
    --algorithm ppo \
    --backbone resnet \
    --hidden-dim 512 \
    --num-layers 4 \
    --batch-size 256 \
    --lr 1e-4 \
    --steps 100000 \
    --device cuda
```

**训练参数:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--algorithm` | ppo | 训练算法 (ppo/dmc/self-play) |
| `--backbone` | resnet | 网络架构 (resnet/transformer) |
| `--hidden-dim` | 512 | 隐藏层维度 |
| `--num-layers` | 4 | 网络层数 |
| `--batch-size` | 256 | 批次大小 |
| `--lr` | 1e-4 | 学习率 |
| `--steps` | 100000 | 总训练步数 |
| `--n-envs` | 8 | 并行环境数 (PPO) |
| `--num-actors` | 5 | Actor 数量 (DMC) |
| `--save-dir` | checkpoints | 模型保存目录 |
| `--device` | auto | 设备 (auto/cpu/cuda) |
| `--resume` | - | 从检查点恢复 |

### 评估

```bash
# 评估模型 vs 随机对手
python scripts/evaluate.py --model checkpoints/final_model.pt --games 100

# 评估模型 vs 规则对手
python scripts/evaluate.py --model checkpoints/final_model.pt --opponent rule

# 模型对比
python scripts/evaluate.py --compare --model1 model_a.pt --model2 model_b.pt

# 锦标赛模式
python scripts/evaluate.py --tournament --models model1.pt model2.pt model3.pt
```

### 人机对战

```bash
# 观看 AI 对战
python scripts/play.py --mode watch

# 与 AI 对战 (你是地主)
python scripts/play.py --mode play

# 使用训练好的模型
python scripts/play.py --mode watch --model checkpoints/final_model.pt

# 调整出牌延迟
python scripts/play.py --mode watch --delay 0.5
```

**对战参数:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | watch | 模式 (watch/play) |
| `--model` | - | 模型路径 |
| `--opponent` | random | 对手类型 (random/rule/model) |
| `--delay` | 1.0 | 出牌延迟 (秒) |
| `--games` | 1 | 游戏局数 |

## 项目结构

```
AlphaDou-v2/
├── configs/          # 配置文件
│   ├── model/        # 模型配置
│   ├── training/     # 训练配置
│   └── env/          # 环境配置
├── core/             # 纯游戏逻辑 (无 ML 依赖)
│   ├── cards.py      # 牌定义与编码
│   ├── actions.py    # 动作类型与生成
│   ├── rules.py      # 规则引擎
│   └── state.py      # 不可变游戏状态
├── env/              # Gymnasium 环境
│   ├── doudizhu_env.py
│   ├── observation.py
│   ├── reward.py
│   └── wrappers.py
├── models/           # 神经网络模型
│   ├── backbone/     # ResNet / Transformer
│   ├── heads/        # Policy / Value / Bid
│   └── registry.py   # 模型工厂
├── training/         # 训练组件
│   ├── trainer.py
│   ├── learner.py
│   ├── buffer.py
│   └── rollout.py
├── evaluation/       # 评估模块
│   ├── evaluator.py
│   ├── agents.py
│   ├── arena.py
│   └── elo.py
├── scripts/          # 入口脚本
│   ├── train.py
│   ├── evaluate.py
│   └── play.py
└── tests/            # 测试
    ├── unit/
    └── integration/
```

## 测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行单元测试
pytest tests/unit/ -v

# 运行集成测试
pytest tests/integration/ -v

# 查看覆盖率
pytest tests/ --cov=. --cov-report=html
```

**测试统计:** 283 测试全部通过

## 性能指标

| 指标 | 数值 |
|------|------|
| 环境吞吐量 | ~20,000 steps/sec |
| 动作空间 | 14,636 |
| 动作编码覆盖率 | 100% |
| 测试覆盖 | 283 测试 |

## 云服务器训练

```bash
# 1. 上传项目到服务器
scp -r AlphaDou-v2 user@server:~/

# 2. SSH 连接并启动训练
ssh user@server
cd AlphaDou-v2

# 3. 单次训练
./scripts/cloud_train.sh --algorithm ppo --steps 1000000

# 4. 后台训练 (自动重启)
nohup ./scripts/auto_restart_train.sh > train.log 2>&1 &

# 5. 查看进度
tail -f train.log

# 6. 下载模型
scp user@server:~/AlphaDou-v2/checkpoints/final_model.pt ./
```

**环境变量配置:**

```bash
# 自定义训练参数
STEPS=5000000 BATCH_SIZE=1024 DEVICE=cuda ./scripts/cloud_train.sh
```

## 文档

详细设计方案见 [DESIGN.md](DESIGN.md)

## 依赖清单

```
torch>=2.0.0
gymnasium>=0.29.0
numpy>=1.24.0
pytorch-lightning>=2.0.0  # 可选
hydra-core>=1.3.0         # 可选
wandb>=0.15.0             # 可选
```

## 许可证

MIT
