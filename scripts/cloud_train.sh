#!/bin/bash
#
# AlphaDou-v2 云服务器训练脚本
#
# Usage:
#   ./scripts/cloud_train.sh                  # 默认 PPO 训练
#   ./scripts/cloud_train.sh --algorithm dmc  # DMC 训练
#   ./scripts/cloud_train.sh --resume         # 从检查点恢复
#

set -e

# ============================================================
# 配置区域 (根据需要修改)
# ============================================================

# 训练参数
ALGORITHM="${ALGORITHM:-ppo}"
STEPS="${STEPS:-1000000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LEARNING_RATE="${LR:-3e-4}"

# 模型参数
BACKBONE="${BACKBONE:-resnet}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
NUM_LAYERS="${NUM_LAYERS:-4}"

# PPO 参数
N_ENVS="${N_ENVS:-16}"
N_STEPS="${N_STEPS:-256}"

# DMC 参数
NUM_ACTORS="${NUM_ACTORS:-8}"
BUFFER_SIZE="${BUFFER_SIZE:-500000}"

# 保存配置
SAVE_DIR="${SAVE_DIR:-checkpoints}"
LOG_DIR="${LOG_DIR:-logs}"
SAVE_FREQ="${SAVE_FREQ:-50000}"

# 设备
DEVICE="${DEVICE:-auto}"

# ============================================================
# 辅助函数
# ============================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        log "GPU 检测:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
        return 0
    else
        log "警告: 未检测到 NVIDIA GPU"
        return 1
    fi
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        log "错误: 未找到 Python3"
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log "Python 版本: $PYTHON_VERSION"

    # 检查 PyTorch
    if python3 -c "import torch" 2>/dev/null; then
        TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
        CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")
        log "PyTorch 版本: $TORCH_VERSION, CUDA: $CUDA_AVAILABLE"
    else
        log "错误: PyTorch 未安装"
        exit 1
    fi
}

find_latest_checkpoint() {
    if [ -d "$SAVE_DIR" ]; then
        LATEST=$(ls -t "$SAVE_DIR"/*.pt 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            echo "$LATEST"
            return 0
        fi
    fi
    return 1
}

# ============================================================
# 参数解析
# ============================================================

RESUME_FLAG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --algorithm)
            ALGORITHM="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --resume)
            RESUME_FLAG="1"
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        *)
            log "未知参数: $1"
            exit 1
            ;;
    esac
done

# ============================================================
# 主流程
# ============================================================

log "=========================================="
log "AlphaDou-v2 云训练启动"
log "=========================================="

# 切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
log "项目目录: $PROJECT_ROOT"

# 环境检查
log "环境检查..."
check_python
check_gpu || DEVICE="cpu"

# 创建目录
mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_DIR"

# 检查恢复
RESUME_ARG=""
if [ -n "$RESUME_FLAG" ]; then
    CHECKPOINT=$(find_latest_checkpoint)
    if [ -n "$CHECKPOINT" ]; then
        log "从检查点恢复: $CHECKPOINT"
        RESUME_ARG="--resume $CHECKPOINT"
    else
        log "未找到检查点，从头开始训练"
    fi
fi

# 构建训练命令
if [ "$ALGORITHM" = "ppo" ]; then
    TRAIN_CMD="python3 scripts/train.py \
        --algorithm ppo \
        --backbone $BACKBONE \
        --hidden-dim $HIDDEN_DIM \
        --num-layers $NUM_LAYERS \
        --batch-size $BATCH_SIZE \
        --lr $LEARNING_RATE \
        --n-envs $N_ENVS \
        --n-steps $N_STEPS \
        --steps $STEPS \
        --save-dir $SAVE_DIR \
        --log-dir $LOG_DIR \
        --save-freq $SAVE_FREQ \
        --device $DEVICE \
        $RESUME_ARG"
elif [ "$ALGORITHM" = "dmc" ]; then
    TRAIN_CMD="python3 scripts/train.py \
        --algorithm dmc \
        --backbone $BACKBONE \
        --hidden-dim $HIDDEN_DIM \
        --num-layers $NUM_LAYERS \
        --batch-size $BATCH_SIZE \
        --lr $LEARNING_RATE \
        --num-actors $NUM_ACTORS \
        --buffer-size $BUFFER_SIZE \
        --steps $STEPS \
        --save-dir $SAVE_DIR \
        --log-dir $LOG_DIR \
        --save-freq $SAVE_FREQ \
        --device $DEVICE \
        $RESUME_ARG"
elif [ "$ALGORITHM" = "self-play" ]; then
    TRAIN_CMD="python3 scripts/train.py \
        --algorithm self-play \
        --backbone $BACKBONE \
        --hidden-dim $HIDDEN_DIM \
        --num-layers $NUM_LAYERS \
        --batch-size $BATCH_SIZE \
        --lr $LEARNING_RATE \
        --steps $STEPS \
        --save-dir $SAVE_DIR \
        --log-dir $LOG_DIR \
        --save-freq $SAVE_FREQ \
        --device $DEVICE \
        $RESUME_ARG"
else
    log "错误: 未知算法 $ALGORITHM"
    exit 1
fi

# 打印配置
log "=========================================="
log "训练配置:"
log "  算法: $ALGORITHM"
log "  总步数: $STEPS"
log "  批次大小: $BATCH_SIZE"
log "  学习率: $LEARNING_RATE"
log "  骨干网络: $BACKBONE"
log "  隐藏维度: $HIDDEN_DIM"
log "  网络层数: $NUM_LAYERS"
log "  设备: $DEVICE"
log "  保存目录: $SAVE_DIR"
log "=========================================="

# 启动训练
log "开始训练..."
log "命令: $TRAIN_CMD"
echo ""

eval $TRAIN_CMD

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    log "训练完成!"
else
    log "训练异常退出，代码: $EXIT_CODE"
fi

exit $EXIT_CODE
