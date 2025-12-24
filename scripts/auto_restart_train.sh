#!/bin/bash
#
# 自动重启训练脚本 (处理意外中断)
#
# Usage:
#   nohup ./scripts/auto_restart_train.sh > train.log 2>&1 &
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 配置
MAX_RESTARTS=10
RESTART_DELAY=30
RESTART_COUNT=0

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

while [ $RESTART_COUNT -lt $MAX_RESTARTS ]; do
    log "=========================================="
    log "训练启动 (第 $((RESTART_COUNT + 1)) 次)"
    log "=========================================="

    # 运行训练 (自动从最新检查点恢复)
    ./scripts/cloud_train.sh --resume "$@"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        log "训练正常完成!"
        exit 0
    fi

    RESTART_COUNT=$((RESTART_COUNT + 1))
    log "训练异常退出 (代码: $EXIT_CODE)"
    log "将在 ${RESTART_DELAY}s 后重启 ($RESTART_COUNT/$MAX_RESTARTS)"

    sleep $RESTART_DELAY
done

log "达到最大重启次数 ($MAX_RESTARTS)，退出"
exit 1
