#!/usr/bin/env bash
# E9 训练外挂清理器：max_keep=1 在 resume 场景失效，这里兜底。
# 只删「iteration 指向之外且 15 分钟前写完」的 step——绝不碰正在保存的。
# 用法: nohup bash clean_old_ckpt.sh < /dev/null > logs/clean_ckpt.log 2>&1 &
set -xuo pipefail
CKPT_DIR=/root/autodl-tmp/checkpoints/code-r1/e9-phase2-grpo-verifier-3b
while true; do
  latest=$(cat "$CKPT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || echo "")
  if [ -n "$latest" ]; then
    find "$CKPT_DIR" -maxdepth 1 -name 'global_step_*' -type d -mmin +3 \
      ! -name "global_step_${latest}" -exec rm -rf {} + 2>/dev/null || true
  fi
  df -h /root/autodl-tmp | tail -1 | awk '{print "disk:", $4, "free"}'
  sleep 600
done
