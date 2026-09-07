#!/usr/bin/env bash
# E9 阶段 2 评估 watchdog：生成完成 → 打分 → 状态落盘 → 自动关机
set -xuo pipefail
cd /root/autodl-tmp
while true; do
  if ! pgrep -f "gen_lcb_fas[t]" > /dev/null; then
    if grep -q "lcb done" logs/gen_lcb_e9p3.log 2>/dev/null; then
      echo "GEN-DONE $(date '+%F %T')" > e9p3_eval_status.txt
      python3 my_lcb_eval.py >> logs/gen_lcb_e9p3.log 2>&1
      echo "SCORE-DONE $(date '+%F %T')" >> e9p3_eval_status.txt
      tail -5 logs/gen_lcb_e9p3.log >> e9p3_eval_status.txt
      sleep 60
      /usr/bin/shutdown
    else
      echo "GEN-CRASH $(date '+%F %T')" > e9p3_eval_status.txt
      tail -30 logs/gen_lcb_e9p3.log >> e9p3_eval_status.txt 2>/dev/null
    fi
    exit 0
  fi
  sleep 30
done
