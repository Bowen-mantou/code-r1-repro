#!/usr/bin/env bash
# E7: GRPO + learned Verifier | Qwen2.5-Coder-3B-Instruct 全参 | 1x A800-80G
# R = Result_bin(0/1) + lambda_v * V_cal（结果主导 + verifier 置信度辅助）
#   回答「learned verifier 能否改进 GRPO」——对比 E1（86.6/77.5/19.8）与 E2（82.3/76.5/21.0）
# 数据/超参/架构全同 E1：n=4, batch 8q/步, 944 步 4ep
# A800 单卡显存：worker 44 + rollout 15 + verifier 4 = 63GB/80 ✅
set -xeuo pipefail
export VERL_USE_UV=0
# 碎片抑制（PRO6000 实测：worker 48→42GB；双 vLLM 共卡场景勿加 expandable_segments）
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.6
export PYTHONPATH=/root/autodl-tmp
# vLLM EngineCore workers need cu13 runtime libs (libnvrtc.so.13) on link path
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}

MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-Coder-3B-Instruct
TRAIN_FILE=/root/autodl-tmp/data/code-r1-lc2k/train.parquet
TEST_FILE=/root/autodl-tmp/data/code-r1-lc2k/val_small.parquet

DATA=(
    algorithm.adv_estimator=grpo
    data.train_files=${TRAIN_FILE}
    data.val_files=${TEST_FILE}
    data.train_batch_size=8
    data.val_batch_size=8
    data.max_prompt_length=2048
    data.max_response_length=2048
    data.filter_overlong_prompts=True
    data.truncation=left
    algorithm.use_kl_in_reward=False
    reward.custom_reward_function.path=/root/autodl-tmp/reward_fn.py
    reward.custom_reward_function.name=compute_score
    # E7 核心：verifier 模式（结果主导 + λ·V）；λ 校准规则见实现文档 5.1
    # verifier 路径硬编码在 agent_loop patch（勿加 distillation 键，会干扰配置树）
    +reward.custom_reward_function.reward_kwargs.use_verifier=True
    +reward.custom_reward_function.reward_kwargs.lambda_v=0.1
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    # worker 峰值 58.3GB 压线：关 forward prefetch 省 ~2GB 激活
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=False
    # E2 实测 worker 峰值 57.57GB@4096；2560 压激活 ~8GB（2402 长样本 < 2560 安全）
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2560
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.checkpoint.save_contents=[model,extra]
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=vllm
    +actor_rollout_ref.rollout.enable_sleep_mode=False
    # 双卡：student 独占卡 0（worker 58.6 + rollout KV 池 13.6GB 共 72.2/79.25 ✅）
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192
    actor_rollout_ref.rollout.n=4
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.project_name=code-r1
    trainer.experiment_name=e7-grpo-verifier-3b
    trainer.n_gpus_per_node=1
    trainer.nnodes=1
    trainer.save_freq=32
    trainer.max_actor_ckpt_to_keep=1
    trainer.test_freq=16
    # 2ep 先行验证 verifier 信号（472 步 ~3.7h ¥18）；有戏再补训到 4ep 对齐 E1
    trainer.total_epochs=2
)

# 启动前强制清卡（前次崩溃的孤儿 vLLM 引擎 19.87GB 曾导致提前 OOM——审查报告）
pkill -9 -f 'VLLM::Worke[r]' 2>/dev/null || true
pkill -9 -f 'main_pp[o]' 2>/dev/null || true
sleep 5
CLEAN_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
echo "启动前显存: ${CLEAN_MEM} MiB（应 <2000）"

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "$@"
