#!/usr/bin/env bash
# E9 阶段 2：GRPO + learned Verifier | 起点 = e5-final-3b（OPD 蒸馏产物）
# R = Result_bin(0/1) + lambda_v * V_cal（结果主导 + verifier 置信度辅助）
# 对照：vs E7（同配方、起点 = Qwen 基模）→ warm-start 增量
# 其余配置全同 E7（run_e7.sh 定稿，446 步 2ep 双卡：student 卡0 + verifier 卡1）
set -xeuo pipefail
export VERL_USE_UV=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.6
export PYTHONPATH=/root/autodl-tmp
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}

MODEL_PATH=/root/autodl-tmp/models/e5-final-3b
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
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=False
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2560
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.checkpoint.save_contents=[model,extra]
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=vllm
    +actor_rollout_ref.rollout.enable_sleep_mode=False
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
    trainer.experiment_name=e9-phase2-grpo-verifier-3b
    trainer.n_gpus_per_node=1
    trainer.nnodes=1
    trainer.save_freq=32
    trainer.max_actor_ckpt_to_keep=1
    trainer.test_freq=16
    trainer.total_epochs=2
)

# 启动前强制清卡（E7 教训：孤儿 vLLM 引擎占 19.87GB 曾导致 OOM）
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
