#!/usr/bin/env bash
# E5: 纯 OPD 蒸馏 | Qwen2.5-Coder-3B-Instruct 全参 student + 7B-Coder teacher | 2x A800-80G
# 双卡官方池分配：student 池 1 卡（worker+rollout ~60GB）+ teacher 池 1 卡（~40GB）
# 纯 OPD：use_policy_gradient=False + use_task_rewards=False + loss_mode=low_var_kl
#   对照 E1（GRPO 二进制执行奖励）回答「代码 RL 到底需不需要执行 reward」
# 数据同 E1：n=4, batch 8q/步 → 236 步/epoch × 2ep = 472 步
# 超参同 E1（lr 1e-6），OPD 官方建议关 ref KL
set -xeuo pipefail
export VERL_USE_UV=0
# 注：expandable_segments 在双 vLLM 共卡场景会撑爆整卡（worker 逐轮膨胀），勿加；
# max_split_size_mb 减少碎片块；garbage_collection_threshold=0.6 让 allocator 在缓存利用率
# 低于 60% 时主动归还空闲显存（抑制 worker 碎片膨胀 48→58GB，PRO6000 实测）
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.6
export PYTHONPATH=/root/autodl-tmp
# vLLM EngineCore workers need cu13 runtime libs (libnvrtc.so.13) on link path
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}

MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-Coder-3B-Instruct
TEACHER_PATH=/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct
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
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.checkpoint.save_contents=[model,extra]
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=vllm
    +actor_rollout_ref.rollout.enable_sleep_mode=False
    # student 卡共 worker+rollout：rollout KV 池实测 4GB 就够，util 0.25（池 ~13.6GB 保守）
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
    trainer.experiment_name=e5-opd-gkd-3b
    trainer.n_gpus_per_node=1
    trainer.nnodes=1
    trainer.save_freq=32
    trainer.max_actor_ckpt_to_keep=1
    trainer.test_freq=16
    trainer.total_epochs=2
    # v1 trainer（E1 同款，显存实测 61GB；TQ ragged bug 已用 pad/unpad 修复）
)

# OPD teacher 资源池：单卡单副本（num_replicas 留空自动填 1）
EXTRA=(
    distillation.enabled=True
    distillation.n_gpus_per_node=1
    distillation.nnodes=1
    distillation.teacher_models.teacher_model.model_path=${TEACHER_PATH}
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1
    # teacher 独占卡：util 0.5 宽裕（80G 卡上权重 15.5 + KV 池 ~24G）
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.5
    distillation.teacher_models.teacher_model.inference.max_model_len=4097
    # low_var_kl：单样本低方差 KL 估计器（DeepSeek 同款），teacher 只返回采样 token
    # 的 logprob，无需 top-k 全词表 log_softmax——省 ~5.5GB（单卡共卡必需）。
    # 仍是纯 OPD（use_policy_gradient=False 梯度直通），实验语义不变。
    distillation.distillation_loss.loss_mode=low_var_kl
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=False
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
