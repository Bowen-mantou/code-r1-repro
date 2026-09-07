# veRL OPD 实现笔记（阶段 3 的工程参考）

## veRL 官方 OPD（PR #5041）结构

- 示例：`examples/on_policy_distillation_trainer/run_qwen_gsm8k.sh`
- 蒸馏 loss 配置：`distillation.distillation_loss.loss_mode` ∈ {k1, k3, forward_kl_topk}
  - k1/k3 + `use_policy_gradient=True`（policy gradient 形式，最接近 GLM-5 公式）
  - forward_kl_topk + `use_policy_gradient=False`（监督 loss 形式，需要全词表 log_softmax）
- teacher 配置：`distillation.teacher_models.<name>.model_path` + inference（vLLM）+
  `enable_resource_pool`（共卡/独立服务两种模式）
- 运行时：`AsyncTeacherLLMServerManager.compute_teacher_logprobs_single`
  （从 AgentLoopWorker._compute_teacher_logprobs 重构而来）

## 我们的差异：low_var_kl（2026-09-07 修正认知）

- 43927 的 verl 文档 `docs/algo/opd.md:281` 明确：**`low_var_kl, k3` 是官方支持的
  per-token Monte Carlo estimators of reverse KL** ——不是自加 patch！
  （PITFALLS 里「自加 low_var_kl」的记忆不准确；当时真正做的是选 loss_mode=low_var_kl
  来避开 forward_kl_topk 的全词表 log_softmax 显存开销）
- 已在 E5 训练中验证（448 步成功，LCB 24.7）
- **阶段 3 配置层即可完成**：loss_mode=low_var_kl 在 43927 的 verl 里原生支持；
  37299 的 verl 需检查（睡醒后 `grep -rn "low_var_kl" /root/autodl-tmp/verl/verl/trainer/`），
  若无则从 43927 传 verl 相关文件或整个 verl 目录

## 关键配置语义（opd.md 原文，2026-09-07 查证）

- `use_task_rewards`（**默认 true**）：
  - true：final loss = **policy_loss + distillation_loss_coef × distill_loss**
    → **这就是 E8 并行三信号，官方一键支持**（coef 默认 1.0）
  - false：PPO 项清零，只留蒸馏 loss → 纯 OPD（E5 模式）
- `use_policy_gradient`：正交开关，控制「蒸馏信号本身」的应用方式（policy gradient
  vs 监督 loss）
- 结论：E8 的技术风险比预想低（官方支持组合），但 Sequential Beats Joint 仍预测
  并行不如串行 → 推荐不变（E9 串行 + 阶段 3 纯 OPD 恢复）
- 阶段 3 配置 = E5 的 run_e5.sh（use_task_rewards=False, loss_mode=low_var_kl）
  只改：起点模型、teacher 模型、experiment_name、total_epochs=1

## E5 蒸馏配置定稿参考（已备份本地 eval_results_e5/e5_overrides.yaml）

```yaml
distillation.enabled=True
distillation.teacher_models.teacher_model.model_path=/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct
distillation.teacher_models.teacher_model.inference.name=vllm
distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.5   # 单卡尝试值；双卡定稿可能不同（E5 训练日志确认）
distillation.teacher_models.teacher_model.inference.max_model_len=4097
distillation.distillation_loss.loss_mode=low_var_kl
distillation.distillation_loss.use_task_rewards=False     # 纯 OPD
distillation.distillation_loss.use_policy_gradient=False
distillation.distillation_loss.loss_max_clamp=10.0
distillation.distillation_loss.log_prob_min_clamp=-10.0
trainer.save_freq=32 / max_actor_ckpt_to_keep=1 / total_epochs=2
```

阶段 3 改动清单：model.path → 阶段 2 产物；teacher.model_path → 前序最优模型（首选阶段 2 产物，
对照可用 7B）；total_epochs=1；experiment_name=e9-phase3

## 阶段 3 配置草案（run_e9_phase3.sh = run_e5.sh 改造）

```diff
- actor_rollout_ref.model.path=/root/autodl-tmp/models/Qwen2.5-Coder-3B-Instruct
+ actor_rollout_ref.model.path=/root/autodl-tmp/models/e9-phase2-final   # 阶段 2 产物
- distillation.teacher_models.teacher_model.model_path=/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct
+ distillation.teacher_models.teacher_model.model_path=/root/autodl-tmp/models/e9-phase2-final
+   # 或保留 7B（对照「大 teacher vs 自蒸馏」）；GLM-5 用前序阶段模型做 teacher
+ trainer.experiment_name=e9-phase3-online-distill
+ trainer.total_epochs=1   # 224 步
```

其余沿用 E5：`loss_mode=low_var_kl`、`use_task_rewards=False`、`use_policy_gradient`（E5 值）、
双卡布局（student 卡 0 / teacher 卡 1）、util 0.19-0.21。

## 检查清单（睡醒后在 37299 上）

1. `grep -r "low_var_kl" /root/autodl-tmp/verl/` ——确认 E5 的 patch 是否在 37299 的 verl 里；
   不在则从 43927 传（或重打 patch）
2. 阶段 2 完成后 convert → e9-phase2-final（复用 convert_ckpt.py 改路径）
3. 阶段 3 的 teacher 决策：先用 7B（E5 原配置）保证可复现，再用阶段 2 模型做对照（预算允许时）
