# SG-OPD（arXiv 2606.09304）与 OPDVR（LeapLabTHU）——verifier 门控蒸馏

**一句话**：这两篇正是我们 experiment-plan 里 E6'「软门控」的对标工作——verifier 不做直接优化目标，而是**蒸馏信号的信任开关**。

## SG-OPD：Sign-Gated On-Policy Distillation（2026-06-08）

**识别 OPD 的两个失效假设**：
1. 冷启动期弱学生生成低质量 rollout → teacher 逐 token 监督无信息量
2. teacher 的 token 级偏好可能与「验证正确的方向」冲突（惩罚风格不同但正确的 token）→ 压制有效探索

**方法（verifier 作为 teacher 的信任信号，两级粒度）**：
- a₁(t)：样本级 advantage（答案正确性，组内归一）
- a₂(t)：token 级 reverse-KL teacher advantage
- **Sign-Consistency Gating**：g_t = I[a₁·a₂ > 0]——teacher 与 verifier 共识的 token 上**外推**蒸馏更新；
  冲突 token 上**软化**（不盲从 teacher）
- **Phased Teacher Sampling（PTS）**：冷启动期混入 verifier 认可的 teacher rollout
  （辅助 CE loss），三阶段余弦衰减到 0

**结果**：竞赛数学（AIME24/25、AMC、MATH500）平均 +1.98（per-sample）/ +7.50（per-question pass@32）；
符号门控让外推系数可安全到 λ=1.8（无门控 1.25 就崩）。

## OPDVR：On-policy Distillation with Verifiable Reward（清华 LeapLab）

**更简单的门控**：teacher-student log-ratio token reward 过一个**由 verifier 决定符号的 ReLU 门**：
- 正确轨迹 → 非负奖励（保留 teacher 信号）
- 错误轨迹 → 非正（翻转方向）
- **verifier 定方向，teacher 定幅度**
- GRPD 变体：把二值信号换成 GRPO 组相对 advantage

## 与我们 E6'/E7 的关系

- 我们的 E7 = verifier 注入 reward（λ_v 加权）；OPDVR = verifier 门控 OPD——两者是
  「verifier 的两种用法」的论文级对照 ✅（experiment-plan E6' 的设计与 OPDVR 撞车，
  说明方向正确；差异化点：我们可以在 GRPO 基底 + 软门控（非二值）上做）
- SG-OPD 的 sign-consistency 是「软门控」的正统实现，E6' 若做可以复现它的 gate 公式
