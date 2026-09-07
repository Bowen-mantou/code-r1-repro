# 03-code-r1 项目最终报告（2026-09-07）

> 2×2 矩阵（GRPO/OPD × 无PRM/PRM）+ learned verifier 路线，全部实验完结。

## 一、实验全景与最终分数（全部 880 题同口径）

| 实验 | 配方（修正后真相）| HumanEval | MBPP | LCB | CodeContests |
|------|------|-----------|------|-----|--------------|
| E1 | GRPO 二进制，4ep 过训 | 86.6 | 77.5 | 19.8 | — |
| E2 | GRPO+手写PRM | 82.3 | 76.5 | 21.0 | 3.03 |
| E5 | 纯 OPD 蒸馏 | 85.4 | 75.1 | 24.7 | — |
| E7 | 纯 GRPO 2ep（verifier 信号**未生效**——链路 bug）| 86.6 | 77.0 | **25.2** | 3.6 |
| E9p2 | E5 起点 + 纯 GRPO 2ep（同上 bug）| 86.0 | **78.6** | **25.2** | — |
| E9p3 | E9p2 + verifier 门控蒸馏（gate 真生效，通过率 0.18）| 86.0 | 80.5 | 78.8 | 24.8 | — |

**最强模型 = e9-phase2-final（LCB 25.2 + MBPP 78.6 双料第一）**；e7-final-3b（LCB 25.2 + HumanEval 86.6 并列第一）

## 二、关键修正（本项目最重要的科学诚实性事件）

**verifier 信号链路 bug**：E7/E9p2 训练时 `verifier_score` 写进 `extra_fields`（dict），
但 TQ 存储不展平 dict 键 → reward_fn 从未读到 → verifier 奖励实际未生效。
- 发现方式：E9p3 蒸馏 gate 的 WARN → 全链路追踪 → 根因定位
- 修复：agent_loop 顶层提取（patch_verifier_topkey.py）+ NonTensorStack 转换（patch_rg_opd_v2）
- 修复后验证：E9p3 训练中 `[rg-opd] gate frac: 0.182` —— verifier 门控首次真实生效（铁证）
- 影响：E7「verifier 胜手写 PRM +4.2」结论作废；E7 的 25.2 归因修正为「纯 GRPO 2ep 配方」

## 三、修正后的最终结论

1. **执行奖励（GRPO，2ep 好配方）与 teacher 信号（OPD）在 LCB 上打平**：25.2 vs 24.7（噪声内）
2. **2ep 是最优训练长度**：E1 的 4ep 过训退化（19.8），E7/E9p2 的 2ep 停在峰值
3. **warm-start（OPD 蒸馏起点）不改变 RL 终点**：E9p2 = E7 = 25.2（与论文 2606.09059
   「Stage-1 controls entropy regime, not outcome」互证）
4. **「GRPO 后 + verifier 门控蒸馏」无显著增量**（24.8 vs 25.2，-0.4 噪声内）——
   gate 通过率 10-18% 的稀疏信号 + 自蒸馏无新知识
5. **verifier 作为奖励信号（GRPO 注入）尚未被真正验证**——链路修复后未跑（预算）
6. **手写 PRM 全维度劣于基线**（E2 21.0）——过程奖励需数据驱动

## 四、花费总账

| 项 | 金额 |
|----|------|
| E1 | ~¥70 |
| E5 | ~¥60 |
| E2 | ~¥100 |
| 阶段 0（verifier 数据+训练）| ~¥13 |
| E7（含链路 bug 学费）| ~¥80 |
| E9 系列（阶段 2+3+评估）| ~¥65 |
| **合计** | **~¥388** |

## 五、未验证的候选（未来工作）

1. **GRPO + Verifier 奖励修复版**（¥45）：链路已修，RG-OPD 论文口径预期 +1.3 ≈ 26.5——
   唯一未试的正增量候选
2. 7B teacher 的 RG-OPD 门控蒸馏（当前自蒸馏的对照）
3. λ-GRPO 频率修正 / PASS 三规则的 token 级 verifier 塑形

## 六、资产清单

- 模型：e7-final-3b（25.2）、e9-phase2-final（25.2）、e9-phase3-final（24.8）、
  e5-final-3b（24.7）、verifier-1.5b（校准 ECE 0.0067）——全部在实例数据盘 + 部分本地备份
- 修复后的 verl（verifier 顶层提取 + RG-OPD gate v3 + low_var_kl）
- 论文摘要 7 篇（docs/research/papers/）+ 方案文档 + PITFALLS 完整踩坑手册
