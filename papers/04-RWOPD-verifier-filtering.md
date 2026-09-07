# RWOPD（arXiv 2605.13501）与 verifier 过滤蒸馏数据实践

**一句话**：verifier 的正确用法是「**为稠密 teacher 监督做过滤/加权**」，而不是单独当稀疏奖励。

## RWOPD：Reward-Weighted On-Policy Distillation（NL-to-SVA 代码生成）

- 属性等价检查器（open PEC，对 Cadence JasperGold 零误报）对 student rollout 打分
  → **reward-weighted** 后，同任务 teacher 提供稠密 forward-KL 梯度
- **关键发现**：7B+LoRA 规模上，verifier 作为「reward-shaped filter + 稠密 teacher 监督」
  **显著优于**把它当独立稀疏奖励（9 个稀疏 verifier-reward/偏好优化基线全部 ≤ SFT 基线）

## Verifier 过滤蒸馏数据的其他实践

- **MinervaRL（2602.00513）**：任务 verifier 打分生成轨迹 → 轻量 TextCNN 分类器二次过滤
  verifier-correct 的 ACR 轨迹 → 蒸馏。~880K 生成尝试 → ~56K 接受轨迹。
- **AutoGEO**：通用 pass-rate 过滤流程——候选集推理 → verifier 判 pass → 只保留成功补全
  → 高质量 SFT 数据集，喂给 cold-start SFT 和 GRPO RL 阶段。
- **OpenThoughts（2505.00551）数据配方消融**（反直觉结论）：
  - **质量 > 数量 > 多样性**（混合 2 个最好来源 > 混合 16 个来源 +5%）
  - LLM 驱动过滤 > fastText 分类器（+4-6%）
  - 「更强的 teacher 一定更好」「答案验证总是必要」在受控消融下**都不成立**
- **通用过滤管线**（便宜→贵）：规则（长度/重复）→ 困惑度阈值（过低=背诵、过高=乱码）
  → reward model 分 → 分类器。风险：过滤器泄露自身偏好（如长度偏好）、过度过滤伤多样性。

## 对我们「阶段 1 蒸馏数据过滤」的启示

- 用户的阶段 1 方案（verifier 过滤 teacher 轨迹做纯 SFT 蒸馏）有充分先例 ✅
- 但 Sequential Beats Joint 说 **OPD 冷启动 > SFT 冷启动**；E5 模型已找回 →
  阶段 1 直接 OPD（免费）优于「过滤 SFT」
- 过滤阈值有讲究：保留「中等偏难」而非全对样本（VibeThinker：middle-to-high learning potential）
