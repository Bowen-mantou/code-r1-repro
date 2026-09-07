# Code-R1 Reproduced: Multi-Signal RL for Code Reasoning (3B)

**独立复现并扩展 Code-R1 的代码推理强化学习研究**：2×2 信号对照矩阵（GRPO/OPD × 手写PRM）+
learned verifier 路线，veRL + Qwen2.5-Coder-3B，A800 双卡，~¥400 预算全程自管理。

## 核心结果（全部 880 题 LiveCodeBench 同口径）

| 模型 | 配方 | HumanEval | MBPP | LCB v5 |
|------|------|:---:|:---:|:---:|
| **e9-phase2-final** | OPD 蒸馏起点 + GRPO 2ep | 86.0 | **78.6** | **25.2** |
| e7-final-3b | 基模 + GRPO 2ep | **86.6** | 77.0 | **25.2** |
| e9-phase3-final | + verifier 门控蒸馏 | 86.0 | 78.8 | 24.8 |
| e5-final-3b | 纯 OPD 蒸馏 | 85.4 | 75.1 | 24.7 |
| e2-final-3b | GRPO + 手写 PRM | 82.3 | 76.5 | 21.0 |
| e1-final-3b | GRPO 4ep（过训）| 86.6 | 77.5 | 19.8 |

**最优模型 LiveCodeBench 25.2**（官方 7B 基线 ~24，3B 达到 7B 水平）。

## 五条核心结论

1. **执行奖励（GRPO 2ep）与 teacher 信号（OPD）在难题上打平**（25.2 vs 24.7，噪声内）
2. **2ep 是 GRPO 峰值**：4ep 过训在难题上退化最严重（LCB 25.2 → 19.8）
3. **warm-start 不改变 RL 终点**（E5 起点 = 基模起点，与 arXiv 2606.09059 互证）
4. **手写 PRM 全维度劣于数据驱动信号**（E2 21.0 垫底）
5. **RG-OPD 门控蒸馏的边界条件**：自蒸馏（teacher=student）场景失效（无新知识 + 稀疏门控）——方法论文未写明的隐含前提被实验界定

## 最有价值的技术贡献

**发现并修复贯穿两轮实验的信号链路缺陷**：
verifier 分数写入 veRL 的 `extra_fields` 嵌套字典后未展平，导致奖励信号**从未真正生效**
（影响两轮实验 ~¥120 的结论归因）。修复后 RG-OPD 门控过滤率 0.18 首次真实观测生效。

## 仓库结构

- [TUTORIAL.md](TUTORIAL.md) — 完整复现教程（数据→训练→评估→verifier→门控蒸馏）
- [REPORT.md](REPORT.md) — 最终报告（六模型全指标 + 结论修正 + 花费账）
- [PITFALLS.md](PITFALLS.md) — 踩坑手册（15+ 条实战教训）
- `papers/` — 7 篇关键论文中文摘要（GLM-5/SeqBeatsJoint/RG-OPD/SG-OPD 等）
- `code/` — 训练与评估脚本、reward 函数、verifier 服务、patch 全集
- `results/` — 六模型评估原始结果

## 方法谱系

| 环节 | 对应论文 |
|------|---------|
| GRPO + 执行奖励 | Code-R1 / DeepSeek-R1 路线 |
| OPD 蒸馏 | On-Policy Distillation（E5 = low_var_kl 纯 OPD）|
| learned verifier | 自研：1.5B 生成式 k/N 预测 + Platt 校准（ECE 0.0067）|
| RG-OPD 门控 | Reward-Gated OPD (arXiv 2607.04037) 复现 |
| 在线蒸馏恢复 | GLM-5 (arXiv 2602.15763) |
| 顺序设计 | Sequential Beats Joint (arXiv 2609.04108) |
