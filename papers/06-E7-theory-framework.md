# E7（GRPO+Verifier）的理论框架：GVPO / Reward-Granularity / λ-GRPO / PASS

**一句话**：E7 的 R = Result + λ_v·V 在 2026 论文里有明确谱系——它介于「过程塑形」（GVPO）
与「λ 加权混合」（Reward-Granularity）之间，且 λ=0.1 的「执行主导」设计恰好避开了
PASS 警告的混合陷阱。

## GVPO（Group Verification-based Policy Optimization，ICLR 2026）——代码域过程塑形

- 背景与我们 E7 几乎相同：代码 agent 的稀疏 outcome reward 有信用分配缺陷
- 方法：GRPO 组相对 advantage 上叠加「**过程可验证**」的 step-wise 修正——
  语法错误/运行时异常/部分单测结果作为确定性中间反馈
- shaping 函数：成功步保留 Â；失败步固定惩罚 −b 或 (1+b)·Â（对负 advantage 失败步放大惩罚）
- 结果：Qwen2.5-32B + veRL + vLLM 在 AppWorld **超过 OpenAI o1**
- **与 E7 的关系**：GVPO 的「过程信号」= 执行中间态；E7 的「过程信号」= verifier 预测 k/N——
  都是「可验证性驱动的稠密塑形」，E7 的 verifier 是它的学习化泛化

## Reward Granularity in RLVR（arXiv 2607.02869）——与我们 λ 公式同构

- **公式完全同构**：R = λ·R_process + (1−λ)·R_outcome，λ ∈ {0.9, 0.5, 0.1}
- 结果（数学域，Qwen2.5-0.5B GRPO GSM8K）：
  - process-only 63.73% > outcome-only 53.75%（近 10 点）
  - **λ=0.1（重 outcome）比纯 outcome 更差**——信号冲突
- **与 E7 的分歧**：E7 用 λ=0.1 有效（LCB +5.4）。解释候选：
  1. 域差异（数学 LLM-PRM vs 代码执行+校准 verifier）
  2. 我们的 verifier 经过 Platt 校准（ECE 0.0067），冲突信号少
  3. 我们的过程信号与 outcome 高度相关（k/N 预测的就是执行结果），冲突天然小

## λ-GRPO（ICML 2026, arXiv 2509.21154）——GRPO 隐含 PRM 的理论

- 理论：vanilla GRPO+ORM ≡ MC-PRM 感知的 RL 目标（共享前缀假设下）
- 缺陷：高奖励轨迹与低奖励轨迹共享前缀的 token 得到负 advantage（前缀频率不均）
- 修复：λ-GRPO 按过程集逆频率 1/|λ(i,t)| 重加权 → ~2× 收敛加速 + ~10% 验证精度提升
- 启示：我们的 verifier 注入如果做 token 级，应参考 λ-GRPO 的频率修正

## PASS（arXiv 2606.29296）——naive 混合的风险警告

- dense process 监督叠加到 cumulative-advantage GRPO 的三个失效模式
  （聚合期通道污染、广播期分辨率失配、轨迹期累积偏置）
- **naive 混合 outcome+process 可能掉双位数精度**
- 启示：E7 未掉分（λ 小 + 校准好 + 样本级而非 token 级注入）；未来若做 token 级
  verifier 塑形，必须先按 PASS 的三规则改造 advantage 注入点
