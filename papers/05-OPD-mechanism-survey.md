# OPD 机制全景与工具链（slime/Miles/ROLL + SDPG + MOPD）

**一句话**：On-Policy Distillation 的公式本质是 `teacher_logps − student_logps` 的 token 级稠密信号，2026 年已是主流训练组件，多框架原生支持。

## 机制

- **核心公式**：token_rewards = teacher_logps − student_logps（= log(π_t/π_s) 的逐 token 比值）
  → 正 = teacher 更认可该 token → 强化；负 → 惩罚。喂进标准 policy gradient 管线
  （GRPO/PPO/REINFORCE++ 同构复用）。
- **加法形式**（SLIME/Miles/ROLL）：`Â_t = A_t − λ_opd · D_KL(P_teacher ∥ P_student)_t`
  —— token 级 reverse KL 作为减法惩罚，与任何 advantage 估计器正交
  （`--opd-kl-coef` 默认 1.0 控制权重）。这就是 veRL 的 low_var_kl 模式（E5 用的）。
- **两种模式**：
  - Pure OPD：advantage = 纯负 KL，任务奖励置 0（能力迁移/压缩）
  - RL-augmented OPD：token_rewards = external_reward − λ·reverse_kl（可验证奖励 + teacher 引导）
- **KL 方向选择**：forward KL（mode-covering，标准蒸馏）/ reverse KL（mode-seeking）；
  自适应散度方法（ToDi/AKL/EOPD）避免「幻觉区」输出。
- **teacher logits 获取**：跨架构大 teacher 走外部 SGLang 服务；同架构走 Megatron 共载。

## 变体

- **Top-k OPD**（Miles）：student 每位置记录 top-k 候选，teacher 只打分候选集（省 O(R²K) 开销）
- **Black-box OPD**（GAD）：无 teacher logits 时训判别器区分 teacher/student 响应当奖励
- **SDPG（Self-Distilled Policy Gradient, 2606.04036）**：全词表 on-policy **自蒸馏**
  （特权上下文 + verifier-grounded policy gradient + KL 锚定）——超越 RLSD/GRPO/OPCD，避免熵坍缩
- **MOPD（多教师 OPD）**：MiMo Flash v2 / Nemotron 3 Ultra 的前沿模式——N 个领域专家
  teacher（SFT+RL 各训）→ 单一 student 对自身轨迹最小化到相关 teacher 的 reverse KL
  （混合域 RL 昂贵且易冲突，故用蒸馏）

## 工程实践

- 训练系统主流形态：**Actor（rollout）/ Verifier（测试、判官、约束、奖励装配）/
  Learner（GRPO/PPO 更新）异步解耦** —— ~5× 训练提速、GPU 近 100% 利用率
- OPD 适用场景：无规则 verifier 的任务、小模型能力迁移、已有 SFT 初始化；
  **OPD 上限受 teacher 质量约束**，超越 teacher 必须靠外部 verifier 奖励
- OPD 的失败模式：teacher 打分「落不到学生能调整的地方」（只惩罚长度/模板）→ 训练停滞

## 对我们的意义

- E5 的 veRL low_var_kl = 论文公式的工程实现，我们已在用 ✅
- GLM-5 的 advantage 公式 = OPD 的 PPO/GRPO 内化形式（group=1）——阶段 3 可以直接用 veRL OPD 实现
- 「OPD 上限受 teacher 约束」→ 我们 E7 用 verifier 已突破 7B teacher 上限（25.2 > 24.7）
  ——这正是 E9「OPD 先、verifier RL 后」顺序的正确性来源
