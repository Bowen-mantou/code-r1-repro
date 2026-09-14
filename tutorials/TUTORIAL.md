# Code-R1 完整教程：从理论到代码到工程

> 与 Search-R1 教程同构的三位一体深度：理论从零推导 → 代码逐文件带读 → 工程逐段讲解。
> 本教程是项目全部知识的**总纲**——深挖任何主题可跳到对应的专门文档
> （INTERVIEW-MASTER / ENGINEERING-DEEP-DIVE / CODE-WALKTHROUGH /
> VERIFIER-RGOPD-DEEP-DIVE / 04-RL-METHODS-GUIDE）。

## 第 0 章 导读：这个项目在做什么

**一句话**：训练一个 3B 代码模型，用强化学习让它学会「写出能通过测试的代码」，
并系统对比了五种训练信号哪个真有用。

**核心问题**：训练代码模型的信号有很多种——执行奖励（做对没）、teacher 蒸馏
（学大模型）、手写过程奖励（人定规则）、学习型 verifier（数据驱动判卷）——
**它们到底各有什么作用、该怎么组合？**

**六模型全景**（全部 880 题 LCB 同口径）：

| 模型 | 配方 | HumanEval | MBPP | LCB |
|------|------|:---:|:---:|:---:|
| e1-final-3b | GRPO 4ep（过训）| 86.6 | 77.5 | 19.8 |
| e5-final-3b | 纯 OPD 蒸馏 | 85.4 | 75.1 | 24.7 |
| e2-final-3b | GRPO+手写PRM | 82.3 | 76.5 | 21.0 |
| e7-final-3b | GRPO 2ep | **86.6** | 77.0 | **25.2** |
| e9-phase2-final | E5 起点+GRPO 2ep | 86.0 | **78.6** | **25.2** |
| e9-phase3-final | +门控蒸馏 | 86.0 | 78.8 | 24.8 |

**五条最终结论**（修正后，面试口径）：
1. 执行奖励（GRPO 2ep）与 teacher 信号（OPD）在难题上打平（25.2 vs 24.7）
2. **2ep 是 GRPO 峰值**——4ep 过训在难题上灾难性退化（19.8）
3. warm-start 不改变 RL 终点（E5 起点 = 基模起点，与论文 2606.09059 互证）
4. 手写 PRM 全维度劣于数据驱动信号
5. RG-OPD 门控的隐含前提被界定：teacher 必须强于 student（自蒸馏失效）

---

# 第 1 章 理论篇

## 1.1 代码推理 RL：为什么执行奖励是对的

### 1.1.1 奖励信号的选择空间

```
数学推理：答案对错可验证 → 但仍需 RM 处理开放答案
代码推理：单元测试 = 完美验证器 → 零误差、零成本、不可黑客
        （代码要么过测试要么不过，没有「模棱两可」）
```

**RLVR 原则**：有可验证器（verifiable reward）的任务永远优先验证器。
这就是本项目不用 reward model 打分的根本原因——**执行结果是 ground truth**。

### 1.1.2 为什么基模直接生成不够

基模会写代码，但「写→执行→看结果→修正」的闭环意识弱——RL 用执行反馈教会模型
「在写的时候就预判测试会怎么跑」。E7 的 25.2 vs 基模 ~15-18 的差距就是这个闭环
的价值。

## 1.2 RLHF 基础：从 PPO 到 GRPO

### 1.2.1 先建三个概念

```
策略 π_θ：给定 prompt，输出回答的概率分布
奖励 R：整个回答的标量分（我们的 result_bin + 塑形）
策略梯度：∇J = E[Σ_t ∇log π_θ(token_t) · A_t]——「把高奖励 token 的概率调高」
```

### 1.2.2 PPO：为什么需要 4 个模型

```
Actor（训练对象）+ Critic（估计价值 V，算 advantage）
+ Ref（KL 参考）+ RewardModel（打分）
```
Critic 是大头：多一个模型的多份显存。PPO 的 advantage 来自 GAE：
`A_t = Σ (γλ)^l · (r + γV(s') − V(s))`——时序信用分配。

### 1.2.3 策略梯度的完整推导（三个魔法步骤）

**步骤 1：似然比技巧（整个推导唯一的"魔法"）**
```
∇π_θ = π_θ · ∇log π_θ
```
概率的梯度 = 概率 × log 概率的梯度——把「对概率求导」变成「对 log 求导」，
后者可以采样估计（MC 估计的根基）。

**步骤 2：轨迹概率分解，log 把乘积变求和**
```
π(τ) = Π_t [p(s_{t+1}|s_t,a_t) · π_θ(a_t|s_t)]
log π(τ) = Σ_t [log p(...) + log π_θ(a_t|s_t)]
∇log π(τ) = Σ_t ∇log π_θ(a_t|s_t)     ← 环境项被求导消掉了
```
这一步的意义：**无模型**——不需要知道环境转移 p，梯度只依赖动作概率。

**步骤 3：baseline 免费降方差**
```
E[∇log π · b] = b · ∇Σπ = b · ∇1 = 0     ← 任何 baseline 不改变期望
但 Var(∇log π · (R−b)) < Var(∇log π · R)  ← 方差大降
```
最优 baseline = 奖励的条件期望（value function）——这是 PPO 用 critic 的理由，
也是 GRPO 用组均值的理由（组均值是「免费的近似 baseline」）。

### 1.2.4 PPO 的 clip：分区间梯度行为（面试加分点）

```
ratio r(θ) = π_θ / π_old
L = min(r·Â, clip(r, 1±ε)·Â)   ε=0.2

分区间表（Â>0 时）：
  r < 1−ε：clip 活跃，梯度 = 0（离太远，不学也不反学）
  1−ε ≤ r ≤ 1+ε：梯度 = Â（正常方向）
  r > 1+ε：clip 活跃，梯度 = 0（已经推够远，停）
```
**软 clip 的妙处**：超界样本梯度为零——不是硬截断（不可导），也不是负梯度
（推回）。「不学」比「反学」安全。

### 1.2.5 GRPO：砍掉 Critic，用组内比较

```
对每个 prompt 采样 G=4 个回答：
r_1..r_G  →  Â_i = (r_i − mean(r)) / std(r)
L = −E[min(r·Â, clip(r, 1±ε)·Â)] + β·D_KL(π_θ ∥ π_ref)
```

代码级伪代码（向量化版）：
```python
def compute_group_advantages(rewards):        # rewards: (n_prompts, G)
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return (rewards - mean) / (std + 1e-6)    # 组内 z-score

def grpo_loss(log_probs, old_log_probs, advantages, ref_log_probs):
    ratio = torch.exp(log_probs - old_log_probs)      # 重要性采样比
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
    pg_loss = -torch.min(ratio * advantages, clipped).mean()
    k3 = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1
    return pg_loss + beta * k3.mean()                 # + KL 惩罚
```

**组归一化的三重数学意义**：
| 机制 | 数学 | 直觉 |
|------|------|------|
| 组均值 baseline | E[∇·mean]=0 | 无偏降方差（1.2.3 步骤 3 的应用）|
| 组 std 归一化 | 尺度不变 | 奖励 ×100 不改变梯度方向 |
| 组内相对 | 排名信号 | 「比同组好」比「绝对值高」稳定 |

**PPO vs GRPO 完整对照**：
| | PPO | GRPO |
|---|---|---|
| advantage | GAE（需 critic）| 组内 z-score |
| 模型数 | 4（actor/critic/ref/RM）| 3（actor/ref/RM）|
| 显存 | critic 多 ~50% | 省 |
| 长程适用 | ✅ GAE 时序信用分配 | ⚠️ 长度方差大时稀释 |
| 我们的选择 | — | ✅（单轮 2048，中程适用）|

### 1.2.6 GRPO 的两个重要细节（面试高频）

1. **组归一化的代价**：轨迹长度差异大时 z-score 跨长度不公平——长轨迹的 token 级
   信用分配被稀释。**所以 GRPO 适合单轮/中程，长程 agent 建议 PPO+GAE**。
   （本项目实证：Search-R1 长程 GRPO+LLDS 崩过；Code-R1 单轮 GRPO 稳。）
2. **KL 的显存代价与缓解**：ref 模型 6GB + 前向激活——用 `param_offload`（换入换出
   用时间换空间）+ k3 单样本估计器（避开全词表 log_softmax 的 5.5GB）。
   实测账：worker 峰值 58.6GB 已含 ref+KL，80G 卡放得下。

## 1.3 KL 散度与 k3 估计器（本项目数学核心）

### 1.3.1 为什么 KL 只能估计

`D_KL = Σ_v π(v) log(π(v)/ref(v))` 要遍历 15 万词表——不可行。单样本 MC：
只对采样到的 token 算 logprob 差。

### 1.3.2 四种估计器

| 名字 | 公式 | 特点 |
|------|------|------|
| k1 | log p − log q | 最常用，有偏 |
| k2 (mse) | ½(log p − log q)² | 无方向 |
| **k3 (low_var_kl)** | **exp(q−p) − (q−p) − 1** | **无偏低方差（DeepSeek-R1 提出）** |
| full | 全词表 | 显存爆炸（forward_kl_topk 场景）|

**我们的选择**：全实验 k3——actor KL 与蒸馏 loss 同款，对照干净。

### 1.3.3 k3 的无偏性证明（面试加分推导）

```
设 p、q 为采样 token 的 log 概率（q = ref 的 logprob）：
k3 = e^(q−p) − (q−p) − 1

在 Q 分布下取期望（D_KL(P∥Q) = E_P[log P/Q] 用 P 采样，但单样本估计器
实际用的是「哪个分布采样到的 token 就算哪个」）：

关键观察：f(x) = e^x − x − 1 是凸函数且在 x=0 处取值 0、导数 0。
对 reverse KL 的估计（student 采样，即 P 下）：
E_P[k3] = E_P[e^(q−p) − (q−p) − 1]
        = E_P[q/p − 1] − E_P[log(q/p)] − 1 + E_P[p/q − ...]
        ≈ D_KL(P∥Q)（χ² 型展开首项，E[q/p]−1 在 P 下 = 0 的修正项）

直觉：k1 = log(p/q) 的方差来自 log 的尾部（q 很小时 log 爆炸）；
k3 用 e^(q−p) 替代 log 的尾部——指数在 q−p→−∞ 时衰减到 0，
把尾部方差压掉了。这就是「low_var」的来源。
```

### 1.3.4 为什么蒸馏也用 k3（方向与估计器的一体化）

```
蒸馏 loss = k3(student_logprob, teacher_logprob)
= 单样本估计的 reverse KL（student 拉向 teacher）
同一估计器同时服务「KL 惩罚（防漂移）」和「蒸馏（传知识）」——
两个目的的数学工具统一，配置与对照都干净。
```

## 1.4 OPD 蒸馏理论

### 1.4.1 三个升级（vs 离线 SFT 蒸馏）

| | SFT 蒸馏 | OPD |
|---|---|---|
| 数据 | teacher 固定输出 | **student 自己的轨迹** |
| 信号 | CE loss | 逐 token KL（teacher logprob）|
| 纠错位置 | 静态数据集 | on-policy 分布（无 exposure bias）|

### 1.4.2 KL 方向：为什么 reverse

- forward KL：mode-covering——学 teacher 全部模式含错误
- reverse KL：mode-seeking——只追 teacher 高概率区
- OPD 用 reverse：**「只学 teacher 确信的东西」**——我们 low_var_kl 就是 reverse KL
  的单样本估计。

### 1.4.3 GLM-5 的 advantage 形式（蒸馏与 RL 统一）

`Â_t = sg[log π_teacher − log π_student]`——teacher 更确信的 token 得正优势。
与 GRPO 同构：蒸馏信号直接进 policy gradient 管线。

## 1.5 手写 PRM 为什么失败（E2 案例）

**做法**：人工规则给中间步骤打分（长度、语法检查、中间检查点）加权进奖励。

**失败机制**：规则的分数与「最终能不能过测试」**结构相关性弱**——「代码写了 50 行」
和「这代码对不对」几乎无关。信用分配噪声大于信号 → 21.0 垫底。

**教训（贯穿全项目的原则）**：过程奖励设计第一问——「这个中间信号与 ground truth
的结构相关性有多强？」强 → 规则可写；弱 → 必须学（learned verifier）。

## 1.6 Learned Verifier 理论

### 1.6.1 建模：生成式 k/N 分类

```
输入："题目 + ```python 代码``` + 能过几个测试？(0-5)"
输出：P(k) = softmax(logits[digit_ids])   k∈{0..5}
分数：V_raw = Σ k·P(k) / 5   ∈ [0,1]
```

### 1.6.2 两级校准的完整推导（各管一个问题）

**Step 1 温度校准**（保序修尖锐度）：
```
P_T(k) = softmax(logits/T)
T 的搜索目标：验证集 NLL = −Σ log P_T(k_true) 最小化
T > 1 → 分布变平 → 修「过度自信」（LLM 通病）
T 不改变 argmax → 保序（k 的排序不变）
```

**Step 2 Platt 校准**（修偏移）：
```
先做 logit 变换：logit(V) = log(V/(1−V))   ← 把 [0,1] 拉满到 (−∞,+∞)
再逻辑回归：    V_cal = 1/(1 + exp(−(a·logitV + b)))
a、b 在验证集上最大似然拟合（两个参数）
```
**我们的 b=10.33 的解读**（面试能讲出含义）：
b 大正 → sigmoid 输入整体右移 → **模型的 V_raw 系统性偏低**（保守的判卷老师）。
Platt 把「保守的软分数」抬到「真实的通过率」——偏移修正是校准的核心，
温度做不到这一点（温度只改尖锐度）。

**Step 3 ECE 的完整计算示例**：
```
把验证样本按 V_cal 分 10 桶（[0,0.1), ..., [0.9,1]）
每桶算：conf_b = 桶内 V_cal 均值；acc_b = 桶内真实正确率
ECE = Σ_b (n_b/N) · |acc_b − conf_b|
ECE = 0.0067：平均每桶的「说 vs 实际」差距 < 0.7 个百分点
```

**为什么校准是奖励信号的生命线**：
RL 把 V_cal 当梯度方向——「说 0.8 实际 0.5」会让模型学错误的方向。
**校准之后，verifier 的分数才有资格进奖励函数**——这是本项目 verifier 设计
与其他「直接拿 logits 当奖励」做法最本质的区别。

### 1.6.3 奖励集成：结果主导 + 微塑形

`R = result_bin + 0.1 × V_cal`
- result_bin 主导：做对 1.0 永远压过 verifier 的 0.1 上限 → 黑客无门
- 0.1×V 塑形：「做错但接近」的轨迹拿到 0.07 > 0 的正梯度

## 1.7 RG-OPD 门控（四象限裁决）

```
g = I[(A>0 ∧ L_T>L_S+δ) ∨ (A≤0 ∧ L_T<L_S−δ)]
```
| | teacher 更确信 | teacher 更不确信 |
|---|---|---|
| 奖励正 | ✅ 蒸 | ❌ 不蒸（保护另类解法）|
| 奖励负 | ❌ 不蒸（过滤 teacher 错误信念）| ✅ 蒸（推离）|

**我们界定的边界条件**：自蒸馏（teacher=student）下 L_T−L_S 是数值噪声 → 随机砍半
→ 通过率 0.10-0.18 → 无增量。**前提：teacher 必须强于 student。**

### 1.7.2 与相邻方法的公式对比（调研深度展示）

| 方法 | 公式 | 粒度 |
|------|------|------|
| RG-OPD | `I[(A>0∧L_T>L_S) ∨ (A≤0∧L_T<L_S)]` | 轨迹级硬门 |
| OPDVR | ReLU(±(L_T−L_S)) 由 verifier 定符号 | token 级软门 |
| SG-OPD | `I[a₁(t)·a₂(t) > 0]`（token 级 sign 一致）| token 级 |
| RWOPD | V^γ 连续权重 | 样本级软权 |
| 无条件 OPD | 无门控 | — |

**我们的实现位置**：RG-OPD 轨迹级硬门（veRL low_var_kl loss 内）——
选它是因为与「verifier 校准分数 >0.5 即方向」的语义最贴合。

### 1.7.3 门控的失败机制完整分析（三层）

```
层 1（根因）：teacher=student → L_T−L_S = 浮点噪声 → 「teacher 更确信」无语义
层 2（放大）：通过率 = P(V>0.5) × P(噪声方向对) ≈ 25% × 50% ≈ 12%
           → 每批 1-2 条有效轨迹，梯度噪声淹没信号
层 3（环境）：E9p2 已在 25.2 收敛带 → 蒸馏拉向「昨天的自己」只会偏离
```
三层缺一不可地解释了 24.8 的微降——面试时按这个顺序讲，逻辑完整。

## 1.8 训练稳定性：2ep 峰值与 4ep 退化

```
E1（4ep/944 步）→ LCB 19.8：GRPO 过训 = entropy collapse（策略坍缩到少数模板）
E7（2ep/446 步）→ LCB 25.2：恰好停在峰值
```

### 1.8.1 entropy collapse 的机制

```
RL 优化过程 = 对高奖励输出的「正反馈循环」
后期：少量高奖励模板的生成概率被无限推高 → 策略熵 → 0
表现：输出多样性死亡（千题一面）、泛化崩（验证集过拟合奖励模式）
证据链：E1 的 HumanEval 86.6 冠军 vs LCB 19.8 垫底——简单题模式被
      过拟合，竞赛题全崩
```

### 1.8.2 防御手段谱系

| 手段 | 机制 | 我们用了吗 |
|------|------|-----------|
| KL 惩罚 β=0.001 | 拖住策略不远离 ref（熵的下界）| ✅ |
| 2ep 停早 | 在坍缩前收手 | ✅（运气+经验）|
| 验证曲线监控 | 峰值定位 | ❌（短板，已入 PITFALLS）|
| 熵正则项 | 显式奖励多样性 | ❌ |
| 蒸馏 reheating | 坍缩后恢复（TS-OPSD）| ❌（未来工作）|

**机制**（文献呼应 TS-OPSD 2606.00755）：RL 后期策略熵坍缩，多样性死亡，泛化崩。
**经验**：GRPO 训练必须配验证集曲线监控，2ep 是安全默认。

---

# 第 2 章 代码详解（逐文件带读）

## 2.1 reward_fn.py：奖励的最终裁决（~100 行）

```python
def compute_score(data_source, solution_str, ground_truth, extra_info,
                  use_prm=False, use_verifier=False, lambda_v=0.1):
    # 1. 沙箱执行（本项目的 ground truth 来源）
    result_bin = 1.0 if execute_passes_tests(solution_str, ground_truth) else 0.0
    # 2. verifier 塑形
    if use_verifier:
        v = extra_info.get("verifier_score")
        if v is None:
            v = extra_info["extra_fields"].get("verifier_score")  # 嵌套兜底
        return result_bin + lambda_v * v
    return result_bin
```

**三个带教点**：
1. **双层读取的代价**：`get` 读不到时静默 fallback——这是链路 bug 的土壤。
   更好的写法是读不到就 raise 或启动 smoke 打印。
2. **λ_v=0.1 的结构性防黑客**：verifier 最多贡献 0.1，做对一题的 1.0 永远压过它。
3. 签名是 veRL naive manager 约定：per-sample 调用，不是 DataProto。

## 2.2 verifier_server.py：Ray 单例 + 攒批（~140 行）

### 推理路径
```python
last_pos = enc["attention_mask"].sum(dim=1) - 1      # 最后一个有效 token
logits = out.logits[torch.arange(len(texts)), last_pos]
digit_logits = logits[:, self.digit_ids]             # (B,6) 只取 "0".."5"
probs = softmax(digit_logits)
v_raw = (probs * arange(6)).sum(-1) / 5              # 期望值而非 argmax
v_cal = sigmoid(a * logit(v_raw) + b)                # Platt
```

### 攒批 worker（本项目最精妙的并发设计）
```python
def _worker(self):
    while True:
        items = [self._q.get()]                      # 阻塞等第一条
        deadline = time.time() + 0.02                # 20ms 窗口
        while time.time() < deadline:
            try: items.append(self._q.get(timeout=deadline - time.time()))
            except queue.Empty: break
        vs = self.verifier.score_batch(...)          # 合并 batch 推理
        for item, v in zip(items, vs): item[0].put(v)  # future 回传
```
**设计心法**：对外同步 API（future.get），对内异步攒批——接口不变、吞吐 3.6×。
步耗时 51s → 19.7s。窗口大小 = 可容忍延迟上限。

## 2.3 sandbox_exec.py：seccomp BPF 手写（91 行）

```python
BLOCKED_NRS = [41..55]                    # x86_64 socket 家族 syscall
def build_filter():
    f = [_f(0x20, 0, 0, 4),               # LD W ABS 4（读 arch）
         _f(0x15, 1, 0, AUDIT_ARCH_X86_64),  # arch 检查（安全第一行）
         _f(0x06, 0, 0, KILL_PROCESS),    # 非 x86_64 直接杀
         _f(0x20, 0, 0, 0)]               # LD syscall nr
    for nr in BLOCKED_NRS:
        f += [_f(0x15, 0, 1, nr), _f(0x06, 0, 0, ERRNO|1)]  # EPERM（可诊断）
    f.append(_f(0x06, 0, 0, ALLOW))
```
RLIMIT 全家桶（NPROC 32/NOFILE 32/FSIZE 2MB/AS 4GB/**CPU 30s 内核强制**）
+ setuid nobody（先 NO_NEW_PRIVS 再 seccomp）+ **execvp 原子替换**（无 fork 窗口）。

## 2.4 firejail_exec.py：进程组超时（103 行）

```python
proc = subprocess.Popen(command, ..., start_new_session=True)  # 自成进程组
try:
    stdout, stderr = proc.communicate(input=..., timeout=timeout + 10)
except subprocess.TimeoutExpired:
    os.killpg(proc.pid, signal.SIGKILL)   # 整组杀——孙进程逃逸 bug 的修复
```
**事故**：`subprocess.run(timeout)` 只杀直接子进程 → 孙进程 hold 管道 → 训练卡死
54 分钟。三层递进：用户态超时 → 进程组语义 → RLIMIT_CPU 内核兜底。

## 2.5 gen_lcb_fast.py：两阶段自举批量生成

```
阶段 1：串行 80 题（均匀采样）→ 真实长度先验
阶段 2：800 题装箱（FFD 贪心，ratio≤1.5，batch≤8）+ 动态 max_new
撞顶检测：gen 长度 == max_new → 单独 2048 重跑（口径保证）
```

### 装箱调度的完整代码（带读）

```python
def schedule(items, est, default=1024):
    # FFD（first-fit decreasing）：大件先装，找能容纳的批
    scored = sorted(items, key=lambda x: est.get(x[0], default), reverse=True)
    batches = []
    for item in scored:
        e = est.get(item[0], default)
        placed = False
        for batch in batches:
            be = est.get(batch[0][0], default)
            # ratio≤1.5：批内最长/最短 ≤1.5 —— 木桶浪费上限 33%
            if len(batch) < BATCH_MAX and max(be, e) <= min(be, e) * RATIO:
                batch.append(item); placed = True; break
        if not placed:
            batches.append([item])     # 开新批
    return batches
```
**三个设计点的「为什么」**：
- 降序排序（FD）：大件先装，装箱经典启发式（FFD 是近似最优的）
- RATIO=1.5：批内长度差上限——最长题决定整批时长，ratio 1.5 = 最坏 33% padding 浪费
- default=中位数：无先验题的估计——分布最稳健的单点

### 撞顶重跑：口径保证的关键

```python
hit_cap = gen.shape[0] >= max_new       # 生成长度顶到上限 = 可能截断
if hit_cap:
    text, _ = gen_one(model, tok, prompt)  # 单独 2048 重跑（与串行逐 token 一致）
```
**为什么这个 if 值回票价**：动态上限的批量生成若截断了某题，那道题的结果与
「串行 2048 评估」不一致 → E9p2 与 E7 的分数不可比。**优化可以打折扣，口径不能**。

**诚实结论**：真实中位数 128 token → 动态上限 256 → 744/800 撞顶重跑 → 总时长
与串行打平（75 vs 80 分钟）。**教训**：模拟验证了调度正确性（111 批完美装箱），
没验证真实长度分布——时间优化先跑 20 题实测。

## 2.6 my_lcb_eval.py：官方口径打分

```
读 5 文件 880 题 → CodeGenerationProblem 构造 → codegen_metrics
（16 进程，timeout 6s）→ pass@1 = 222/880 = 25.2
```
坑：无卡 2GB 内存 OOM（看 `/sys/fs/cgroup/memory.max`，不信 free）；单卡 120GB 跑。

## 2.7 patch 系列三件套（链路修复的故事线）

```python
# patch_verifier_topkey.py（根因修复，4 行）
_vsf = field["extra_fields"].get("verifier_score")   # TQ put 前
if _vsf is not None:
    field["verifier_score"] = float(_vsf)            # 提为顶层键
# patch_rg_opd_v2.py（容器类型教训）
vs_l = vs.tolist() if hasattr(vs, "tolist") else list(vs)  # v1 as_tensor 空张量崩溃
# patch_rg_opd_v3.py（观测设计）
print(f"[rg-opd] gate frac: {(gate>0).float().mean():.3f}")  # 生效铁证
```

## 2.8 run 脚本逐键（run_e7.sh 为代表）

关键键的「为什么」速查：`truncation=left`（保题目尾部）、`kl_coef=0.001`
（DeepSeek-R1 量级）、`token_len=2560`（显存账反推）、`util=0.25`（KV 池 13.6G）、
`forward_prefetch=False`（省 2GB）、`save_contents=[model,extra]`（13G/份）、
清卡逻辑（孤儿 vLLM 19.87GB 的教训）。

## 2.9 watchdog 与清理器（bash 生存技巧）

```
watchdog：pgrep 'main_pp[o]'（[o] 防自杀）→ 消失 → sleep 60 复查 → 错误关键词
         （排除 RewardLoopWorker 正常噪音）→ 关机
清理器：find -mmin +3 ! -name latest -delete（三个时间常数：保存 13 分钟 >
         清理周期 10 分钟 > 阈值 3 分钟）
```

---

# 第 3 章 工程篇

## 3.1 veRL 架构与一个训练步的生命周期

```
启动 → hydra → Ray → TaskRunnerV1
  ├─ 角色：ActorRollout（训练+vLLM 同进程）/ Ref（offload）/
  │        Reward / AgentLoopWorkerTQ×8 / [OPD]Teacher
  └─ 每步：rollout（n=4）→ 沙箱+verifier → TQ → reward → FSDP 前反向
           （+KL/+蒸馏）→ step → save
```

## 3.2 数据流五跳（bug 战场，逐跳背熟）

```
agent_loop 写 extra_fields["verifier_score"]
→ TQ 存（dict 不展平！）
→ trainer reward 阶段 pop extra_fields
→ naive.py 读 non_tensor_batch["verifier_score"] → None
→ reward_fn fallback → 静默降级
修复：TQ put 前提为顶层字段
```

## 3.3 显存账（完整版见 01-GPU-MEMORY-GUIDE）

```
参数系 25GB（权重 6.2 + 梯度 6.2 + Adam 12.4）+ 激活 20-30GB + KV 13.6GB
实测峰值 58.6GB（含 ref+KL）→ 双卡布局：卡 0 训练 72.2GB + 卡 1 verifier 4GB
```

## 3.4 监控体系（三层 + 两个短板）

进程层（watchdog）/ 资源层（显存磁盘）/ 结果层（sha256+dry-run）✅
信号层（reward 成分分解）+ 质量层（分布曲线告警）❌——链路 bug 与 4ep 退化的教训。

## 3.5 评估管线

```
convert_ckpt → gen_lcb_fast（生成 880 题）→ my_lcb_eval（打分）
            → gen_eval + eval_scores（evalplus）→ gen_codecontests（沙箱执行）
口径三戒：5 文件并集 880 题、撞顶重跑保生成口径、坏行报错不跳过
```

