# LLM 强化学习方法指南：从策略梯度到门控蒸馏（04）

> 本项目方法谱系的完整推导版：策略梯度 → GRPO → KL 估计器 → OPD 蒸馏 →
> 概率校准 → RG-OPD 门控。每条公式都能白板推导 + 说出「我们的实验验证了什么」。

---

## 第 1 章 策略梯度：全文的地基

### 1.1 目标与似然比技巧

```
目标：J(θ) = E_{τ~πθ}[R(τ)]  （τ=轨迹，R=总奖励）
∇J = E[Σ_t ∇log πθ(a_t|s_t) · R(τ)]
似然比：∇π = π·∇log π  —— 把「对概率求导」变成「对 log 概率求导」
```

### 1.2 三个关键化简（面试能讲清每一个）

1. **轨迹分解**：π(τ)=Π p(s_{t+1}|s_t,a_t)π(a_t|s_t)，log 把乘积变求和 → 逐 token 梯度
2. **环境项为零**：Σ∇π(a) = ∇Σπ = ∇1 = 0 → 环境转移项对梯度无贡献（无模型 RL 的根基）
3. **因果性裁剪**：t 时刻动作不影响 t 之前的奖励 → 只加未来奖励

### 1.3 baseline 免费降方差

减 baseline b 不改变期望（∇E[b]=0），但方差大幅下降。最优 b 是奖励的条件期望
——这就是 value function 的动机，也是 GRPO 组均值的动机。

## 第 2 章 GRPO：扔掉 critic 的现代方案

### 2.1 从 PPO 到 GRPO 的唯一改动

```
PPO:  Â_t = GAE(r_t, V(s_t))   ← 需要 value network（多一个模型）
GRPO: 对每个 prompt 采样 G 个回答：
      Â_i = (r_i − mean(r_group)) / std(r_group)
      L = −E[min(r·Â, clip(r, 1±ε)·Â)] + β·D_KL(πθ ∥ π_ref)
```

### 2.2 组归一化的三重数学意义

| 机制 | 数学 | 直觉 |
|------|------|------|
| 组均值 baseline | E[∇·mean]=0 | 无偏降方差 |
| 组 std 归一化 | 尺度不变性 | 奖励从 [0,1] 变 [0,100] 梯度方向不变 |
| 组内相对 | 排名信号 | 「比同组好」比「绝对值高」更稳 |

### 2.3 GRPO 的代价与边界（我们的实验贡献）

- 组归一化**稀释 token 级信用分配**：长轨迹（agent 1-20 步）方差大时失效
  ——本项目只在单轮 2048 token 场景用 GRPO，长程任务建议 PPO+GAE
- **G 的选择**：G=4 是「归一化稳定性 vs rollout 成本」的平衡点

## 第 3 章 KL 散度与 k3 估计器（本项目的数学核心）

### 3.1 为什么 KL 必须估计而不能直接算

```
D_KL(πθ ∥ π_ref) = Σ_v πθ(v)·log(πθ(v)/π_ref(v))  ← 词表 15 万维求和不可行
单样本 MC 估计：只对「采样到的 token」算 logprob 差
```

### 3.2 四种估计器与选择

```
k1("kl"):        log p − log q            # 有偏，最常用
k2("mse"):       ½(log p − log q)²        # 无方向
k3("low_var_kl"): exp(q−p) − (q−p) − 1    # 无偏低方差（DeepSeek-R1 提出）
```
k3 的无偏性：E_q[k3] = D_KL（χ² 型展开首项）。**我们的全部实验用 k3**——
actor KL 惩罚和蒸馏 loss 同款，配置统一便于对照。

### 3.3 forward vs reverse KL（蒸馏的方向选择）

- forward KL（E_q log q/p）：mode-covering——覆盖 teacher 全部模式含错误
- reverse KL（E_p log p/q）：mode-seeking——只追 teacher 高概率区
- OPD 用 reverse：**「只学 teacher 确信的东西」**——我们的 low_var_kl 是 reverse
  KL 的单样本估计

## 第 4 章 OPD：把蒸馏搬进 RL 循环

### 4.1 三个升级（vs 离线 SFT 蒸馏）

| 维度 | SFT 蒸馏 | OPD |
|------|---------|-----|
| 数据 | teacher 的固定输出 | **student 自己的轨迹** |
| 信号 | 输出 token 的 CE | 逐 token 的 KL（teacher logprob）|
| 纠错位置 | 静态数据集 | **on-policy 分布**（无 exposure bias）|

### 4.2 我们的实现（veRL low_var_kl 纯 OPD）

```
distillation.enabled=True
distillation.distillation_loss.loss_mode=low_var_kl   # k3 估计 reverse KL
distillation.distillation_loss.use_task_rewards=False # 纯蒸馏（E5/E9p3）
# teacher=7B（E5）→ 24.7；teacher=student 自蒸馏（E9p3）→ 无增量
```

### 4.3 GLM-5 的 advantage 形式（蒸馏与 RL 的统一）

```
Â_{i,t} = sg[log π_teacher(y_t|x,y_<t) − log π_student(...)]
```
蒸馏信号直接进 policy gradient 管线（group size 可=1）——**SFT/RL/蒸馏在同一个
引擎里统一**。我们 E9p3 用 loss 形式（use_policy_gradient=False）实现了等价物。

## 第 5 章 概率校准（verifier 可信度的数学）

### 5.1 两级校准各管一个问题

```
温度 T：P_T(k)=softmax(logits/T)     → 保序修尖锐度（T 在验证集最小化 NLL）
Platt： V_cal = sigmoid(a·logit(V_raw)+b)  → 修偏移（a=1.571, b=10.333）
```
b=10.33 的解读：模型 V_raw 系统性偏低（保守模型）——Platt 把「保守的软分数」
抬到「真实的通过率」。

### 5.2 ECE 的完整定义与实测意义

```
ECE = Σ_b (n_b/N)·|acc_b − conf_b|
分桶后预测置信度 vs 实际频率的平均差。
ECE=0.0067 → 「说 0.8 就是 80%」误差 <0.7% —— 奖励信号可信度的硬指标
```

## 第 6 章 RG-OPD 门控：信号冲突的裁决机制

### 6.1 公式与四象限

```
g = I[(A>0 ∧ L_T>L_S+δ) ∨ (A≤0 ∧ L_T<L_S−δ)]
```
| | teacher 更确信 | teacher 更不确信 |
|---|---|---|
| 奖励正 | ✅ 蒸（强化）| ❌ 不蒸（保护另类解法）|
| 奖励负 | ❌ 不蒸（过滤 teacher 错误信念）| ✅ 蒸（推离）|

### 6.2 我们界定的边界条件（可发表结论）

自蒸馏（teacher=student）下 L_T−L_S 是噪声 → 门控随机砍半 → 叠加 V>0.5 条件
→ 通过率 0.10-0.18 → 无增量。**RG-OPD 的隐含前提：teacher 必须强于 student**。
改进方向：7B teacher + 沙箱结果做 A（OPDVR 原味）+ δ 放宽。

## 第 7 章 方法谱系一句话卡片（面试速记）

```
策略梯度 ——「log 概率的梯度就是尝试频率的调整」
PPO      ——「clip 防单步跳太远」
GRPO     ——「组内相对分替代 critic」
k3       ——「无偏低方差的 KL 单样本估计」
OPD      ——「student 自己的轨迹上，teacher 逐 token 纠正」
校准     ——「让模型说真话（保序 + 修偏移）」
门控     ——「两个老师意见一致才听讲」
```
