# Code-R1 项目深度教程与面试弹药库（INTERVIEW-MASTER）

> 面向面试的完整知识体系：公式理论 → 代码实现 → 工程实践 → 实验设计 → 方案权衡 → 100+ 面试问答。
> 配套：`docs/2026-09-07-FINAL-REPORT.md`（结果与结论）、`docs/PITFALLS.md`（踩坑手册）、
> GitHub `Bowen-mantou/code-r1-repro`（代码+TUTORIAL）。

---

# 第一章 项目全景（30 秒电梯陈述）

**我做了什么**：独立复现并扩展 Code-R1 代码推理强化学习研究——在 3B 模型上完成
「执行奖励 / teacher 蒸馏 / 手写过程奖励 / 学习型 verifier」四类训练信号的对照矩阵，
共训练评估 6 个模型，最优 LiveCodeBench 25.2（超官方 7B 基线），并发现修复了一个
让两轮实验结论作废的信号链路缺陷。

**核心数字**：6 模型 / 5 信号 / 4 基准 / ECE 0.0067 / 预算 ~¥388 / 独立完成全链路。

**一句话卖点**：我不仅跑通了前沿 RL 训练管线，还用「两个实验分数完全相同」的统计
巧合反向定位了一个静默信号缺陷——这是面试官最喜欢追问的故事。

---

# 第二章 公式理论层（从零推导）

## 2.1 策略梯度 → PPO → GRPO

### 策略梯度基础
目标：最大化期望奖励 `J(θ) = E_{τ~π_θ}[Σ r_t]`。
策略梯度定理：`∇J = E[Σ_t ∇log π_θ(a_t|s_t) · A_t]`。
`A_t`（advantage）决定「这个动作比平均水平好多少」，常见估计 GAE。

### PPO（Proximal Policy Optimization）
```
L_CLIP = E[min(r_t(θ)·Â_t, clip(r_t(θ), 1±ε)·Â_t)]
r_t(θ) = π_θ(a|s) / π_old(a|s)   # 重要性采样比
```
ε=0.2。clip 防止单步更新过大破坏策略。需要 value network（critic）估计 A——**贵一倍显存**。

### GRPO（Group Relative Policy Optimization，DeepSeek-R1）
**核心创新：扔掉 critic，用「组内相对」估计 advantage**：
```
对每个 prompt 采样 G 个回答（G=4）：
r_i = 该回答的标量奖励
Â_i = (r_i - mean(r_group)) / std(r_group)   # 组内 z-score
L = -E[min(r·Â, clip(r)·Â)] + β·D_KL(π_θ ∥ π_ref)
```
**为什么 group 归一化有效**：
1. 相对比较比绝对分更稳（奖励尺度漂移不影响梯度方向）
2. 天然 baseline：组均值就是蒙特卡洛 baseline，降低方差
3. 省掉 critic 的显存/计算（3B 场景省 ~20GB，这就是小模型能跑 RL 的关键）

**KL 惩罚**（防止策略偏离 ref 太远）：β·D_KL，我们 β=0.001，估计器用 k3（见 2.4）。

### 面试必背公式卡
| 公式 | 含义 |
|------|------|
| Â_i = (r_i − r̄)/σ | GRPO 组内优势 |
| L_CLIP | PPO 裁剪目标 |
| r = π_θ/π_old | 重要性采样比 |
| R = result + λ_v·V_cal | 我们的奖励函数（λ=0.1）|

## 2.2 KL 散度与 k3 估计器（本项目的数学核心）

**KL 定义**：D_KL(P∥Q) = E_P[log P/Q] = Σ p log(p/q)。

**为什么 RL 里不能直接算**：全词表求和不可行（词表 15 万）。用**单样本 MC 估计**：
采样到的 token 的 logprob 差。

veRL 支持四种估计器（kl_penalty 参数）：
```
k1 ("kl"):   log p − log q                        # 最常用，有偏
k2 ("mse"):  0.5·(log p − log q)²                 # 无方向性
k3 ("low_var_kl"):  exp(q−p) − (q−p) − 1          # 无偏低方差（DeepSeek-R1 提出）
```
其中 p、q 是采样 token 的 log 概率。**k3 是我们全部实验的选择**（actor KL 惩罚和蒸馏 loss 都用它）：
- 无偏性：E[k3] = D_KL（在 Q 支撑下的 χ² 型展开）
- 低方差：相比 k1 的 (log p/q)² 尾部，k3 的指数形式方差更小
- DeepSeek-R1 技术报告明确用了 k3 作为 KL 估计器

## 2.3 OPD（On-Policy Distillation）的数学

**本质**：把蒸馏从「SFT 拟合 teacher 输出」升级为「RL 循环内、student 自己的轨迹上、
teacher 给 token 级稠密信号」。

**我们的实现（E5/E9p3，veRL low_var_kl 模式）**：
```
loss_distill = k3(student_logprob, teacher_logprob)  # reverse KL：student 拉向 teacher
纯 OPD：use_task_rewards=False → 总 loss = 只有蒸馏项
```

**为什么 on-policy 优于离线 SFT 蒸馏**（面试要点）：
1. 纠错发生在 student **实际行为分布**上（无 exposure bias）
2. token 级稠密信号 vs SFT 的均匀修改力——GLM-5 论文证明 SFT 式蒸馏会破坏知识结构
3. GLM-5 的 advantage 形式：`Â_t = sg[log π_teacher − log π_student]`——与 GRPO 同构，
   蒸馏信号可以直接进 policy gradient 管线（group size 甚至可以 =1）

## 2.4 Learned Verifier 的数学（自研核心）

**问题**：手写 PRM（E2）效果差（LCB 21.0 垫底）——人工规则的信用分配是拍脑袋。
**出路**：数据驱动。

**建模**：1.5B 生成式模型，输入「题目+代码」，输出**6 分类**（代码能通过 0-5 个测试）：
```
P(k) = softmax(logits[:6])     # k ∈ {0..5}
V_raw = Σ k·P(k) / 5           # 期望通过率 ∈ [0,1]
```

**两级校准**（面试重点，这是 verifier 可信度的关键）：
1. **温度校准**：softmax 前除以温度 T，T 在验证集上搜（通常 >1 平滑过度自信）
2. **Platt 校准**：对 logit(V_raw) 做逻辑回归
   ```
   logit(V) = log(V/(1−V))
   V_cal = 1 / (1 + exp(−(a·logit(V) + b)))
   ```
   最终 a=1.571、b=10.333，**ECE = 0.0067**（几乎完美校准）

**ECE（Expected Calibration Error）**：
```
ECE = Σ_b |acc_b − conf_b| · n_b/n   # 分桶后「预测置信度 vs 实际正确率」的平均差距
```
ECE < 0.01 意味着「verifier 说 0.8 的代码，真的约 80% 通过」——**奖励信号的可信度上限**。

**奖励集成**：`R = result_bin + λ_v · V_cal`（λ_v=0.1）。
设计考量：结果主导（防奖励黑客：V 高但沙箱不过时 R 仍 ≤1.1 满分结构内 result 为 0），
verifier 只做**过程塑形**（shaping）——给「对了但差点」的轨迹梯度方向。

## 2.5 RG-OPD 门控的数学（E9p3）

**动机**：teacher 不总是对的（可能给错误解高概率、给正确但风格不同的解低概率）——
无条件蒸馏把 teacher 当无误神谕。

**Sign gate**（arXiv 2607.04037）：
```
g = I[(A>0 ∧ L_T > L_S+δ) ∨ (A≤0 ∧ L_T < L_S−δ)]
A   = verifier 分（我们：V_cal>0.5 视为正）
L_T = teacher 轨迹级 log-lik = Σ_t log π_t(token_t)   # 轨迹级求和
L_S = student 同理
```
语义：**正轨迹只在 teacher 比 student 更确信时蒸；负轨迹只在 teacher 更不确信时蒸**。
蒸馏 loss 乘 gate → 冲突轨迹不参与。

**我们的观测**：gate 通过率仅 0.10-0.18。原因（面试必讲）：teacher=student 自蒸馏时
L_T≈L_S 的差异是数值噪声 → 随机砍一半 → 叠加 V>0.5 条件（正确率 25% 的模型约 1/4 轨迹）
→ ~10%。**结论：RG-OPD 的前提是 teacher 强于 student**——这是我们实验界定的方法边界条件。

## 2.6 奖励函数全景（reward_fn.py）

```
R = result_bin + λ_v·V_cal      （use_verifier=True，λ_v=0.1）
R = result_bin                  （无 verifier，E1/E5/阶段 2 实际生效的）
result_bin = 1 如果沙箱执行通过全部测试，否则 0
```
配套 coder1 格式分：模型输出必须含 `<think>` 前缀，满分 1.1（format 0.1 + answer 1.0）。

---

# 第三章 代码实现层（逐文件讲解）

## 3.1 训练入口：run_e7.sh（GRPO+Verifier 定稿配置）

```bash
MODEL_PATH=/root/autodl-tmp/models/e5-final-3b   # E9p2 版起点；E7 用 Qwen 基模
DATA=(
    algorithm.adv_estimator=grpo          # 算法：GRPO
    data.train_batch_size=8               # 每步 8 个 prompt × n=4 = 32 轨迹
    data.max_prompt_length=2048
    data.max_response_length=2048
    data.truncation=left                  # 超长左截断（保尾部代码）
    reward.custom_reward_function.path=/root/autodl-tmp/reward_fn.py
    +reward.custom_reward_function.reward_kwargs.use_verifier=True   # E7/E9p2 核心
    +reward.custom_reward_function.reward_kwargs.lambda_v=0.1
)
ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6            # 小学习率（RL 稳定性）
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
    actor_rollout_ref.actor.use_kl_loss=True         # KL 惩罚开
    actor_rollout_ref.actor.kl_loss_coef=0.001       # β=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl  # k3 估计器
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2560  # 显存约束（见 4.1）
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=False # 省 ~2GB 激活
    actor_rollout_ref.actor.checkpoint.save_contents=[model,extra]  # 13G/份
)
ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25  # KV 池 ~13.6GB
    actor_rollout_ref.rollout.n=4                   # GRPO 组大小 G=4
    +actor_rollout_ref.rollout.enable_sleep_mode=False     # 6000D 不支持的 cumem 分配器
)
TRAINER=(... total_epochs=2 save_freq=32 max_actor_ckpt_to_keep=1 ...)
```

**面试要点**：每个配置键都能说出「为什么」——lr 1e-6 防策略崩塌、kl 0.001 防漂移、
n=4 是 GRPO 组归一化的最小稳定值、token_len 2560 是显存账算出来的。

## 3.2 数据流全链路（本项目最重要的架构知识）

```
AgentLoopWorkerTQ（agent loop）
  ├─ vLLM 生成 n=4 条回答（卡 0）
  ├─ 沙箱执行 → result_bin（reward_fn 内）
  ├─ verifier actor 打分 → V_cal（Ray 调用，卡 1）
  └─ final_output.extra_fields["verifier_score"] = V_cal
        ↓
TransferQueue（TQ，AgentLoopManagerTQ.async_kv_batch_put）
  └─ list_of_dict_to_tensordict(fields)   ← extra_fields 作为 dict 整体存（不展平！）
        ↓
trainer_base.py：reward 阶段
  └─ data.pop("extra_fields") → reward_extra_infos_dict
        ↓
naive.py reward manager：per-sample
  └─ non_tensor_batch.get("verifier_score") → extra_info → reward_fn
        ↓
loss 计算（transformer_impl / distillation losses）
  └─ data.get("verifier_score") → RG-OPD gate
```

**链路 bug 的根因（面试故事的核心）**：agent_loop 写进 `extra_fields`（dict）的键
**不会被 TQ 展平**——`list_of_dict_to_tensordict` 把 dict 整存为 NonTensorStack。
下游读 `non_tensor_batch["verifier_score"]` 永远 None → reward_fn fallback 静默回退
纯 result_bin。修复：TQ put 前把 verifier_score **提为顶层 field**（patch_verifier_topkey.py）。

## 3.3 verifier_server.py：Ray 单例 actor + 内部攒批

**架构演进**（面试讲「为什么这样设计」）：
1. v1：8 个 agent loop worker 各加载一份 1.5B verifier → 8×4GB OOM
2. v2：**Ray 命名单例 actor**（`ray.get_actor("e7_verifier_actor")`）——8 worker 共享 1 份模型，
   用 `ActorAlreadyExistsError` 兜底 check-then-act 竞态
3. v3：**内部攒批**——actor 里 queue + 20ms 窗口把 per-sample 调用合并成 batch 推理，
   步耗时 51s → 19.7s

```python
class _VerifierActor:
    def __init__(self, model_path, calibration_path):
        self.verifier = VerifierInference(...)   # 1.5B + Platt 校准参数
        self._q = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()
    def _worker(self):
        while True:
            items = [self._q.get()]              # 阻塞等第一条
            deadline = time.time() + 0.02        # 20ms 攒批窗口
            while time.time() < deadline:
                try: items.append(self._q.get(timeout=deadline - time.time()))
                except queue.Empty: break
            vs = self.verifier.score_batch(...)  # 合并 batch 推理
            for item, v in zip(items, vs): item[0].put(v)   # future 回传
```

**VerifierInference.score_batch 推理路径**：
```
build_inputs: "题目 + ```python\n代码\n``` + 指令(回答0-5)"
tokenize → 取最后一个有效 token 的 logits
digit_logits = logits[:, digit_ids]        # 只取 "0".."5" 六个 token 的 logit
probs = softmax(digit_logits)
E[k] = Σ k·P(k) → V_raw = E[k]/5
Platt: V_cal = sigmoid(a·logit(V_raw) + b)
```
**设计点**：生成式 6 分类（而非回归头）——复用预训练的语言建模能力，输出空间对齐
「数字 token」的自然语义；截断策略 prompt≤1024 字符、代码取尾部 2048 字符（题目头部
信息量高、代码尾部是关键）。

## 3.4 RG-OPD gate 的实现（losses.py，三版迭代）

```python
# v3 最终版（patch_rg_opd_v3.py）
if getattr(loss_config, "verifier_gating", False):
    vs = data.get("verifier_score", None)
    if vs is not None:
        try: vs_l = vs.tolist() if hasattr(vs, "tolist") else list(vs)
        except Exception: vs_l = None          # v1 教训：NonTensorStack 不能直接 as_tensor（空张量）
        if vs_l is None or len(vs_l) != student_log_probs.shape[0]:
            print("[rg-opd] WARN ... gate disabled")   # fallback：无条件蒸馏
        else:
            vs_t = torch.as_tensor(vs_l, ...)
            mode = getattr(loss_config, "gating_mode", "rlopd")
            if mode == "soft":                 # E6' 原版：V^γ 加权
                w = vs_t ** gamma; losses *= w.unsqueeze(-1)
            else:                              # rlopd sign gate
                mask_t = response_mask.to(dtype)
                lt = (teacher_log_probs * mask_t).sum(dim=-1)   # 轨迹级 L_T
                ls = (student_log_probs * mask_t).sum(dim=-1)   # 轨迹级 L_S
                pos = (vs_t > 0.5) & (lt > ls + delta)
                neg = (vs_t <= 0.5) & (lt < ls - delta)
                gate = (pos | neg).to(dtype)
                distillation_losses *= gate.unsqueeze(-1)
                print(f"[rg-opd] gate frac: {(gate>0).float().mean():.3f}")  # v3 观测
```

**三版迭代故事**：v1 as_tensor 空张量崩溃（shape 0 vs 10）→ v2 NonTensorStack 转 list
修复 + fallback 兜底 → v3 加 gate frac 观测（这就是「gate 真生效」的铁证来源）。

## 3.5 gen_lcb_fast.py：两阶段自举批量生成

```
阶段 1：串行生成 80 题（均匀采样）→ 记录每题真实生成 token 数（长度先验）
阶段 2：剩余 800 题装箱批量：
  schedule()：按预估长度降序，装箱 ratio≤1.5，batch≤8（避免短题等长题的木桶效应）
  动态 max_new_tokens = 批内最大预估×1.5+64（而非统一 2048）
  撞顶检测：生成长度 == max_new → 单独 2048 重跑（保证与串行口径逐 token 一致）
```
**诚实结论**：真实答案中位数仅 128 token → 744/800 题撞顶重跑 → 总时长与纯串行打平
（75 vs 80 分钟）。**教训：时间优化必须小样本实测，装箱正确 ≠ 吞吐快**——面试讲这个
负结果比讲「3× 提速」更可信。

---

# 第四章 工程实践层（分布式训练生存指南）

## 4.1 显存账（面试高频：为什么双卡、每块显存去哪了）

A800 80GB ×2 的分配（E7/E9p2 实测）：
```
卡 0（student 训练）：
  FSDP worker 峰值         58.6GB   ← 前向+反向+激活（gradient checkpointing 后）
  vLLM rollout 引擎        13.6GB   ← util=0.25 的 KV 池
  ref 模型（offload）        ~0      ← param_offload=True 换入换出
  合计 ≈72.2GB / 79.25GB 可用   → 压线（这就是 token_len 2560 的原因）
卡 1（verifier）：
  verifier 1.5B（Ray actor） 4GB
```
**单卡死刑的账**（E5 教训）：worker 37 + 7B teacher 引擎 38-40（其中 ~24GB 是 vLLM
不透明固定开销，util 调小只降 1GB）+ rollout 6.2 > 任何单卡容量。

**面试讲法**：「我先算账再开机」——引用实测指标（perf/max_memory_allocated）而不是
拍脑袋，每换一次配置重算一次账。

## 4.2 双卡与资源池

- GRPO+Verifier：student 独占卡 0，verifier 走 Ray actor `runtime_env={CUDA_VISIBLE_DEVICES:1}`
- OPD：student 卡 0 + teacher 卡 1（veRL 原生 distillation 资源池，n_gpus_per_node=1）
- 单卡 OPD 需要 patch 把 teacher 映射 global_pool 共卡——**验证过是死刑，双卡是唯一解**

## 4.3 训练生命周期管理（三件套）

```
watchdog（remote_watchdog_e9p2.sh）：30s 轮询 main_ppo 进程
  → 进程消失 → 查日志错误关键词（排除 RewardLoopWorker 正常噪音）
  → 崩溃/完成 → 状态落盘 → sleep 120 → /usr/bin/shutdown（AutoDL 脚本）
清理器（clean_old_ckpt.sh）：max_keep=1 在 resume 场景失效 → 外挂兜底
  → find global_step_* -mmin +3 ! -name latest → 删（3 分钟阈值跟得上 13 分钟保存节奏）
断点续训：veRL 从 latest_checkpointed_iteration 自动 resume（save_freq=32 保命）
```
**磁盘事故链**（PITFALLS #8）：切双卡后没重查磁盘 → ckpt 13G 写入满盘 → 白训 32 步。
**教训：换卡/换实例后先跑资源检查（磁盘/显存/环境），"上次检查过"不算数。**

## 4.4 沙箱安全（执行模型生成代码的基建）

`sandbox_exec.py`：seccomp BPF 禁网络 + RLIMIT（CPU 30s/NPROC/NOFILE/FSIZE/AS）
+ setuid nobody + exec。
**超时漏洞**：模型代码 spawn 孙进程，`subprocess.run(timeout)` 只杀直接子进程 → 管道
hang → 训练卡死 54 分钟。修复：`start_new_session=True` + `os.killpg(SIGKILL)`
+ RLIMIT_CPU=30 内核兜底（用户态 kill 不到的内核保证）。

## 4.5 无卡预演方法论（省钱核心）

**原则：能在 ¥0.1/hr 无卡模式验证的，绝不上 ¥10/hr 的 GPU**：
- 配置 dry-run（hydra 键合法性）
- 数据校验（sha256、行数、JSON 合法性——抓到了 NUL 空洞）
- patch 无卡测试（mock TensorDict 跑 gate 逻辑，5 个 case）
- 代码审查（跨行 sed 失败这类低级 bug）
**反面教材**：E7 的七次崩溃大半是「没在无卡验证的设计错误」（8 worker 各加载 verifier）。

## 4.6 数据与传输

- LCB 口径：main 分支 5 文件 = 880 题（2025-06-05 冻结）；**历史 commit 冷对象 60KB/s vs
  main 热缓存 31MB/s**——pin commit 是上个会话的错误
- aria2c 16 连接 + `--file-allocation=falloc`（防 NUL 空洞）+ 官方 LFS sha256 校验
- SFTP put-dir 递归 mkdir（AutoDL 网关不支持 stdin 写，base64 命令通道只适合小文件）
- **模型备份纪律**：每个实验产物立即下载本地（E5 模型丢失事故的教训）

---

# 第五章 实验设计与原因（科学方法层）

## 5.1 2×2 矩阵的设计逻辑

```
            无 PRM              有 PRM
GRPO    E1（执行奖励）      E2（GRPO+手写PRM）
OPD     E5（teacher 蒸馏）   E6（未跑：OPD+PRM）
```
**设计原则：每相邻两格只差一个变量**——E1 vs E2 隔离「PRM 的贡献」、
E1 vs E5 隔离「信号类型（稀疏执行 vs 稠密 teacher）」、E5 vs E6 隔离「teacher 上 PRM
的增量」。**矩阵是控制变量法在训练范式对比上的应用**。

## 5.2 每个实验的假设与结论（含修正后真相）

| 实验 | 原假设 | 结果 | 修正后解读 |
|------|--------|------|-----------|
| E1 | 执行奖励基线 | LCB 19.8 | 4ep 过训退化；2ep 才是峰值（E7 证明）|
| E5 | teacher 稠密信号更强 | 24.7 | ✅ 成立（+4.9 vs E1）|
| E2 | 手写 PRM 补过程奖励 | 21.0 | ❌ 手写信用分配全劣 |
| E7 | verifier 奖励提升 | 25.2 | ⚠️ verifier 未生效——实际是「纯 GRPO 2ep」的上限证明 |
| E9p2 | warm-start 提升终点 | 25.2 | ❌ 起点不影响终点（= E7 双证）|
| E9p3 | 门控蒸馏恢复/提升 | 24.8 | ❌ 自蒸馏+稀疏门控无增量（界定 RG-OPD 边界）|

## 5.3 E9 三阶段的设计依据（论文驱动）

```
阶段 1（E5 OPD）：Sequential Beats Joint（2609.04108）——「OPD 先、RL 后」最优
阶段 2（GRPO）：warm-start 对照（E9p2 vs E7 只差起点变量）
阶段 3（门控蒸馏）：GLM-5（2602.15763）在线蒸馏恢复 + RG-OPD（2607.04037）门控
```
**顺序对称实验**（本项目原创性所在）：
- 「先蒸馏后 RL」→ 无增量（E9p2，与 2606.09059「Stage-1 controls entropy regime,
  not outcome」互证）
- 「先 RL 后蒸馏」→ 无增量（E9p3）
- **合起来：顺序问题在我们场景闭合——终点由配方决定，不由起点或收尾决定**

## 5.4 五条最终结论的形成逻辑

1. GRPO(2ep) ≈ OPD 在难题上打平（25.2 vs 24.7，噪声内）
2. 2ep 峰值 / 4ep 退化（entropy collapse，文献 TS-OPSD 呼应）
3. warm-start 不改变 RL 终点（双证：E9p2=E7 + 论文 2606.09059）
4. 手写 PRM 全维度劣于数据驱动信号
5. RG-OPD 门控的自蒸馏边界条件（teacher 必须强于 student）

**负结果的价值陈述**（面试万能句式）：「负结果也是结果——它划定了方法的适用边界，
而边界正是下一个研究者最需要的信息。我这条'自蒸馏门控失效'的结论，让后来者不会在
teacher=student 的设置上浪费一个实验周期。」

---

# 第六章 方案权衡与考量（优缺点全景）

## 6.1 四类信号的优缺点对比

| 信号 | 优点 | 缺点 | 我们的数据 |
|------|------|------|-----------|
| 执行奖励（GRPO）| 无偏 ground truth、防黑客、稀疏但可信 | 难题 credit assignment 差；过训退化 | 2ep=25.2 / 4ep=19.8 |
| teacher 蒸馏（OPD）| 稠密 token 级、样本效率高、稳 | 上限受 teacher 约束；~24GB 引擎开销；无条件蒸馏会吸收 teacher 错误 | 24.7 |
| 手写 PRM | 可解释、无需标注 | 人工规则拍脑袋、与 ground truth 脱节 | 21.0 垫底 |
| learned verifier | 数据驱动、校准后可信（ECE 0.0067）| 需额外训练管线；信号链路工程复杂 | 门控版 24.8（自蒸馏边界）|

## 6.2 关键设计取舍

**为什么 λ_v=0.1 而不是更大**：RWOPD 论文证明「verifier 独立稀疏奖励 ≤ SFT 基线」；
PASS 论文警告 naive 混合掉双位数。0.1 = 结果主导（防黑客）+ 微塑形。λ 小是安全的默认。

**为什么 verifier 用 1.5B 生成式而非 3B 回归式**：显存约束（4GB 共卡）；生成式复用
预训练能力；6 分类输出天然有语义。

**为什么 reward 是「结果主导」**：奖励黑客的实证风险——V 高但沙箱不过时，R 仍被
result_bin=0 主导，模型无法靠糊弄 verifier 拿分。

**为什么评估 LCB 而不是只测 HumanEval**：难度分层——E1 的 HumanEval 86.6 冠军却在
LCB 19.8 垫底。只测简单题会得出「PRM 有害、OPD 无用」的错误结论（E5 时代就发现了：
加了竞赛基准，「难度依赖」的真相才浮现）。

**为什么双卡而不是加大单卡**：A800 上限 80G，显存账 72.2G 已压线；teacher 引擎的
24GB 不透明开销无法通过配置压缩——双卡是工程现实而非偏好。

## 6.3 预算与止损方法论

红线 ¥400 实际花 ~¥388。止损规则：无卡预演一切 → GPU 只跑验证过的 → watchdog 自动
关机 → 每步记账（MISTAKES 账本：可避免浪费 ~¥25，磁盘事故占 ¥10.5 最大）。
**面试讲法**：「我在真金白银的约束下做研究，这训练了我对'什么值得花 GPU 时间'的判断力。」

---

# 第七章 面试问答库（100+ 题，分类弹药）

## 7.1 理论类（RL 基础，30 题）

**Q1. 用一句话解释 GRPO 和 PPO 的区别？**
PPO 需要 critic（value network）估计 advantage，GRPO 扔掉 critic，用同一 prompt 的
G 个回答的组内相对分（z-score）直接当 advantage。省一半模型显存，样本利用率略降。

**Q2. GRPO 的组归一化为什么有效？给数学直觉。**
组均值是蒙特卡洛 baseline（无偏减方差的标准技巧）；除以 std 让优势对奖励尺度不变
（奖励从 [0,1] 变 [0,10] 不改变梯度方向）。代价是「组内都好」时优势都小——所以 G=4
是平衡点。

**Q3. KL 惩罚的作用？为什么 RL 需要它？**
防止策略在优化 reward 时偏离 ref 太远导致语言能力崩塌（reward 黑客/格式崩坏）。
β 太小没约束、太大锁死探索。我们 β=0.001。

**Q4. k3 估计器（low_var_kl）是什么？为什么选它？**
k3 = exp(q−p) − (q−p) − 1（p,q 为 logprob）。DeepSeek-R1 提出的 KL 无偏低方差
单样本估计器；相比 k1=log(p/q) 方差更小、训练更稳。veRL 的 kl_penalty 原生支持。

**Q5. RL 里为什么需要 gradient checkpointing 和 offload？**
显存换计算：checkpointing 不存中间激活、反向时重算（省 ~50% 激活）；optimizer/param
offload 把状态放 CPU。我们的 ref 模型 param_offload=True，worker 峰值 58.6GB。

**Q6. reward 黑客是什么？你项目里怎么防的？**
模型钻奖励函数漏洞拿高分但任务没做好（如刷格式分、生成测试通过但语义错的代码）。
防线：result_bin 主导（λ_v=0.1）+ 沙箱真实执行 + 格式分上限 0.1。

**Q7. 手写 PRM 为什么效果差？**
信用分配是拍脑袋：人工规则（长度/关键词/中间检查点）与真实「这步对最终对错有多少
贡献」脱节；而且 PRM 的分与 ground truth 无校准——分高不一定对。

**Q8. ORM 和 PRM 的区别与取舍？**
ORM（结果奖励模型）：对完整答案打一个分——低方差、易训，但稀疏、易被黑客；
PRM（过程奖励）：对中间步骤打分——稠密、可解释，但标注难、易被游戏。
我们的路线：用「执行结果」当 ORM（无偏 ground truth）+ learned verifier 当塑形信号。

**Q9. 什么是 entropy collapse？你遇到过吗？**
RL 后期策略熵坍缩到接近 0：只输出少数高奖励模板，多样性死掉，泛化崩。
E1 的 4ep 退化（LCB 19.8）高度疑似；文献 TS-OPSD 专门研究「RL 后蒸馏 reheating 恢复
熵」。这是「为什么 2ep 是峰值」的理论解释。

**Q10. importance sampling ratio 超出 clip 范围会怎样？**
ratio·Â 被 clip 到 [1−ε, 1+ε]·Â——防止单步策略跳太远。GRPO 同样用 clip（我们
clip_ratio=0.2 默认）。

**Q11. 什么是 on-policy / off-policy？RL 训练为什么必须 on-policy？**
on-policy：更新用的数据来自当前策略；off-policy：来自旧策略/缓冲。PPO/GRPO 靠
importance sampling 修正，但策略漂太远 IS ratio 爆炸 → 每步必须新鲜 rollout
（这也是「步耗时」的大头）。

**Q12. 为什么代码 RL 用「执行奖励」而数学 RL 常用 RM？**
代码有完美验证器（单元测试）——ground truth 免费且无偏；数学的证明验证难自动化，
才需要 RM。**有验证器的任务永远优先验证器**（这是我们的路线正确性来源）。

**Q13. RLVR 是什么？你的项目是 RLVR 吗？**
RL with Verifiable Rewards：用可验证规则（代码测试/数学答案）当奖励的 RL。
E1/E5/E7 本质都是 RLVR（执行奖励），verifier 是额外塑形。

**Q14. 温度校准和 Platt 校准的区别？**
温度：softmax 前除以 T，改变分布尖锐度（单一参数，保序）；Platt：对 logit 做逻辑
回归（两参数仿射，可改变排序）。我们两级都做：温度粗调 + Platt 精调，ECE 0.0067。

**Q15. ECE 是什么？为什么 verifier 需要它？**
Expected Calibration Error：分桶后「预测概率 vs 实际频率」的平均差。verifier 的输出
要当奖励信号用——「说 0.8 实际 0.5」会让 RL 学到错误梯度。校准后信号才可信。

**Q16. 什么是 credit assignment 问题？代码 RL 里为什么严重？**
「最终失败该怪哪一步」的问题。稀疏奖励（只给最终对错）下，长轨迹的早期 token 分不到
有效梯度。这就是过程奖励/verifier 塑形的动机——给中间状态梯度方向。

**Q17. KL 的方向性：forward vs reverse KL，蒸馏该用哪个？**
forward KL（E_q log q/p）：mode-covering，会覆盖 teacher 所有模式（包括错的）；
reverse KL（E_p log p/q）：mode-seeking，只追 teacher 高概率区。OPD 用 reverse KL
（我们的 low_var_kl 是 reverse KL 的估计器）——「只学 teacher 确信的东西」。

**Q18. GRPO 组大小 G 的影响？**
G=1 退化成 REINFORCE+基线缺失；G 越大组内归一化越稳但 rollout 成本 ×G。我们 G=4：
成本可控 + 归一化稳定。

**Q19. 为什么 RL 学习率比 SFT 小两个数量级（1e-6 vs 1e-4）？**
RL 的梯度来自 advantage 估计（高噪声）+ 策略分布敏感（更新太大会崩 KL 约束）。
小步慢走是稳定性第一。

**Q20. 蒸馏和 RL 的本质区别？**
蒸馏：模仿（学 teacher 的分布，上限 = teacher）；RL：探索（用奖励搜索，可能超 teacher）。
我们的数据完美演示：E5 蒸馏 → 24.7（teacher 7B 的知识压缩上限）；GRPO → 25.2
（用执行奖励突破）。**「OPD 上限受 teacher 约束」的实证。**

**Q21. 什么是 exposure bias？OPD 怎么解决？**
离线蒸馏训练时学生只见过 teacher 的分布，推理时却吃自己的输出（分布漂移）。
OPD 在 student 自己的轨迹上纠错——训练分布 = 推理分布。

**Q22. policy gradient 的高方差问题与解法？**
单轨迹梯度方差巨大。解法谱系：baseline（组均值）、优势函数（A=Q−V）、
GAE（λ 折中偏差-方差）、clip（PPO 防大跳）。GRPO 是「组 baseline + clip」的组合。

**Q23. 为什么 3B 模型能超过官方 7B 基线？**
官方 7B 基线（LCB ~24）是纯 SFT/轻量调优；我们的 3B 经过执行奖励 RL 的搜索强化。
小模型 + 好 RL 管线 > 大模型 + 差管线——这正是 Code-R1 论文的核心主张（我们复现成功）。

**Q24. 训练和推理的 max_len 为什么分开？**
训练 2048（显存：KV+激活随长度平方级）；推理评估 2048 生成上限一致保证口径。
超长题左截断保尾部（代码尾部信息密度高）。

**Q25. 什么是 stop gradient（sg）？GLM-5 公式里为什么用？**
sg[log π_t − log π_s]：把 teacher-student 差当**常数奖励**，不反传梯度给 teacher
（teacher 冻结）。让蒸馏信号进 RL 的 advantage 管线而不引入二阶优化。

**Q26. 多轮 RL 会遗忘吗？怎么恢复？**
会——GLM-5 报告顺序优化不同目标累积退化。恢复方案：混入前序最优模型的在线蒸馏
（GLM-5 OPD）、reheating（TS-OPSD）。我们 E9p3 尝试了恢复但自蒸馏设置无增量。

**Q27. 什么是 reward shaping？你的 λ_v·V 是 shaping 吗？**
在稀疏真奖励上叠加中间信号的势函数/辅助项，保序前提下加速学习。λ_v·V_cal 是
shaping（结果主导保序，verifier 给中间梯度）。

**Q28. 为什么 PPO 的 clip 用 1±ε 而不是硬截断？**
软 clip 让目标可导且超界样本梯度为零（不学也不反学），硬截断会引入不可导点。

**Q29. 沙箱为什么用 seccomp 而不是容器/VM？**
容器在 AutoDL 容器内嵌套不可用（firejail 自我降级实测）；seccomp BPF 只禁网络系统
调用，轻量 + 内核级；RLIMIT_CPU 兜底无限循环。

**Q30. 验证集过拟合在 RL 里长什么样？**
训练 reward 涨但 benchmark 掉（E1 的 4ep 是典型：LCB 从峰值退化）。对策：test_freq
定期评估 + 停早（2ep）+ 保留最好 ckpt。

## 7.2 项目特定类（你的研究故事，25 题）

**Q31. 讲讲你发现信号链路 bug 的完整过程？**
① 起疑：E7 与 E9p2 的 LCB 分数小数点后 15 位完全相同（都是 222/880）——统计上
~2.5% 概率，但直觉不对；② 交叉验证：逐题 diff 生成结果，849/880 不同（确实两个模型）
→ 排除「读错文件」；③ 追数据流：agent_loop 写入点（extra_fields["verifier_score"]）
→ TQ 存储（list_of_dict_to_tensordict 不展平 dict）→ naive.py 读取点
（non_tensor_batch.get 恒 None）→ reward_fn fallback；④ 修复：顶层字段提取 +
NonTensorStack→list 转换；⑤ 验证：gate frac 0.18 打印 + WARN 消失。

**Q32. 这个 bug 为什么训练不崩？**
fallback 设计：reward_fn 读不到 v 时静默回退 result_bin——系统「优雅降级」了，代价是
科学结论失效。这是「静默降级」类缺陷的教科书案例：**不报错 ≠ 没问题**。

**Q33. 如果没有那次巧合，这个 bug 会被发现吗？**
可能不会——所有工程监控（watchdog/显存/磁盘）都正常。教训已写进 PITFALLS：每个信号
源要启动 smoke 打印实际值 + reward 成分分解监控（verifier 项恒 0 = 信号没进）。

**Q34. 修复后重新训练了吗？**
没有（预算 ¥45 超出剩余）。但修复后的链路在 E9p3 的门控里真实生效（gate frac 铁证）。
「verifier 作为奖励注入」的正解实验列为未来工作——基建全就绪，零调试成本可启动。

**Q35. RG-OPD 门控为什么在你项目里没效果？给三个原因。**
① teacher=student 自蒸馏：L_T≈L_S 是噪声比较，蒸馏无新知识可传（根本原因）；
② 通过率仅 10-18%：信号过稀疏，每批 1-2 条有效轨迹；③ 已收敛模型上扰动：
E9p2 在 25.2 收敛带，KL 拉向旧分布只会偏离。**结论：RG-OPD 的前提是 teacher 强于
student——这是方法论文没写明的隐含前提，我们实验界定了。**

**Q36. 为什么 E7 和 E9p2 分数完全一样这件事本身就是重要发现？**
因为两者的 verifier 都未生效 → 实际都是「纯 GRPO 2ep」→ 只有起点不同（基模 vs
E5 蒸馏）→ 终点相同 = **warm-start 不改变 RL 终点**。与同期论文 arXiv 2606.09059
（Stage-1 controls entropy regime, not outcome）独立互证。

**Q37. verifier 为什么选 1.5B 而不是 3B 同规模？**
① 显存：共卡 4GB（3B 要 8GB+）；② 速度：rollout 后打分在关键路径上，小模型快；
③ 任务简单：k/N 预测不需要大模型（校准后 ECE 0.0067 证明 1.5B 够用）。

**Q38. verifier 的 6 分类输出怎么想出来的？**
「代码能通过几个测试」天然是 0-5 的离散量（LCB 每题 5 个测试）——生成式模型输出
数字 token 的概率分布就是分类器，零额外结构、复用预训练语义。比回归头更稳。

**Q39. λ_v 怎么定的 0.1？有调参吗？**
预算内没做 λ 扫描。0.1 是「结果主导」原则下的保守默认：RWOPD/PASS 论文都警告过程
信号占比过高有害。调 λ 是未来工作。

**Q40. 为什么阶段 2 的步耗时（25-35s）比 E7（20s）慢？**
E5 起点模型的生成长度分布更长（蒸馏模型输出更啰嗦）+ 撞顶重跑更多（762 vs 744）。
**起点影响过程（轨迹分布），不影响终点（分数）**——又一个「熵 regime」的证据。

**Q41. 你的「顺序对称实验」是什么？为什么有价值？**
「先蒸馏后 RL」（E9p2）与「先 RL 后蒸馏」（E9p3）构成顺序翻转对照对，两者都无终点
增量 → 结论：顺序在我们场景不改变终点。文献里 Sequential Beats Joint 只做了单方向
（先蒸馏），反向是空白——负结果补上了这一半。

**Q42. 如果重新做这个项目，第一件事改什么？**
加「信号有效性自检」：训练前 smoke 打印每个信号的实际值 + 每 N 步 reward 成分分解。
这会在 E7 第一分钟暴露链路 bug，省两轮实验（~¥120）。

**Q43. 手写 PRM 的 E2 具体怎么做、为什么失败？**
用 Code PRM 规则（中间步骤的启发式分：长度惩罚、语法检查、中间检查点）加权奖励。
失败原因：规则与「最终对错」相关性弱 → 信用分配噪声比信号还大 → 比无 PRM 的 E1 还
低 3.1 分（LCB 21.0 vs 19.8 是 E1 的 4ep 退化基线，公平比较是 25.2——差距更大）。

**Q44. 为什么用 LCB 当主基准？**
① 竞赛难度（区分度：四实验差距主要在此）；② 官方 7B 基线可比；③ 880 题口径稳定。
HumanEval/MBPP 太简单（E1 86.6 的模型在 LCB 只有 19.8）——只用简单基准会误判方法优劣。

**Q45. 你的五个结论里哪个最有价值？为什么？**
「warm-start 不改变 RL 终点」——它挑战了「蒸馏先行」的流行实践（Sequential Beats
Joint 主张 OPD 先行），并解释了「为什么长 RL 下初始化不重要」。负结果 + 论文互证 +
直接指导预算分配（省掉阶段 0 的重跑）。

**Q46. 如果给你 ¥1000 预算，你怎么花？**
① ¥45 阶段 A（verifier 奖励修复版——唯一未验证的正增量候选）；② ¥20 阶段 3 换
7B teacher 重跑（RG-OPD 完整语义）；③ ¥100 4ep 曲线研究（2ep→4ep 的退化机制，
配 reward/熵监控）；④ 其余：λ 扫描、G 扫描、更大模型（7B student）单点验证。

**Q47. 你怎么证明「verifier 门控真的生效」而不是代码空转？**
三级证据：① 无 WARN（数据到达 loss 函数）；② gate frac 0.10-0.18 的每批打印
（真实过滤发生）；③ 与 E7 时代的对比（当时同样代码路径读不到数据，WARN 刷屏）。
**观测即验证**——这就是 v3 加统计打印的原因。

**Q48. 你项目的可复现性如何保证？**
固定 LCB 口径（880 题 main 冻结版 + sha256 校验）、配置全落脚本、评估断点续跑、
结果原始 json 保留、PITFALLS 记录所有环境坑。

**Q49. 项目里你最自豪的技术决策？**
Ray 单例 verifier actor + 20ms 攒批：把 8 个 worker 的 per-sample 调用合并成 batch
推理，步耗时 51s → 19.7s。小而美的系统设计——「先数清楚调用方，再设计常驻服务」。

**Q50. 项目里最失败的技术决策？**
gen_lcb_fast 的装箱优化：模拟验证全绿（111 批完美装箱），实际撞顶重跑 744 题、
总时长与串行打平。**模拟验证了调度正确性，没验证真实吞吐分布**——时间预估必须
小样本实测的教训。

**Q51. 你为什么认为 3B 全参比 LoRA 好？**
Code-R1 论文本身是 3B 全参；LoRA 在「大幅改变行为」的 RL 场景表达力受限（rank 瓶颈），
且我们的预算够全参（3B bf16 6GB 权重）。小模型全参是大模型 LoRA 的预算等价物。

**Q52. E2 的模型丢了你慌不慌？**
不慌但记教训：E2 结论已记录（分数在文档里），丢失只影响「重评/对照」能力。教训写进
PITFALLS #15：**每个实验产物立即本地备份**（E5 模型找回是运气，不能靠运气）。

**Q53. 你的「无卡预演」方法论具体是什么？**
一切不依赖 GPU 的验证（配置合法性、数据校验、patch 逻辑 mock 测试、代码审查）在
¥0.1/hr 无卡模式做；GPU 开机只跑「验证过的东西」。E7 七次崩溃里大半是没预演的设计
错误——学费换来的方法论。

**Q54. 你项目里的「技术债」有哪些？**
① reward_fn 的 fallback 静默（本应告警）；② naive.py 的重复 patch 代码块（编辑事故）；
③ max_keep=1 的 resume 失效靠外挂清理器兜底；④ 评估脚本 MODEL 路径硬编码（每次 sed）。

**Q55. 最难 debug 的一次是什么？**
就是链路 bug：没有报错、没有性能异常、只有「分数巧合」这一个线索，从统计学直觉
追到框架存储语义。花了一整天，最后一行代码修复——**难度在「发现问题是问题」**。

## 7.3 工程类（分布式训练/系统，20 题）

**Q56. 80GB 卡跑 3B 训练为什么还紧张？显存都去哪了？**
权重 6GB 只是零头。大头：FSDP 激活（前向中间张量）+ 优化器状态（Adam 的 m/v 各 4 字节/
参数）+ 梯度 + vLLM 的 KV 池。3B 全参 Adam 状态 ≈ 6×3 = 18GB，激活在 batch 8×2048
下 ~20-30GB，KV 池 13.6GB——合计就 70+。**「模型小 ≠ 显存宽裕」是面试反直觉点。**

**Q57. FSDP 是什么？为什么用 FSDP 不用 DDP/DeepSpeed？**
Fully Sharded Data Parallel：参数/梯度/优化器状态分片到所有卡。单卡场景我们的
n_gpus_per_node=1（多卡是 veRL 的池化布局，student 实际单卡训练），FSDP 的价值在
分片 + 我们的 offload 配置。veRL v1 原生走 FSDP 路径。

**Q58. vLLM 的 continuous batching 为什么比静态 batch 快？**
请求完成即插入新请求（iteration-level 调度）vs 等整批完成——消除「短请求等长请求」
的 GPU 空转。我们的 rollout 用它，util 0.25 只留 KV 池（省显存给训练）。

**Q59. 为什么 rollout 和训练共卡要 sleep mode 关掉？**
vLLM 的 sleep 模式用 cumem 分配器（6000D 不支持）且 sleep/wake 有开销。训练步之间
频繁切换不如保持引擎热。我们 +enable_sleep_mode=False。

**Q60. Ray 在你的项目里起什么作用？**
进程编排：agent loop worker 池、reward loop worker、teacher 服务、verifier 单例
actor。Ray actor 的命名机制实现「全局单例」（多 worker 共享一份 verifier 模型）。

**Q61. Ray actor 的并发安全怎么处理？**
check-then-act 竞态：8 个 worker 同时 get_or_create → 多个 create 竞速。用
ActorAlreadyExistsError 捕获后 get_actor 兜底——**先检查再创建的模式必须配异常兜底**。

**Q62. 20ms 攒批窗口怎么选的？**
权衡：窗口越大 batch 越大（吞吐高）但首条请求延迟越大。20ms ≈ 推理延迟的零头
（1.5B batch 推理 ~15-30ms），步耗时 51→19.7s 验证有效。**「攒批窗口 = 容忍的延迟上限」**
这类系统参数要能说出手感。

**Q63. watchdog 误判怎么防？**
① 错误关键词排除 RewardLoopWorker（沙箱失败 traceback 是正常业务噪音）；② 进程消失
后 sleep 60 复查（Ray 重启窗口）；③ 完成/崩溃都落状态文件，事后可审计。

**Q64. checkpoint 为什么 13GB？怎么省？**
save_contents=[model,extra]：fp32 权重 12GB + extra（优化器状态）1GB。
省法：只存 model（省 1GB，代价是不能续训优化器状态）；max_keep=1（但 resume 场景
失效——外挂清理器兜底）；bf16 化（hf 转换时做，评估用 6.2GB）。

**Q65. 断点续训的原理？**
veRL 读 latest_checkpointed_iteration.txt 定位最新 step，加载 model+optimizer 状态
继续。save_freq=32 保证最多丢 32 步（~13 分钟）。余额耗尽强制关机靠它活命了两次。

**Q66. 数据盘 60G 怎么规划 ckpt 的？**
峰值 = 基础 44G + 新旧 ckpt 共存 26G ≈ 70G > 60G → 爆盘事故。解法：扩到 80G +
清理器（3 分钟阈值删旧 step）。**保存是「写新后删旧」，峰值要按 2 份算**。

**Q67. 多进程下载大文件（aria2c）的坑？**
① --file-allocation=none 稀疏文件：失败分片留 NUL 空洞、文件大小却"对"（JSON 解析
在洞处报错）→ 必须 falloc；② 多连接要 sha256 对账（大小一致 ≠ 内容完整）；③ 历史
commit 冷对象回源 60KB/s vs main 热缓存 31MB/s——先 HEAD 对比再决定下载源。

**Q68. 为什么评估要在单卡跑而不是双卡？**
评估（生成+打分）单卡足够，双卡是 ¥10/hr 的浪费；且切卡重启能清掉训练残留进程
（清卡逻辑的另一种实现）。无卡 2GB 内存打分会 OOM（cgroup 限制，free 显示的是宿主
机 1TB——**看 /sys/fs/cgroup/memory.max 才是真相**）。

**Q69. torch.save 写到一半磁盘满的报错长什么样？**
`RuntimeError: unexpected pos 4327871680 vs 4327871576`（zip 容器写尾失败）——不熟悉
会以为是模型 bug，实际是磁盘 100%。**「看到奇怪报错先查资源」**。

**Q70. 训练日志怎么看健康度？**
① Training Progress 步耗时波动（慢批正常，卡死 15 分钟不正常）；② 显存曲线
（75G 压线要警惕，81G 必 OOM）；③ RewardLoopWorker 的 Final Reward 值分布
（全 0 或全 1 是信号失效）；④ 错误关键词（排除正常噪音后）。

**Q71. 为什么 ppo_max_token_len 2560 而不是 2048/4096？**
显存账反推：4096 时 worker 峰值 57.6GB（E2 实测），2560 压激活 ~8GB 留出安全余量；
2048 会截断长样本（2402 token 的样本触发断言）。**配置值要有实测依据**。

**Q72. 你的「清卡逻辑」为什么必要？**
崩溃残留的孤儿 vLLM 引擎占 19.87GB，不清卡下次启动直接 OOM。run 脚本开头
pkill VLLM::Worker + 显存校验（<2000MiB 才继续）。**分布式训练的进程卫生**。

**Q73. 训练进程的 SSH 后台化有什么坑？**
`cd X && nohup Y &` 的解析把复合命令整体后台化，子 shell 持有 SSH 通道 → 本地读超时
（但远端已执行！）——重发命令前必须 pgrep 查远端，否则双进程写同一 checkpoint。
**超时 ≠ 没执行**。

**Q74. 你项目里数据流经过几跳？每跳的字段名是什么？**
agent_loop（extra_fields["verifier_score"]）→ TQ（dict 整存）→ trainer reward 阶段
（pop extra_fields）→ naive.py（non_tensor_batch["verifier_score"]）→ reward_fn
（extra_info["verifier_score"]）→ loss（data["verifier_score"]）。**写 patch 前先读
数据流全程**——链路 bug 就是少画了这张图。

**Q75. 如果 vLLM 和训练争显存怎么办？**
util 调小（KV 池让位）、free_cache_engine=True（步间释放）、分卡（我们的终局方案）。
三者的取舍：util 太小吞吐崩、太大训练 OOM——0.25 是 A800 上的实测平衡点。

## 7.4 论文与方法类（10 题）

**Q76. 你读论文的方法论？**
带着问题读：项目每个阶段的问题 → 定向搜索 → 精读方法段 + 结果表 → 判断「对我的
设置是否成立」→ 写进方案文档。E9 的设计就是三篇论文（SeqBeatsJoint/GLM-5/RG-OPD）
的公式落到同一张流程图。

**Q77. Sequential Beats Joint 的核心主张？对你的实验意味着什么？**
OPD-then-RL 串行 > 所有并行融合；「OPD 是比 SFT 更好的 RL 冷启动」。我们验证了
「OPD 先行」但发现长 RL（446 步）下终点无差异——**论文的步数尺度与我们不同，
这是复现研究的价值：界定结论的适用域**。

**Q78. GLM-5 的在线蒸馏 advantage 公式？**
Â = sg[log π_teacher − log π_student]。与 GRPO 同构（都能进 policy gradient），
group size 可以 =1。我们 E9p3 用 veRL 的 low_var_kl OPD 实现了等价物（use_task_rewards
=False 纯蒸馏模式）。

**Q79. Direct-OPD（2607.05394）对你的警示？**
vanilla OPD 用「弱 post-RL teacher」会把强学生拖垮（56.7→50）；应该迁移 RL-induced
policy shift 而非最终策略。我们的自蒸馏问题与此同源：teacher 无信息增量。

**Q80. 2606.09059 与你的互证是什么意思？**
该论文用 7B 多模态模型测三种 warm-start，结论「Stage-1 控制熵 regime 不控制终点」；
我们独立用 3B 代码模型（E5 蒸馏 vs 基模）得到同结论——**小模型代码域 + 大模型多模态域
的交叉验证**，比单篇论文更硬。

**Q81. 如果面试官问「你做的和 RG-OPD 论文有什么区别」？**
复现 + 边界测试：他们在数学域、teacher 强于 student 的设置下 +2.9；我们在代码域、
teacher=student 的设置下测出失效——**补上了方法的适用边界**。实现上我们多了
gate frac 实时观测和 fallback 容错。

**Q82. 为什么选这几篇论文而不是别的？**
搜索策略：项目问题（信号融合顺序）→ 关键词命中 SeqBeatsJoint（9/3 刚出）→ 顺藤摸瓜
它的引用链（KDRL/TRRD/RLSD）→ GLM-5（用户方案阶段 3 的原型）→ RG-OPD（门控）。
**论文不是装饰，是设计输入。**

**Q83. 你的工作可以怎么包装成「研究贡献」？**
① verifier 奖励注入的代码域验证（GVPO 的 learned 泛化）；② RG-OPD 的自蒸馏边界
条件；③ warm-start 终点的 3B 代码域证据（与 2606.09059 互证）；④ 顺序对称实验
（先蒸后 RL vs 先 RL 后蒸的空白对照）。四条合起来可写一篇 4 页的 workshop 短文。

**Q84. 你觉得这个领域下一个值得做的问题？**
① verifier 奖励的 λ 敏感度（我们只测了 0.1）；② 7B teacher 的门控蒸馏（自蒸馏的
对照）；③ token 级 verifier 塑形（PASS 三规则 + λ-GRPO 频率修正）；④ RL 后熵恢复
（TS-OPSD reheating）——「2ep 峰值」意味着大家都在峰值附近，突破需要新信号而非新顺序。

**Q85. 面试官问「你看过最新的训练配方论文吗」怎么答？**
GLM-5（多阶段 RL + 在线蒸馏恢复）、Nemotron-Cascade 2（多域 OPD 恢复退化）、
TS-OPSD（熵坍缩 reheating）、Metis-RISE（RL 先 SFT 后）——都能说出一句核心 + 与
自己实验的关联。

## 7.5 行为与软技能类（15 题）

**Q86. 你最大的失败是什么？**
信号链路 bug 潜伏两轮实验——但我把它转化成了项目最有价值的部分：完整的事故复盘、
根因分析、修复验证、方法论沉淀（PITFALLS 新增「静默降级」防御）。**失败不可怕，
没有沉淀的失败才可怕。**

**Q87. 预算紧张时你怎么做决策？**
止损思维：阶段 2 无增量 → 阶段 3 是否继续的决策按「预期信息价值 vs 成本」评估；
评估只测 LCB（区分度最高）省一半钱；无卡预演把调试费降到接近零。**研究也要 ROI。**

**Q88. 独立做研究和团队做研究的区别？**
独立 = 全链路责任（数据/训练/评估/预算/文档），逼出方法论（预演/监控/记账）；
代价是没有同行 check——链路 bug 就是例子，如果有 code review 可能更早暴露。
所以我把方法沉淀成可 review 的文档和开源仓库。

**Q89. 你怎么保证结论可信？**
口径冻结（880 题 sha256 对账）、对照单变量、负结果不粉饰（报告里修正 E7 结论）、
论文互证、原始数据保留。**「25.2 是我跑出来的」和「25.2 是可信的」是两件事。**

**Q90. 面试官问「你项目里别人做过的部分有多少」？**
框架和数据集是现成的（veRL/LCB），配方基础来自 Code-R1 论文；我的增量：verifier
全链路（数据/训练/校准/集成）、RG-OPD 复现与边界测试、链路缺陷发现与修复、
顺序对称实验设计、无卡预演方法论。**站在巨人肩上，但做了没人做过的对照。**

**Q91. 你如何保持对最新工作的跟踪？**
项目驱动 + 定向搜索（WebSearch 关键词 → arxiv → 顺引用链），每篇产出中文摘要笔记
（papers/ 目录 7 篇）。不求广度求深度：每篇都能讲出「对我的实验意味着什么」。

**Q92. 这个项目对你的最大成长？**
① 从「跑通」到「跑对」的质变（信号有效性自检意识）；② 在预算硬约束下的工程判断；
③ 负结果的叙事能力——科研的日常是负结果，能从中提取价值才是研究者。

**Q93. 如果入职后发现同事的代码有类似静默 bug，你会怎么做？**
先确认（最小复现）→ 私下拉对齐（不直接公开指责）→ 给出修复 + 回归测试 + 监控防线
（信号 smoke）→ 沉淀进团队 checklist。**我的事故复盘流程可以复用。**

**Q94. 你为什么做这个项目而不是刷题/实习？**
想验证「我能不能独立完成一个有科学产出的完整研究闭环」——从假设到结论全链路。
现在答案是：能，而且找到了自己最擅长的事（发现别人没发现的问题）。

**Q95. 项目里你每天的工作节奏？**
无卡预演（白天）→ GPU 窗口（训练/评估，脚本化+watchdog 自动化）→ 轮询监控
（15 分钟）→ 结果分析/文档。**GPU 是稀缺资源，人的注意力也是——都要调度。**

**Q96. 怎么向非技术面试官解释你的项目？**
「我教一个 3B 的小模型通过'做题-对答案-反思'的循环学会写竞赛代码，还给它配了一个
'判卷老师'（verifier）告诉它哪些答案值得学。最后它超过了官方 7B 模型。过程中我发现
并修好了一个'判卷老师其实没在打分'的系统 bug。」

**Q97. 你对岗位的匹配度怎么自评？**
RL 训练栈全链路（数据/训练/评估/推理部署）、分布式调试（显存/Ray/数据流）、
研究素养（实验设计/论文调研/负结果）、工程纪律（监控/备份/预算）——四件套。

**Q98. 有什么想问面试官的？（反向问题示例）**
「团队现在做 RL 训练时，怎么保证'信号真的在生效'这类静默问题被及早发现？」
——这个问题本身就在展示你的方法论深度。

**Q99. 用三个词总结你自己？**
（按岗位挑）：signal-honest（对信号诚实）、budget-disciplined（预算自律）、
closure-driven（闭环导向）。

**Q100. 最后一句话总结项目？**
「我在 ¥400 预算内独立完成了从假设到结论的完整强化学习研究闭环，其中最值钱的产品
不是 25.2 的模型，而是一套'如何发现并修复静默缺陷'的方法论。」

---

# 第八章 速查卡（面试前一小时过一遍）

## 8.1 核心数字
```
LCB：E1 19.8 | E5 24.7 | E2 21.0 | E7 25.2 | E9p2 25.2 | E9p3 24.8 | 官方7B ~24
HumanEval：E7/E1 86.6 最高 | MBPP：E9p2 78.6 最高
ECE 0.0067 | λ_v=0.1 | β_kl=0.001 | G=4 | lr=1e-6 | batch 8 | 2ep=472步 | 预算 ~¥388
```

## 8.2 核心公式
```
GRPO:  Â_i=(r_i−r̄)/σ;  L=−E[min(rÂ, clip(r)Â)]+β·D_KL
k3:    exp(q−p)−(q−p)−1        (p,q=logprob)
OPD:   loss = k3(π_student ∥ π_teacher) 逐 token
GLM-5: Â = sg[log π_t − log π_s]
Verifier: V_raw=Σk·softmax(logits)[k]/5 → Platt → V_cal
RG-OPD: g = I[(A>0 ∧ L_T>L_S+δ) ∨ (A≤0 ∧ L_T<L_S−δ)]
奖励:  R = result_bin + 0.1·V_cal
```

## 8.3 三个必讲故事
1. **链路 bug**（发现→追踪→修复→验证，Q31 答案）
2. **门控失效的边界条件**（自蒸馏无新知识，Q35 答案）
3. **warm-start 终点无关**（E9p2=E7 + 论文互证，Q36 答案）

## 8.4 一句话结论卡
- 「GRPO 2ep 与 OPD 在难题上打平，都是 ~25 分收敛带」
- 「起点和收尾都不决定终点——配方决定」
- 「verifier 的价值在信号可信度（ECE），不在信号花哨度」
- 「负结果划定边界，边界是下一个研究者最需要的信息」

---

# 第九章 跨项目对比：Search-R1 vs Code-R1（面试官最爱的追问）

> 面试官看到两个 RL 项目必然对比——本章是预设答案。

## 9.1 任务本质差异

| 维度 | Search-R1（agent 搜索）| Code-R1（代码生成）|
|------|------|------|
| 任务结构 | 多步循环（检索→读→推理→再检索，1-20 步）| 单轮生成（一条轨迹）|
| 奖励来源 | 最终答案对错（稀疏）+ 格式 | 单元测试（**完美验证器**，无偏）|
| 核心难点 | 搜索**行为**本身难学 | 难题 credit assignment |
| 侧重点 | 过程结构塑造（LLDS 控制步骤数分布）| 最终执行结果 |

## 9.2 冷启动方案为何不同（高频考点）

- **Search 必须 SFT 蒸馏冷启动**：搜索行为（检索格式/步骤节奏）不在基模的预训练分布里，
  RL 冷启动连格式都凑不出（奖励恒 0 学不动）。蒸馏在这里教的是**新技能**。
- **Code 可以直接 RL**：代码能力在预训练里已有；E9p2 更证明 warm-start 不改变终点。
  蒸馏只是可选初始化。
- **一句话公式**：冷启动的必需性 = 目标行为是否在预训练分布内。Search 不在 → 必蒸；
  Code 在 → 不必。

## 9.3 过程奖励为何一个管用一个不管用（高频考点）

- Search 的过程信号是**行为约束**（格式/步骤规则）——与最终正确率结构性强相关，
  人工规则就能写好。
- Code 的过程信号必须是**正确性判断**——「中间步骤对不对」只有执行才知道，
  手写规则（长度/语法）与过测试相关性弱（E2 21.0 垫底）→ 必须数据驱动（learned verifier）。
- **提炼**：过程奖励的设计第一问——「这个中间信号与最终 ground truth 的结构相关性
  有多强？」强 → 规则可写；弱 → 必须学。

## 9.4 GRPO 稳定性对比（高频考点）

- Search 崩溃根源：**LLDS 长度加权**（按搜索步数分布缩放 loss）的比例 bug——
  轨迹长度方差 1-20 步，长度类加权的数值稳定性极差。
- Code 稳定三要素：① 奖励值域窄（0/1+0.1 塑形，无长度加权）；② 长度截断集中
  （2048+left 截断吃掉长尾）；③ KL β=0.001 + lr 1e-6 双保险。
- **提炼**：轨迹长度方差大的 agent 任务，任何长度相关加权/归一化先做数值稳定性审查。

## 9.5 训练监控与图像分析

指标分层：RL（reward 均值/方差、KL、步耗时、响应长度分布、优势分布）、
蒸馏（distill loss、teacher/student mass、gate frac）、资源（显存/磁盘/进程）。
图像：e1_reward_trajectory.png + training_metrics.json。
判读法则：reward 上升后平台/掉头 = 过训点；长度爆炸 = 黑客；长度坍缩 = 熵死；
训练 reward 与验证分叉 = 过拟合。诚实短板：做到数字轮询、未做到曲线自动告警
（异常检测体系问答里的改进点）。

## 9.6 跨项目一句话总结

「Search 项目教会我**行为注入**（冷启动蒸馏教新技能 + 过程规则约束结构），
Code 项目教会我**信号诚实**（验证器优先 + 数据驱动过程信号 + 静默缺陷检测）。
两个项目合起来覆盖了 RL 后训练的两大范式：行为塑造 vs 能力锐化。」

## 9.7 LLDS vs KL（机制选择追问）

- **LLDS**：把「搜索步数分布」作为训练目标偏移——**给方向**（鼓励多步搜索），
  缺点：长度加权数值敏感（我们崩过）、只控长度不控质量。
- **KL**：策略不偏离 ref——**防崩塌**的保险丝，只防偏不给方向。
- 选择逻辑：奖励稀疏的任务需要行为注入（LLDS 给方向）；奖励可信/行为已存在的任务
  只需要保险丝（KL）。Code 单轮 + 执行奖励 → KL 足够。

## 9.8 GRPO 与长程任务（考量）

组归一化在长轨迹上的短板：长度差异大的轨迹用同一标量奖励 z-score，长轨迹的 token 级
信用分配被稀释；长程 + 稀疏奖励 = credit assignment 灾难，GRPO 不解决（它只省 critic）。
实证：Search 长程 GRPO+LLDS 崩溃；Code 单轮 2048 是「中程」适用。
若再做长程 agent：PPO+GAE（时序信用分配）或密集过程奖励（GLM-5 agentic RL 异步框架）。

## 9.9 数据/测评/沙盒速答

- 训练：code-r1-lc2k（2k 题，唯一训练集）；评估三层难度：evalplus（简单）+
  LCB 880 题（竞赛）+ CodeContests（超难）——难度分层才能暴露「难度依赖」
- verifier 标注：沙箱执行自动标注（k/N），零人工
- 测评口径：LCB 官方 codegen_metrics + evalplus 官方 harness，880 题 sha256 冻结
- 沙盒：seccomp 禁网 + RLIMIT 全家桶 + setuid + killpg 进程组 + 内核兜底

## 9.10 迁移与泛化（诚实边界）

- 可迁移：方法论层（信号自检/预演/三层异常检测/负结果叙事）零成本复用
- 需重调：配方（λ/步数/冷启动）——每任务重做「信号与 ground truth 结构相关性」分析
- 泛化证据有限：E1 的 HC 86.6 → LCB 19.8 证明简单题分数不泛化到难题；
  结论跨域互证靠论文 2606.09059；真实世界代码泛化未测——**知道边界是加分项**

## 9.11 小模型蒸馏五问（本项目能说明的）

① 蒸馏上限 = teacher 上限且有容量压缩损耗（E5：3B 学 7B → 24.7 < teacher ~30+）
② 小模型能力上限可被 RL 突破（GRPO 25.2 > 蒸馏 24.7——「学到的」不如「搜出来的」）
③ teacher 必须强于 student（E9p3 自蒸馏无增量——RG-OPD 隐含前提）
④ 弱 teacher 拖垮强 student（Direct-OPD 56.7→50 + 我们自蒸馏同源证据）
⑤ 反面：小模型做 verifier 够用（1.5B → ECE 0.0067，任务简单则小模型足够）

## 9.12 GRPO+OPD 能否结合（技术可行性 vs 文献 vs 数据）

- 技术：veRL `use_task_rewards=True` = policy_loss + coef×distill_loss，官方一键
- 文献：SeqBeatsJoint——并行融合劣于串行，teacher 项可能翻转 advantage 符号（KDRL 病）
- 我们数据预测：串行两方向都无终点增量 → 并行预期也无增量
- 结论：能做，适合当 ablation（证明「串行优于并行」在我们场景也成立），别指望 SOTA

## 9.13 下一步改进优先级（信息价值递减原则）

1. 阶段 A：GRPO+Verifier 奖励修复版（¥45，唯一未验证正增量候选）
2. 7B teacher 门控蒸馏（¥20，验证边界条件结论）
3. λ/G 超参扫描（¥60，配方微调）
4. 4ep 退化机制研究（¥50，把「2ep 峰值」从观察升级为机制）
5. Direct-OPD policy-shift 迁移（¥20，理论上新信息源）
原则：新信号 > 新对照 > 超参扫描。

## 9.14 扩展问答：升级成多模态 coding agent 怎么做（Q101）

**问：「如果让你把这个项目升级成多模态 coding agent，怎么规划？」**

答（四步结构）：
1. **先厘清定义**：「多模态」分两层——输入多模态（题含截图/图表/UI 设计稿）和
   agent 多轮（写码→执行→看报错→修，SWE-bench 类）。先问清对方指哪个。
2. **三阶段、每阶段一个变量**：① 多模态单轮（换 Qwen-VL 基模，执行奖励不变，
   隔离「视觉输入」的贡献，~¥60）；② 多轮 agent loop（SWE-bench Lite 环境，
   **GRPO 换 PPO+GAE**——我在 Search-R1 和 Code-R1 学到的长程教训：轨迹长度方差大时
   组归一化被稀释 + LLDS 类长度加权会崩，~¥150）；③ UI agent（截图+点击动作，可选）。
3. **资产映射**：veRL async agent loop / 沙箱 / 监控三件套全复用；verifier 迁移为
   「patch 通过测试比例」预测（校准链不变）；**关键判断**：agent 的中间信号（patch 是否
   缩小错误、测试通过数变化）与最终成功**结构性强相关**——所以可能先用规则奖励起步，
   learned verifier 留给弱相关信号（这是 Code-R1「信号相关性第一问」的直接应用）。
4. **最大风险**：长程 credit assignment + 训练稳定性——对策是 PPO+GAE 起步、密集
   过程奖励兜底、轨迹长度硬截断。

**话术收尾**：「每步有对照有止损，不一口吃成胖子——这套方法就是从两个项目里长出来的。」






