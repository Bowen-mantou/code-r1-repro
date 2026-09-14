# veRL 框架导读（06）

> 以 v1 trainer（本项目所用）为主线：架构、角色、数据流、配置体系。
> 目标：面试被问「veRL 内部怎么工作的」时能画图 + 讲数据流。

---

## 一、一张图：一个训练步的生命周期

```
python3 -m verl.trainer.main_ppo（hydra 解析配置 → Ray 初始化）
  └─ TaskRunnerV1.run()
       ├─ 建角色（ray worker group）：
       │   ActorRollout（student：FSDP 权重 + vLLM 引擎同进程）
       │   RefPolicy（ref：KL 参考，param_offload）
       │   RewardModel 角色（沙箱执行 + reward_fn）
       │   AgentLoopWorkerTQ ×8（并发生成，带 verifier 引用）
       │   [OPD 时] TeacherModel（7B vLLM 独立资源池）
       └─ fit() 每步循环：
           1. agent loop 并发 rollout（n=4 轨迹/题）
           2. 轨迹内：沙箱 → result；verifier actor → V_cal
           3. 入 TransferQueue（异步存储）
           4. trainer 取 batch → reward 阶段 → 展平
           5. micro-batch FSDP 前向/反向（+KL/+蒸馏 loss）
           6. optimizer step → save_freq 存 ckpt
```

## 二、角色与资源池（双卡布局的原理）

veRL 用**角色（Role）→ 资源池（ResourcePool）**的两层抽象：
- 每个 Role 声明自己的 GPU 需求；多个 Role 可以共池（共卡）或分池（分卡）
- 我们的布局：
  - ActorRollout + RefPolicy + Reward + AgentLoop → 卡 0（global_pool）
  - TeacherModel → 卡 1（teacher_pool，distillation.n_gpus_per_node=1）
  - verifier actor → 卡 1（Ray runtime_env CUDA_VISIBLE_DEVICES=1 手动指定）

**单卡 OPD 的 patch 就是把 TeacherModel 强行映射进 global_pool 共卡**——
然后被显存账判了死刑（24GB 不透明开销）。

## 三、数据流（本项目 bug 的战场，逐跳背熟）

```
agent_loop_tq.py:205
  final_output.extra_fields["verifier_score"] = V_cal
        ↓
list_of_dict_to_tensordict(fields)  →  TQ 存储
  extra_fields 是 dict → NonTensorStack（不展平！bug 根源）
        ↓
trainer_base.py（reward 阶段）
  data.pop("extra_fields")  →  reward_extra_infos_dict
        ↓
naive.py run_single（per-sample）
  non_tensor_batch.get("verifier_score")  →  None（键不存在）
        ↓
reward_fn.py
  v = extra_info.get(...)  →  None  →  fallback 纯 result_bin（静默降级）
        ↓
loss（蒸馏）
  data.get("verifier_score")  →  修复后：顶层字段 → gate 生效
```

**修复位置的选择**：TQ put 之前提为顶层字段——那是「dict 键 → 独立列」的唯一
信息完整时刻。

## 四、配置体系（hydra + 我们踩过的坑）

- 配置树：`actor_rollout_ref.actor.*` 等分层；hydra 的 `+` 前缀 = 追加无关键
- `struct 模式`：未声明键必须 `+`，否则启动报错
- **坑 1**：把无关键挂靠现有命名空间（+distillation.verifier_path）会改变配置树行为
- **坑 2**：dataclass 是 frozen 的——运行时改配置要用构造参数而非属性赋值
- **坑 3**：sed 改配置脚本时跨行字符串匹配不上（路径分行的脚本必须整段替换）

## 五、三个关键设计模式（veRL 教你的分布式套路）

1. **Ray 命名单例**（verifier）：check-then-act 竞态 → ActorAlreadyExistsError 兜底
2. **攒批窗口**（20ms）：per-sample 调用合并 batch——「窗口 = 可容忍延迟上限」
3. **生产者-消费者解耦**（TQ）：rollout 与训练异步化；我们 sync 模式下 TQ 只当
   结构化存储

## 六、面试速答模板

「veRL v1 我讲三个东西：**角色-资源池抽象**（双卡布局就是这么排的）、**agent loop
数据流**（每个字段从写到读经过五跳，我就是在这里抓到链路 bug 的）、**配置体系**
（hydra struct 模式与 + 前缀的坑）。深度调试能力就建立在这三张图上。」
