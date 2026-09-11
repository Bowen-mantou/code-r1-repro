# 工程深度解析（ENGINEERING-DEEP-DIVE）

> 带教级：veRL 训练栈的每个工程决策——「为什么这么设计、怎么验证、出问题怎么查」。
> 面向面试的工程深度问答 + 上手第二项目（agentic-grpo）的迁移手册。

---

## 一、veRL v1 架构全景（先有地图再走路）

### 1.1 角色与进程拓扑（一个训练步的生命周期）

```
启动（python3 -m verl.trainer.main_ppo）
  ├─ hydra 解析配置 → omega_conf_to_dataclass → 各 Config 对象
  ├─ Ray 初始化（本机 head，无集群）
  └─ TaskRunnerV1.run()
       ├─ 建角色池（ray worker group）：
       │   ├─ Role.ActorRollout（student：训练权重 + vLLM 引擎）
       │   ├─ Role.RefPolicy（ref：KL 用，param_offload）
       │   ├─ Role.RewardModel（沙箱执行 + reward_fn）
       │   ├─ Role.AgentLoop（AgentLoopWorkerTQ ×8，带 verifier 引用）
       │   └─ distillation.enabled 时：Role.TeacherModel（7B vLLM）
       └─ fit() 循环（每步）：
           1. AgentLoopWorkerTQ 并发生成 n=4 轨迹（rollout）
           2. 每条轨迹：沙箱执行 → result_bin；verifier actor → V_cal
           3. 轨迹入 TransferQueue（TQ）
           4. trainer 从 TQ 取 batch → reward 阶段（naive.py 逐样本算 R）
           5. 展平 → micro-batch → FSDP 前向/反向（含 KL loss / 蒸馏 loss）
           6. optimizer step → save_freq=32 存 ckpt
```

**面试要点**：能画出这张图 + 说出每一步在哪个进程、哪个卡。

### 1.2 三个关键设计模式（veRL 教你的分布式套路）

**① Ray 命名单例**（verifier actor）：
```
try: return ray.get_actor("name")
except ValueError: create → 竞态 → ActorAlreadyExistsError 兜底 get_actor
```
适用场景：多调用方共享一份重资源。面试扩展：Ray 的 detached actor / placement group。

**② 攒批窗口**（20ms）：per-sample 调用合并 batch——「吞吐 vs 延迟」的系统权衡。
设计心法：窗口大小 = 可容忍延迟上限；验证靠实测（51s→19.7s）。

**③ 异步生产者-消费者**（TransferQueue）：rollout（GPU 密集）与训练（GPU 密集）
解耦——veRL 的 TQ 就是给 off-policy 容忍度/多步并行用的。我们 `sync=True` 同步模式，
TQ 只当结构化存储。

---

## 二、显存账的完整算法（面试必考实操）

### 2.1 训练侧（卡 0，student FSDP）

```
基础项：
  W：权重 bf16          6.2GB
  G：梯度 bf16          6.2GB
  O：Adam 状态 fp32×2  12.4GB  （m 和 v 各 4 字节/参数 × 3B）
  小计（参数系）      ~25GB

动态项（batch 相关）：
  A：激活（无 checkpointing）  = batch_tokens × hidden × layers × 常数
  A'：激活（gradient checkpointing 后）≈ 前者的 ~30-40%（重算代价 = +30% 计算）
  K：vLLM KV 池（util 0.25 × 80G = 20G 预留，实际 ~13.6G）

实测：worker 峰值 58.6GB（含 ref 换入和 KL 前向）
```

### 2.2 为什么 58.6GB 不是 25+13.6=38.6GB？

中间 20GB 来自：micro-batch 前向激活（即使 checkpointing，边界层激活还在）、
FSDP all-gather 的权重副本、通信 buffer、vLLM 的调度缓冲。
**教训（PITFALLS #6）**：显存账不能用「权重+KV」推，必须引用
`perf/max_memory_allocated_gb` 实测。我们的账本错误（44GB 估 vs 57.6GB 实测）
直接导致过 E2 一次 OOM。

### 2.3 压显存的武器谱（按「代价从小到大」排序）

| 手段 | 效果 | 代价 | 我们用了吗 |
|------|------|------|-----------|
| gradient checkpointing | 激活 -60% | 计算 +30% | ✅ |
| ref param_offload | ref 权重出卡 | 步耗时 + | ✅ |
| ppo_max_token_len 2560 | 激活线性降 | 长样本截断 | ✅ |
| forward_prefetch=False | -2GB | 无（仅省 prefetch buffer）| ✅ |
| optimizer_offload | Adam 状态出卡 | 步耗时 ++ | ❌（预留）|
| micro_batch 减半 | 激活减半 | 步耗时 ↑（更多 micro 步）| 备用 |
| KL 关掉 | -ref 全部开销 | 策略稳定性风险 | ❌ 不砍 |

**面试句式**：「我压显存有顺序——先动零代价的（prefetch），再动换时间的（offload），
最后才牺牲吞吐（micro_batch）。每个手段的代价都说得出。」

---

## 三、FSDP 机制速通（面试理论）

- **分片什么**：参数、梯度、优化器状态按 rank 分片（ZeRO-3）
- **前向**：需要全参数 → all-gather → 用后释放；**反向**：再 all-gather 算梯度 → reduce-scatter
- **我们的场景**：n_gpus_per_node=1 时 FSDP 退化为「单卡 + offload 开关」——分片
  无意义，但 offload/checkpointing 机制保留
- **为什么不用 ZeRO-2**：veRL 集成的是 FSDP 路径，单卡场景差异不大

## 四、vLLM 共卡的工程细节

- `gpu_memory_utilization=0.25`：只留 20G KV 池——训练占大头，rollout 够用
  （batch 8×4 轨迹 × 2048 token 的 KV 需求 ~13.6G）
- `free_cache_engine=True`：训练步间释放 vLLM 缓存（归还显存给训练）
- `enable_sleep_mode=False`：sleep/wake 的 cumem 分配器在 6000D 不支持；
  A800 上热保持开销小，频繁 wake 反而贵
- **争卡协调**：veRL 在训练与 rollout 之间串行调度（同一张卡时间片轮转），
  显存共享靠「训练时 vLLM 释放、rollout 时训练释放」的节奏

## 五、checkpoint 与断点续训机制

- `save_contents=[model,extra]`：fp32 权重（12GB）+ 优化器状态（~1GB）
- `save_freq=32`：每 32 步（~13 分钟）存一次——最多丢 13 分钟
- `latest_checkpointed_iteration.txt`：resume 的指针
- **save 的时序陷阱**：veRL「写新 → 更新 iteration → 删旧」（max_keep=1）——写新时
  旧份还在 → 磁盘峰值 = 2 份。且 resume 场景 max_keep 会漏删旧份（我们实测失效）→
  外挂清理器（mmin +3 删非 latest）
- **续训语义**：加载 model+optimizer 继续——lr 调度、epoch 计数从 ckpt 恢复

## 六、数据流细节（bug 高发区，逐跳画图）

```
[agent_loop_tq.py:205]
final_output.extra_fields["verifier_score"] = float(_vs[0])
        ↓（AgentLoopOutput 打包，TQ put）
[list_of_dict_to_tensordict]
key="extra_fields" → NonTensorStack（每元素 dict）   ← 不展平！bug 根源
        ↓（kv_batch_get 取出，构造 DataProto）
[trainer_base.py reward 阶段]
data.pop("extra_fields") → reward_extra_infos_dict     ← 从主数据流摘除
        ↓
[naive.py run_single]
non_tensor_batch.get("verifier_score") → None → extra_info 无此键
        ↓
[reward_fn.py]
v = extra_info.get("verifier_score") → None → fallback result_bin   ← 静默降级点
```

**带教结论**：写任何跨跳字段传递前，先画这张图；每跳的键名和容器类型都要标注。
我们的修复（顶层提取）改在 TQ put **之前**——因为那是「dict 键转顶层」的唯一
有完整信息的位置。

## 七、监控与告警体系（完整版，含已知短板）

| 层 | 机制 | 已实现 | 短板 |
|----|------|--------|------|
| 进程 | watchdog 关键词 + 关机 | ✅ | 依赖人工 grep 模式维护 |
| 资源 | 显存/磁盘轮询 + 清理器 | ✅ | 无自动扩容 |
| 结果 | sha256/dry-run/smoke | ✅ | 仅启动时 |
| **信号** | reward 成分分解、信号 smoke | ❌ | 链路 bug 的教训（已入 PITFALLS）|
| **质量** | reward/长度/熵曲线告警 | ❌ | E1 4ep 退化的教训 |

## 八、无卡预演的操作清单（第二项目直接复用）

1. 配置 dry-run：hydra 解析跑通（键名合法性）
2. 数据校验：sha256 + 行数 + JSON 严格解析
3. patch 无卡测试：mock 数据结构跑关键函数（参考 test_rg_opd_gate.py 的 monkeypatch 法）
4. 数据流静态审查：逐跳画字段图
5. 资源账：显存/磁盘按「实测 + 峰值 2 份」算
6. 传输方案：大文件 aria2c 或平台迁移（先测 SFTP put 可用性）

## 九、bash/SSH 工程技巧（我们的血泪清单）

- `pkill -f 'xxx'` 自杀 → 正则 `xxx[y]` 字符类
- `cd X && nohup Y &` 整体后台化 → SSH 通道挂起 → 用 `;` 分隔 + `< /dev/null`
- SSH 超时 ≠ 命令没执行 → 重发前先 pgrep
- 远程长 sleep 被网关 120s 断 → 本地 sleep 再查
- Windows Git Bash 路径转换 → `MSYS_NO_PATHCONV=1` 或 `//root` 前缀
- CRLF 毁 shell 脚本 → 用 Write 工具写（LF），不用本地 python 写
- 后台任务 `< /dev/null` 缺了 → SSH 通道不 EOF → read 阻塞

---

## 十、迁移清单（第二项目 agentic-grpo 直接复用）

- [ ] 环境红线：不装 flashinfer；装包后验证 torch 版本
- [ ] 清卡逻辑进 run 脚本（pkill + 显存校验）
- [ ] watchdog + 清理器 + 断点续训三件套
- [ ] 信号 smoke（每个新信号启动时打印实际值）
- [ ] 无卡预演 6 项清单
- [ ] 每实验产物立即本地备份
- [ ] 换实例后重查全部资源
