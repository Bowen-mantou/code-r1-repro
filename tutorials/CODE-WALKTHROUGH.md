# 代码带教（CODE-WALKTHROUGH）

> 逐文件、逐段精读项目关键代码。每个片段讲「它在做什么、为什么这样写、bug 长什么样」。
> 目标：面试被问「这段代码你怎么写的」时能讲到行级细节。

---

## 1. run_e7.sh 逐键带读（训练配置的灵魂）

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.6
```
**碎片抑制**：max_split_size 限制大块 split（减少碎片化）、GC 阈值 0.6（显存紧张时早回收）。
PRO6000 实测 worker 48→42GB。**坑**：双 vLLM 共卡场景勿加 expandable_segments
（缓存池膨胀撑爆卡）。

```bash
export LD_LIBRARY_PATH=.../nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
```
vLLM EngineCore 子进程**清除环境变量** → 找不到 libnvrtc.so.13。环境级修复：
ld.so.conf.d + ldconfig 系统级写入（比 env 更可靠，因为子进程清 env 也绕不过 ldconfig）。

```bash
data.truncation=left
```
超长 prompt 左截断：**保留尾部**。为什么：prompt 是题目（信息在尾部：完整题目文本），
左截断丢前缀（system prompt 部分）伤害最小。反向（right 截断）会砍掉题目结尾。

```bash
+reward.custom_reward_function.reward_kwargs.use_verifier=True
```
`+` 前缀 = hydra 的「追加键」语法（config 树里没有的键，struct 模式必须 +）。
普通键 = 覆盖默认值。**坑**：把无关键挂靠现有命名空间（如 distillation 键）会干扰
配置树行为——PITFALLS #3。

```bash
actor_rollout_ref.actor.use_kl_loss=True / kl_loss_coef=0.001 / kl_loss_type=low_var_kl
```
三键联动：开 KL 损失、权重 β=0.001、估计器 k3。面试追问「0.001 怎么来的」：
DeepSeek-R1 同款量级（1e-3），大一个数量级锁死探索、小一个数量级防不住漂移。

```bash
actor_rollout_ref.actor.checkpoint.save_contents=[model,extra]
```
model = fp32 权重（12GB）；extra = 优化器状态（~1GB）。**只存 model 省 1GB 但失去
续训优化器状态**。max_actor_ckpt_to_keep=1 配合——但 resume 失效（外挂清理器）。

```bash
# 启动前强制清卡
pkill -9 -f 'VLLM::Worke[r]' 2>/dev/null || true
pkill -9 -f 'main_pp[o]' 2>/dev/null || true
sleep 5
CLEAN_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
echo "启动前显存: ${CLEAN_MEM} MiB（应 <2000）"
```
**进程卫生三件套**：pkill 正则字符类防自杀（`[r]`/`[o]` 让模式不匹配自身命令行）、
`|| true` 容忍「没进程可杀」的非零退出、显存断言（>2000MiB 说明有孤儿，人工介入）。

---

## 2. reward_fn.py 精读

```python
def compute_score(data_source, solution_str, ground_truth, extra_info,
                  use_prm=False, use_verifier=False, lambda_v=0.1):
    # 1. 沙箱执行 → result_bin ∈ {0,1}
    result_bin = 1.0 if execute_passes_tests(solution_str, ground_truth) else 0.0

    # 2. verifier 塑形（E7/E9p2 的设计意图）
    if use_verifier:
        v = extra_info.get("verifier_score")
        if v is None:
            v = extra_info["extra_fields"].get("verifier_score")  # 嵌套兜底
        return result_bin + lambda_v * v        # 结果主导 + 微塑形

    return result_bin                            # 无 verifier 的纯执行奖励
```

**带教三个点**：
1. **双层读取**（顶层 + extra_fields 嵌套）——防御性读取的代价是：**读不到时静默
   fallback**，这正是链路 bug 的土壤。更好的写法：`if v is None: raise` 或启动 smoke。
2. **λ_v=0.1 的语义**：满分 1.1（含 format 0.1），verifier 最多贡献 0.1×1.0=0.1——
   永远低于「做对一道题」的 1.0。**奖励结构上杜绝黑客**。
3. 签名是 veRL naive manager 的约定：`(data_source, solution_str, ground_truth,
   extra_info)`——per-sample 调用，不是 DataProto（v1 manager 的签名陷阱，PITFALLS #59）。

## 3. verifier_server.py 精读

```python
class VerifierInference:
    def score_batch(self, prompts, codes):
        texts = self.build_inputs(prompts, codes)
        enc = self.tok(texts, truncation=True, max_length=4096, padding=True)
        out = self.model(**enc, use_cache=False)
        last_pos = enc["attention_mask"].sum(dim=1) - 1
        logits = out.logits[torch.arange(len(texts)), last_pos]
        digit_logits = logits[:, self.digit_ids]        # (B,6) 只取 "0".."5"
        probs = torch.softmax(digit_logits.float(), dim=-1)
        e_k = (probs * torch.arange(6, device=self.device)).sum(dim=-1)
        v_raw = (e_k / 5.0).cpu().numpy()
        # Platt: logit(V) → sigmoid(a·lv+b)
        ...
```

**带教点**：
1. **取「最后一个有效 token」的 logits**：`attention_mask.sum(-1) - 1`——padding 场景
   下 naive 的 `logits[:,-1]` 会取到 pad token 的垃圾输出。**这是 batch 推理的经典 bug**。
2. **6 分类只在 6 个 token 上 softmax**：`digit_logits = logits[:, digit_ids]` 而非全词表
   再取——省 15 万维的 softmax 开销，且数值更稳。
3. **use_cache=False**：单步推理不需要 KV cache（我们只取最后位置，不生成）。
4. **期望值而非 argmax**：E[k]/5 是「软」分数（0.73 比 1.0 携带更多信息），
   校准后作为塑形信号比硬 0/1 更有梯度信息。

```python
class _VerifierActor:
    def __init__(self, model_path, calibration_path):
        self.verifier = VerifierInference(model_path, calibration_path)
        self._q = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()
    def _worker(self):
        while True:
            items = [self._q.get()]                     # 阻塞等第一条
            deadline = time.time() + 0.02               # 20ms 窗口
            while time.time() < deadline:
                try: items.append(self._q.get(timeout=deadline - time.time()))
                except queue.Empty: break
            vs = self.verifier.score_batch(...)
            for item, v in zip(items, vs): item[0].put(v)
    def score_batch(self, prompts, codes):
        fut = queue.Queue(maxsize=1)                    # per-sample future
        self._q.put((fut, prompts[0], codes[0]))
        return [fut.get()]                              # 阻塞等回传
```

**带教点**：
1. **future 模式**：调用方同步阻塞（`fut.get()`），内部异步攒批——**对外 API 不变，
   对内吞吐提升**。这是「接口与实现分离」在并发场景的教科书用法。
2. **daemon 线程**：actor 销毁时 worker 线程随进程死——不阻塞 Ray 回收。
3. **阻塞式攒批**：第一条 `get()` 阻塞等（省 busy-wait CPU），窗口内非阻塞抢收。
   窗口语义 = 延迟上限（第一条最多等 20ms+推理时间）。
4. 这个 40 行的类把步耗时从 51s 降到 19.7s——**「会攒批」是 RL 工程的核心技能**。

## 4. patch 系列精读（链路修复三件套）

**patch_verifier_topkey.py（根因修复，4 行）**：
```python
# TQ put 前，field 构造处：
_vsf = field["extra_fields"].get("verifier_score")
if _vsf is not None:
    field["verifier_score"] = float(_vsf)
```
为什么在这里修：TQ put 是「dict 键变成独立列」的唯一时机——put 之后 dict 被
NonTensorStack 封装，下游只能按整体读。**修 bug 要修在数据形态变化点**。

**patch_rg_opd_v2.py（NonTensorStack 转换，1 个关键教训）**：
```python
try: vs_l = vs.tolist() if hasattr(vs, "tolist") else list(vs)
except Exception: vs_l = None
```
v1 的 `torch.as_tensor(vs)` 对 NonTensorStack 产生**空张量 (B,0)**——报错
`size of tensor a (0) must match (10)`。教训：**容器类型转换前先确认源类型的方法**。
`.tolist()` 是 tensordict 容器与 numpy 的通用约定。

**patch_rg_opd_v3.py（观测设计，2 行）**：
```python
gate_frac = (gate > 0).float().mean().item()
print(f"[rg-opd] gate frac: {gate_frac:.3f} (bsz={gate.shape[0]})", flush=True)
```
为什么加：**「门控是否生效」需要一个可观测的代理指标**——gate frac < 1 且 > 0 说明
过滤在发生；恒 1 说明门控空转；恒 0 说明信号缺失。**科学实验里「能观测」和「能生效」
一样重要**。

## 5. gen_lcb_fast.py 算法精读

```python
def schedule(items, est, default=1024):
    scored = sorted(items, key=lambda x: est.get(x[0], default), reverse=True)
    batches = []
    for item in scored:                      # 贪心装箱（first-fit decreasing）
        e = est.get(item[0], default)
        placed = False
        for batch in batches:                # 找能容纳的批
            be = est.get(batch[0][0], default)
            if len(batch) < BATCH_MAX and max(be, e) <= min(be, e) * RATIO:
                batch.append(item); placed = True; break
        if not placed: batches.append([item])
    return batches
```
**带教点**：
1. **FFD 贪心**：先按长度降序（大件先装）再 first-fit——经典装箱启发式。
2. **RATIO=1.5 的语义**：批内最长/最短 ≤1.5——控制「短题等长题」的木桶浪费上限
   （最长题决定整批时长，ratio 1.5 = 最坏浪费 33% padding）。
3. **default 先验**：无实测长度的题用阶段 1 的中位数——比固定 1024 好的原因：
   中位数是「分布的最稳健单点估计」。

```python
def gen_batch(model, tok, batch, est, max_new):
    ...
    for j, out in enumerate(outs):
        plen = inputs["attention_mask"][j].sum().item()
        gen = out[plen:]
        hit_cap = gen.shape[0] >= max_new        # 撞顶 = 可能截断
        results.append((tok.decode(gen, ...), hit_cap))
```
**带教点**：`hit_cap` 检测是「口径保证」的关键——撞顶的题单独 2048 重跑，
**保证与串行评估逐 token 一致**（E9p2 vs E7 的 25.2=25.2 才有可比性）。
优化可以打折扣，口径不能。

**最终教训（面试必讲）**：这个方案模拟验证全绿（111 批完美装箱），实际打平串行
（744/800 撞顶重跑）——**模拟验证了调度正确性，没验证真实长度分布**。真实中位数
128 token 让动态上限只有 256，几乎全撞。**任何时间优化先跑 20 题实测**。

## 6. bash 工具链精读

**watchdog 的判断逻辑**：
```bash
if ! pgrep -f 'main_pp[o]' >/dev/null 2>&1; then
  sleep 60                                   # 允许 Ray 重启窗口
  if ! pgrep -f 'main_pp[o]' >/dev/null 2>&1; then
    if grep -E 'RayTaskError|OOM|AssertionError' "$LOG" | grep -v RewardLoopWorker; then
      # 崩溃路径 → 状态落盘 → 关机
    else  # 正常完成 → 关机
    fi
```
**带教点**：双重确认（60s 窗口）防误判；错误关键词**排除正常噪音**（沙箱失败
traceback 天天有）；崩溃和完成都关机（GPU 烧钱），但状态文件区分原因供事后审计。

**清理器**：
```bash
find "$CKPT_DIR" -maxdepth 1 -name 'global_step_*' -mmin +3 \
  ! -name "global_step_${latest}" -exec rm -rf {} +
```
**带教点**：`-mmin +3`（只删 3 分钟前写完的）防误删正在写的 step；
`! -name latest` 保命。时序账：保存间隔 13 分钟 > 清理周期 10 分钟 > 阈值 3 分钟——
**三个时间常数的关系要能口算**（面试追问「为什么是 3 分钟」的答案）。

---

## 7. 代码能力自查清单（面试前过一遍）

- [ ] run_e7.sh 每个键的「为什么」能说一句
- [ ] 数据流五跳的键名能默画
- [ ] verifier 攒批的 future 模式能白板画出
- [ ] gate 公式能写出并解释每个符号
- [ ] hit_cap 重跑为什么保证口径一致
- [ ] watchdog 三个时间常数（60s/30s/120s）各管什么
- [ ] patch 三版的迭代逻辑（v1 崩 → v2 修 → v3 观测）
- [ ] 能说出「如果重写，reward_fn 的 fallback 会改成 raise + smoke」
