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

---

## 8. 沙箱与 harness 带读（执行模型生成代码的安全基建）

### 8.1 为什么需要自建沙箱：firejail 的「静默降级」

```
容器内无 CAP_SYS_ADMIN → firejail 检测到"已存在沙箱" → 静默运行且不设防
```
这是本项目第二个「静默降级」实例（第一个是 verifier 链路）——**安全工具检测到环境
不支持时选择"假装成功"而不是报错**。面试时点出这个 pattern 会非常加分。

### 8.2 sandbox_exec.py 逐段精读（seccomp BPF 手写）

```python
BLOCKED_NRS = [41, 42, ..., 55]   # x86_64 socket 家族 syscall 号
def build_filter():
    f = [_f(0x20, 0, 0, 4),                    # LD W ABS 4：读 arch
         _f(0x15, 1, 0, AUDIT_ARCH_X86_64),    # JEQ arch==x86_64 → skip（跳 1 到 next）
         _f(0x06, 0, 0, SECCOMP_RET_KILL_PROCESS),  # 非 x86_64 → KILL
         _f(0x20, 0, 0, 0)]                    # LD W ABS 0：读 syscall nr
    for nr in BLOCKED_NRS:
        f.append(_f(0x15, 0, 1, nr))           # JEQ nr==blocked → fall through（jf=0）
        f.append(_f(0x06, 0, 0, SECCOMP_RET_ERRNO | 1))  # RET ERRNO(EPERM)
    f.append(_f(0x06, 0, 0, SECCOMP_RET_ALLOW))
```
**带教点**：
1. **seccomp 过滤器是 BPF 字节码**：LD（加载）/JEQ（条件跳）/RET（返回值）三条指令
   构成一个线性程序，内核在每次 syscall 时执行。jt/jf 是「命中跳几格/不中跳几格」。
2. **arch 检查必须第一**：不同架构 syscall 号不同——不检查的话 x86 的 41 号在 arm 上
   是别的调用，堵错系统调用。安全代码的第一个 pattern：**先验身份再验行为**。
3. 选择 **ERRNO(EPERM) 而非 KILL**：让模型代码看到「网络不可用」的正常错误（socket()
   返回 EPERM），而不是进程暴毙——**可诊断性优于暴力**。

```python
def apply_seccomp():
    libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)   # 先 no_new_privs
    libc.prctl(PR_SET_SECCOMP, 2, byref(fprog), 0, 0)  # SECCOMP_MODE_FILTER
```
**为什么 NO_NEW_PRIVS 必须在前**：否则代码可能通过 setuid 二进制提权后绕过 seccomp。
顺序即语义：**先锁提权路径，再上过滤器**。

```python
def apply_rlimits():
    RLIMIT_NPROC 32     # 防 fork 炸弹
    RLIMIT_NOFILE 32    # 防 fd 耗尽
    RLIMIT_FSIZE 2MB    # 防写盘
    RLIMIT_AS 4GB       # 防内存炸弹
    RLIMIT_CPU 30       # 防死循环——内核级 SIGXCPU，孙进程也逃不掉
```
**RLIMIT_CPU 是本设计最硬的一层**：它由内核强制（超时发 SIGXCPU 然后 SIGKILL），
**与进程树无关**——用户态的 timeout+killpg 杀不到的孙进程，内核照样算它们的 CPU 时间。

```python
def drop_privs():
    os.setgroups([])        # 清附加组（很多系统的漏洞点）
    os.setgid(65534); os.setuid(65534)   # nobody
def main():
    apply_rlimits(); apply_seccomp(); drop_privs()
    os.execvp(sys.argv[1], sys.argv[1:])   # 原子替换，无 fork 窗口期
```
**execvp 而非 fork+exec**：沙箱属性（rlimit/seccomp/uid）是进程属性，exec 后保留；
fork 会多一个「未执行代码的窗口」。**一行 execvp 消灭一类 TOCTOU**。

### 8.3 firejail_exec.py（harness 调用方）逐段精读

```python
proc = subprocess.Popen(command, ..., start_new_session=True)   # 新会话 = 新进程组
try:
    stdout, stderr = proc.communicate(input=..., timeout=timeout + 10)
except subprocess.TimeoutExpired:
    os.killpg(proc.pid, signal.SIGKILL)   # 杀整组——孙进程逃逸 bug 的修复
    proc.kill()
```
**这个 bug 的完整故事**（面试必讲）：v1 用 `subprocess.run(timeout)`——它只 kill
直接子进程；模型代码 `os.system()` spawn 的孙进程 hold 住 stdout 管道 →
`communicate()` 永远阻塞 → **训练卡死 54 分钟**（不是崩，是死锁）。修复三件套：
① `start_new_session=True`（子进程自成进程组）② `killpg`（整组杀）③ RLIMIT_CPU
内核兜底（上面 8.2 的那层）。**「用户态超时杀不干净」→「进程组语义」→「内核兜底」
三层递进**。

```python
if len(code) < CLI_ARG_SIZE_LIMIT:   # 3KB
    command.extend(["python3", "-c", code])          # 短代码走 argv
else:
    ... NamedTemporaryFile 写入 → os.chmod(tmp, 0o644) → 执行文件
```
**为什么短代码走 -c**：省文件 I/O；**为什么 chmod 644**：执行时已 setuid nobody，
临时文件必须对 nobody 可读（root 创建的临时文件默认 600）。**降权后每个资源都要
检查访问权限**——这是 drop_privs 场景的第二类经典 bug。

```python
if pytest:   # LCB 风格：测试文件 + solution 文件分离
    os.chmod(tmpdir, 0o777)   # nobody 需要写 pytest cache
    ... python3 -m pytest -p no:cacheprovider -q tmpdir
env["OPENBLAS_NUM_THREADS"] = "1"    # 防 BLAS 多线程抢 CPU
del env["PYTHONPATH"]                # 隔离环境
```
**env 清洗的两个理由**：OPENBLAS 多线程会让 RLIMIT_CPU 的「30 秒」语义混乱
（多核并行消耗 CPU 时间快 8 倍）；PYTHONPATH 污染会让模型代码 import 到训练环境的包。

### 8.4 三个评测 harness 的调用链

| 基准 | harness | 沙箱 | 口径 |
|------|---------|------|------|
| evalplus | 官方 `evalplus.evaluate`（base + plus 双测试集）| 官方自带 | 生成→sanitize→评估 |
| LCB | 官方 `codegen_metrics`（16 进程、timeout 6s）| 官方自带 | 880 题同口径 |
| CodeContests | 自写（`code_prm/exec_backends.exec_code`）| sandbox_exec | public+private+generated 全过才算对 |

**关键坑**（PITFALLS #69）：evalplus 生成后直接评估 0 分——官方流程要求先
**sanitize**（提取代码块、去 markdown 围栏）。评估 harness 的「预处理步骤」和
「打分步骤」分开理解。

### 8.5 沙箱设计的面试总结句

「我的沙箱是**四层纵深**：seccomp 堵网络（可诊断的 EPERM 而非 KILL）、rlimit 限资源
（CPU 由内核强制、逃逸不了）、setuid 降权（nobody + 清附加组）、进程组超时（用户态
杀不干净的内核兜底）。每层都有对应的真实攻击场景，不是我拍脑袋加的。」

---

## 9. 实验回顾与复盘（哪些决定有意义，当时为什么，现在怎么看）

> 完整版见 docs/EXPERIMENT-RETROSPECTIVE.md，这里是面试速记版。

| 决策 | 当时理由 | 现在回看 |
|------|---------|---------|
| 2×2 矩阵 | 控制变量隔离信号类型 | ✅ 最有价值的决定——所有结论都来自「相邻格差一变量」 |
| E1 跑 4ep 944 步 | 对齐论文口径 | ⚠️ 过训退化——但意外产出「2ep 峰值」结论 |
| E5 纯 OPD 而非混合 | 隔离 teacher 信号 | ✅ E9p2 证明起点无关后，这个隔离更有价值 |
| E2 手写 PRM | 补矩阵格 | ✅ 失败但必要——没有它就没有「PRM 必须数据驱动」 |
| 转向 learned verifier | E2 失败后的 pivot | ✅ 方向正确（链路 bug 让验证迟到但基建全对）|
| 评估加 LCB/CodeContests | 难度分层 | ✅ 只用简单题会得出完全错误的结论 |
| 2ep 停早（E7 起）| 省钱先行验证 | ✅✅ 最正确的决定——4ep 是退化区 |
| pin commit 0687ab61 | 担心口径漂移 | ❌ 错误决定（main 早已冻结），学费 = 一天下载折腾 |
| 无卡预演方法论 | E7 七次崩溃的学费 | ✅ 第二项目直接复用 |
| λ_v=0.1 结果主导 | 防黑客的保守默认 | ✅ RWOPD/PASS 论文事后背书 |
| E9 三阶段设计 | 论文驱动（SeqBeatsJoint/GLM-5/RG-OPD）| ✅ 结构正确，负结果也成体系 |
| 监控三件套 | 预算止损需求 | ✅ 崩了 7 次才凑齐，但齐了 |

