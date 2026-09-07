# 03-code-r1 踩坑手册（E1 全程实战总结）

> **后续每个对话开始前必读**。每一条都是 E1 真实踩过、修过、验证过的。
> 适用：6000D-84G 实例 + veRL main + CUDA 13 全栈。2026-09-02 定稿。

---

## 一、AutoDL 平台

| 坑 | 正确做法 |
|----|---------|
| 容器内 `/usr/sbin/shutdown` 无效（是 systemd 符号链接，容器无 systemd） | 用 **`/usr/bin/shutdown`**（AutoDL 自己的脚本：清 Trash → 杀 supervisord → 触发实例关机） |
| AutoDL API 关机（备用） | base=`https://www.autodl.art`、路径 `/api/v1/adl_dev/dev/instance/pro/power_off`、header `Authorization: <token>` **不带 Bearer**、body `{"instance_uuid": ...}`；token 在控制台「设置→开发者 Token」 |
| 无卡模式 | ¥0.1/hr 适合装环境/下模型，但 **容器内存限 2GB**（torch.load 大模型必死）、CPU 1 核。**free -m 显示的是宿主机内存（1TB）不可信**，容器真实限制看 `/sys/fs/cgroup/memory.max`（2026-09-06 打分 OOM 三次才定位）|
| 无卡 ↔ GPU 切换 | 必须关机重启，**会杀掉所有进程**（切换前保存进度） |
| 余额耗尽 | 强制关机（无预警）——靠 save_freq 小步保存保命；充值要留足余量 |
| 训练完自动关机 | watchdog 检测进程消失 → 微信通知 → sleep 120 → `/usr/bin/shutdown` |

## 二、SSH 驱动与文件上传（run_ssh.py 的教训）

| 坑 | 正确做法 |
|----|---------|
| AutoDL 网关 SFTP 不可用、stdin 传输被断、单条命令 >100-200KB 被拒 | 用 **双层 base64 命令块上传**（内层 chunk 50KB → 外层 base64 免疫引号 → `echo X \| base64 -d \| base64 -d >> file`）；md5 校验 |
| 本地 Git Bash 路径转换（/root → C:/Program Files/Git/root） | 前缀 **`MSYS_NO_PATHCONV=1`** |
| `pkill -f xxx` 会匹配自身命令行自杀 | 正则技巧 `pkill -f 'main_pp[o]'`（`[o]` 让模式不匹配自身文本） |
| `A && nohup B &` 后台化产生 subshell 持有 SSH 通道 → 读超时 | 后台命令单独一条、`;` 分隔、加 `< /dev/null`；或用 `>` 重定向后立即返回。**超时 ≠ 命令没执行**：远端往往已跑起来，重发启动命令前必须先 pgrep 查远端进程（2026-09-06 双进程写同一 checkpoint 事故：E7 生成两进程并行污染结果，双杀重启白跑 ~10 分钟）|
| 命令里含反引号/引号嵌套 | 代码一律**写成文件上传执行**，别在 SSH 命令里写 python 代码 |
| 输出中文/emoji 在 Windows GBK 控制台报 UnicodeEncodeError | run_ssh.py 已加 `sys.stdout.reconfigure(encoding='utf-8', errors='replace')`（勿删） |

## 三、环境（6000D Blackwell + veRL main 0.10 + CUDA 13）——最重要

| 坑 | 正确做法 |
|----|---------|
| **⚠️ 装 flashinfer 会把 torch 静默降级到 2.10 并覆盖 NCCL cu13**（E1 评估期崩了 1 小时才定位） | **永远不要 pip install flashinfer**；任何 pip 装包后立即验证 `python -c "import torch; print(torch.__version__)"` == `2.11.0+cu130` |
| firejail 在容器内自我降级（"existing sandbox detected"，无 CAP_SYS_ADMIN） | 用自制 **`sandbox_exec.py`**：seccomp BPF 禁网络 + RLIMIT（CPU 30s/NPROC/NOFILE/FSIZE/AS）+ setuid nobody + exec |
| 沙箱超时漏洞：模型代码 spawn 孙进程，subprocess.run(timeout) 只杀直接子进程 → 管道永久 hang → 训练卡死 54 分钟 | firejail_exec.py 已改：`Popen(start_new_session=True)` + 超时 `os.killpg(SIGKILL)`；sandbox_exec 加 `RLIMIT_CPU=30` 内核兜底 |
| vLLM EngineCore 子进程**清除 LD_LIBRARY_PATH** → 找不到 libcudart.so.13 | 写 **`/etc/ld.so.conf.d/cu13.conf`**（内容：cu13/lib 路径）+ `ldconfig`（系统级，所有进程生效）；CUDA_HOME 也指 cu13 |
| veRL 默认 `enable_sleep_mode=True` → vLLM cumem 分配器在 6000D 不支持 | 已改 veRL 源码 `workers/config/rollout.py` 为 False（勿动）+ run 脚本加 `+actor_rollout_ref.rollout.enable_sleep_mode=False`（hydra 键不在 struct 需 + 前缀） |
| `transfer_queue` 依赖没装上（pip install -e 跳过 git 依赖） | 手动 `git clone gh-proxy .../Ascend/TransferQueue.git` + `pip install -e ./TransferQueue` |
| nvidia/cuda_nvrtc 包里是 12.8 版（cu12 包覆盖）→ NVRTC 12.8 不支持 sm_120 | cu13 库在 `nvidia/cu13/lib/`；用 ld.so.conf 方案；flashinfer 的 CUDA 版本检查绕不开（根源是它 cu128 编译）——**不装 flashinfer 就完事** |
| 镜像源 | 阿里 pytorch-wheels 无 cu130 torch → 官方 `download.pytorch.org/whl/cu130`；阿里 PyPI 下大包卡死 → 清华 `pypi.tuna.tsinghua.edu.cn/simple`；GitHub 直连 → `gh-proxy.com/https://github.com/...`；HF → `hf-mirror.com` |
| hf CLI 已改名 | 新版是 `hf download`（huggingface-cli 已废弃）；大文件 hf 下载会卡 → `wget -c --tries=30 --timeout=60 --waitretry=20` + `< /dev/null` |
| **hf-mirror 的 Xet 后端**（2026-09-03 E5 踩） | ① 默认走 Xet：xethub.hf.co 的 cas-server 不被 hf-mirror 代理 → 401 崩 → **必须 `export HF_HUB_DISABLE_XET=1`**；② 禁用后 hf-mirror 的 resolve 仍 302 到 `us.aws.cdn.hf.co/xet-bridge`，**可能返回损坏字节流**（sha256 与 etag 不符、safetensors header 截断）——大文件优先 **ModelScope 镜像**（`modelscope.cn/models/Qwen/.../resolve/master/...`，Qwen 官方维护、国内快）|
| hf CLI 新版无 `-x` | 并发参数是 `--max-workers N`（不是 `-x`） |
| 下载完整性校验 | 下完必须校验：`sha256sum` 对比 `X-Linked-Etag`（curl -sI 拿）+ safetensors header 解析（`safe_open` 只读头不加载张量，无卡模式内存安全）+ index.json `total_size` 对账 |

## 四、训练配置（run_e1.sh 的定稿参数，E5/E2/E6 沿用）

| 坑 | 正确做法 |
|----|---------|
| checkpoint 默认存 optimizer（一份 36GB）→ 数据盘 50G 秒爆 | `save_contents=[model,extra]`（13G/份）+ `max_actor_ckpt_to_keep=1` + `save_freq=32` |
| `ppo_max_token_len_per_gpu=2048` 太小，长样本 2402 tokens 触发断言崩溃 | 用 **4096**（micro 4 × 4096 = 16K tokens 显存安全） |
| 84GB 卡 OOM（vLLM 0.35 + micro 8） | 定稿：`gpu_memory_utilization=0.25` + `micro_batch=4` + `log_prob_micro=2` + 激活 61GB 峰值实测通过 |
| `data.truncation` 默认 error 崩 | `data.truncation=left` |
| hf_model 保存是 fp32 12GB（不是 6.2GB bf16） | 评估用转换脚本转 bf16（convert_ckpt.py） |
| 训练结束 final save 没写完就关机 → ckpt 损坏 | 评估取**最近一个周期 ckpt**（E1 用 928 而非损坏的 944）；watchdog 关机前 sleep 120 留保存时间 |
| 监控误判：reward 计算里模型代码测试失败的 traceback 是**正常输出** | watchdog 错误检测要 `grep -v RewardLoopWorker`，只认 RayTaskError / Error executing job / CUDA out of memory / AssertionError |
| Windows python 重写 shell 脚本产生 CRLF → `set: pipefail: invalid option` | 改 .sh 必须用 Write/Edit 工具（LF），别用本地 python 写 |
| E1 的 reward_fn 签名 | naive manager 逐样本调 `(data_source, solution_str, ground_truth, extra_info)`，**不是** DataProto（e6b 记忆里的 v1 描述不适用于 naive manager） |
| coder1 奖励格式 | 模型输出**必须含 `<think>...</think>` 前缀**（`validate_response_structure` 正则要求），只有 `<answer>` 判 Bad format → -1.1；满分 = 1.1（format 0.1 + answer 1.0），smoke 断言别写成 1.0 |

## 五、评估（evalplus + LCB）

| 坑 | 正确做法 |
|----|---------|
| evalplus 0.3.1 无 CLI（`evalplus.generate` 不存在） | API：`from evalplus.codegen import run_codegen`；backend 用 `"hf"`（vllm 后端撞 EngineCore 环境问题） |
| transformers 5.10：`from_pretrained` 传 `attn_implementation=`（任何值）或字符串 dtype 都崩 "Could not import Qwen2ForCausalLM" | **不传 attn_implementation、dtype 用 `torch.bfloat16` 对象** |
| evalplus evaluate 加载旧缓存结果（0 分缓存） | 评估前 `rm -f *_eval_results.json` |
| 生成后直接评估 0 分 | 官方流程要先 **sanitize**（`evalplus.sanitize.script(f, inplace=True)`）提取代码块 |
| LCB v5 = **5 个 test*.jsonl 的并集**（880 题），不是 test5 一个 | 全部 5 个文件下载（HF resolve URL 直下，datasets 新版不支持脚本数据集）；评估用官方 `codegen_metrics`（自写入口 my_lcb_eval.py 已就绪） |
| LCB 的 prompts 模块 import anthropic 旧常量崩 | 生成脚本自抄 CodeQwen prompt 构造（gen_lcb.py 已就绪，勿改回 import LCB prompts） |
| 静态 batch 生成反而比单题慢（padding 序列也跑满 max_new_tokens） | 新方案 **`batch_gen.py`（通用调度器）**：用上次评估代码重建长度先验 → 长度分组装箱（ratio≤1.5）+ 动态 max_new_tokens（预估×1.5+64）→ 预期 3× 提速。题目固定时（同 benchmark 复评）先验近乎完美。E2/E5/E6 评估用它替换单题生成。**E7 实测（gen_lcb_fast.py 两阶段自举）基本没提速**：LCB 真实答案中位数仅 128 token → 动态上限 256 → 长题大量撞顶串行重跑 → 总 75 分钟 ≈ 串行 80 分钟。**教训：时间预估必须真实小样本实测（跑 20 题测速），模拟装箱正确 ≠ 吞吐快**；且动态上限策略对「短尾长尾分布」无效——先验中位数低的场景别用装箱 |
| HumanEval 官方基线数字 | HumanEval 84.1 / MBPP 73.6（官方 3B，EvalPlus 口径）；OpenCompass 的 45 分是 harness 差异，**不要用** |
| **LCB 数据口径真相**（2026-09-06 E7 评估踩） | main 从 **2025-06-05 起冻结未变**；E1/E2/E5 用的就是 **main 的 test1-5 并集 = 880 题**（gen_lcb.py 只读 5 文件）。「1480 题」= main 另有 test6。**pin release_v5 commit 0687ab61 是错的**（2025-01 老快照，test1-5 仅 584 题）。验证口径：下载后数行数（400+111+101+101+167=880）+ HF API `?blobs=true` 的 LFS sha256 对账 |
| **hf-mirror 历史 commit 冷对象极慢** | `resolve/{commit}` 无 CDN 缓存 → 回源仅 ~60KB/s；`resolve/main` 有热缓存 371KB/s~31MB/s。**先 curl -sIL 对比 main 与目标 commit 的 Content-Length**，相同就用 main 下载 |
| **aria2c 多连接下载**（2026-09-06 实测） | 单连接 371KB/s → `aria2c -x 16 -s 16 -k 1M` 后 4.5~31MB/s（10-80×）。**必带 `--file-allocation=falloc`**：`none` 造稀疏文件，失败分片留下 **NUL(0x00) 空洞**且文件大小仍"对"（JSON 在洞处报 Invalid control character）。**下完必须 sha256sum 对比 HF API `?blobs=true` 的 lfs.sha256**——大小一致 ≠ 内容完整 |
| 评估脚本「坏行跳过」容错是危险的 | gen_lcb.py 的 try/except json.loads + continue 会**静默丢题改变口径**。数据必须校验通过；坏行应报错退出，不能跳过 |
| **LCB 打分在 2GB 容器 OOM**（2026-09-06） | my_lcb_eval.py 一次加载 4.3GB 题目 + 16 进程 → OOM（exit 137 SIGKILL，日志 0 字节）。同进程分批 + gc 也救不回（codegen_metrics 内部对象累积）。**解法：每批独立进程**——split_lcb_shards.py 切 9 片 × 100 题 → my_lcb_eval_worker.py 每片一个 python 进程（进程退出=内存清零）→ run_lcb_score_shards.sh 串行驱动 + 加权合并 pass@1。口径与原脚本一致（shard 0/1 数字与同进程分批完全吻合）。9 片 ~9 分钟 |

## 六、通知与监控

| 坑 | 正确做法 |
|----|---------|
| Server酱：Windows GBK 编码中文标题 → 数据库拒收（SQLSTATE 1366） | 本地 hook 脚本用**英文**标题/正文；远程（Linux）发中文没问题 |
| 通知骚扰用户 | 只发重要事件：训练完成/崩溃/指标告警/止损关机；**恢复不发**（用户明确要求） |
| cost_tracker 是进程态（跨 Bash 调用丢 session） | 手动 append jsonl 记录，或单次调用完成 start+end |
| Claude Code Stop hook（本地微信通知） | `~/.claude/hooks/notify-stop.sh` + settings.json hooks（UserPromptSubmit 记时间 + Stop 调用），60s 阈值防刷屏 |

## 七、单卡 OPD 的坑（E5 调试全程，2026-09-03）

| 坑 | 正确做法 |
|----|---------|
| veRL OPD 默认 teacher 独立 GPU 池（student+teacher 各 1 卡）| 单卡需 patch：v0 `main_ppo_v0.py` / v1 `trainer/ppo/v1/trainer_base.py` 把 `Role.TeacherModel` 映射到 `"global_pool"` 且不建独立 teacher_pool spec（分数 GPU 共卡，`num_gpus=1/max_colocate_count`） |
| **TransferQueue `_pack_field_values` 无条件把每样本张量组装成 jagged nested**（simple_storage_manager.py:370）| 规整张量也会变 nested（offsets 等差全宽）——下游断言按"真实序列长"对比必炸。修复：agent_loop 侧 pad 到固定宽（patch1）+ 训练入口 `to_padded_tensor` → 按 `input_ids.offsets()` 取前 len[i] 行重建 nested（patch2，`patch_tq_ragged_v2.py`） |
| v0 trainer 的 mini batch 语义 | v0 的 `mini_batch = ppo_mini_batch_size × n`（v1 是纯样本数）；batch 32 轨迹 → v0 要设 ppo_mini_batch_size=8 |
| v0 三进程共卡 OOM（worker 静态 37GB + 双 vLLM）| v0 抠不出来（FSDP1 的 optimizer_offload 仅 load_ckpt 时生效，不省训练显存）；**单卡 OPD 只能走 v1**（E1 实测 61GB 有空间） |
| forward_kl_topk 需要全词表 log_softmax（~5.5GB）| 单卡换 `loss_mode=low_var_kl`（单样本 KL 估计，仍是纯 OPD 语义）；topk 模式留 2 卡用 |
| expandable_segments 在双 vLLM 共卡场景 | 会让 PyTorch 缓存池膨胀撑爆整卡，**勿加** |
| teacher util 0.20 起不来 | 7B 权重 15.5GB，util 0.20 只剩 1.1GB KV 池不够 → 0.21 起 |
| 无卡审核方法（省 GPU 钱）| TQ 的元数据/组装逻辑是纯 CPU 的，可无卡复现；veRL 数据流逐环节静态推演 + CPU 验证脚本，比"跑一轮看打印"快且省 |
| **veRL 启动 cwd 决定 outputs/checkpoint 位置**（2026-09-04 step32 保存崩）| SSH 默认 cwd=/root（系统盘 30G）→ checkpoint 13G/份写系统盘 → 100% 满 → `basic_ios::clear: iostream error`。**必须 `cd /root/autodl-tmp && bash run_*.sh` 启动**（E1 就是从数据盘 cwd 启动的所以没踩）。教训：任何写盘操作先确认 cwd 所在盘 |
| **vLLM teacher 引擎的 ~24GB 不透明固定开销**（2026-09-04 单卡死刑的根源）| 7B teacher 在 PRO 6000 上实测 38-40GB，其中权重仅 15.5G，**util 0.25→0.19 只降 1GB**——大头 ~24GB 与任何配置无关（疑似 CUDA 上下文/工作区/allocator 池）。单卡 OPD 总账 = worker 静态 37 + teacher 38 + rollout 权重 6.2 + 波动 ≈ 任何单卡容量。**结论：OPD 单卡不可行，必须双卡**（student/teacher 分卡）。教训：显存预算不能只加"权重+KV"，要给 vLLM 引擎留 20G+ 的不可控开销 |
| **AutoDL 镜像克隆后 torch 损坏**（2026-09-03 PRO6000 迁移踩）| 症状：`import vllm` 崩 `AssertionError: duplicate template name`（torch._inductor 模板重复注册）。根因：镜像恢复后 site-packages/torch 里混入旧版孤儿文件（如 `_inductor/kernel/flex_attention.py` 顶层旧布局 vs 新版 `kernel/flex/` 子目录），`import_submodule` 全目录 import → 模板注册两次。修复：**用 `torch-*.dist-info/RECORD` 清单对比找出全部孤儿文件删除**（142 个）；`pip --force-reinstall` 不删孤儿文件。重装 torch 前先清空间（pip cache 8G + /tmp）|

## 八、可复用资产清单

- 实例 `/root/autodl-tmp/`：run_e1.sh、remote_watchdog_run.sh、remote_monitor.py、sandbox_exec.py、gen_eval.py、eval_scores.py、gen_lcb.py、my_lcb_eval.py、convert_ckpt.py、analyze_training.py、remote_smoke.py、debug_reward.py、test_timeout_fix.py
- 本地 `C:/new/intern/plan/projects/03-code-r1/`：run_ssh.py、e1_report.html、plot_training.py、costs.jsonl
- 已安装环境（勿重装）：veRL main 0.10.0.dev、torch 2.11 cu130、vLLM 0.24、flash-attn 2.8.3、evalplus 0.3.1、LCB repo（/root/autodl-tmp/LiveCodeBench）
