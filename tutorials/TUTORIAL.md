# TUTORIAL：3B 代码推理 RL 多信号对照实验全流程

> 从零复现本仓库的六个模型。硬件：A800 80GB ×2（AutoDL），软件：veRL main 0.10 + torch 2.11 cu130 + vLLM 0.24。
> 总预算参考：~¥400（含全部学费）；熟练后单实验 ¥30-45。

## 目录

1. 环境准备（30 分钟）
2. 数据（code-r1-lc2k）
3. E1：GRPO 基线
4. E5：OPD 蒸馏
5. Verifier 全链路（数据→训练→校准）
6. E7/E9p2：GRPO + verifier 集成
7. E9p3：RG-OPD 门控蒸馏
8. 评估体系
9. 监控与止损

---

## 1. 环境准备

```bash
# 红线：永远不要 pip install flashinfer（会静默降级 torch）
python3 -c "import torch; print(torch.__version__)"  # 必须 2.11.0+cu130

# 关键依赖
git clone https://github.com/volcengine/verl && cd verl && pip install -e .
pip install vllm==0.24 evalplus

# vLLM EngineCore 需要 cu13 运行库
echo "/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib" > /etc/ld.so.conf.d/cu13.conf
ldconfig
```

**无卡模式预演原则**：所有配置验证、数据检查、代码审查在无卡模式（¥0.1/hr）完成，
GPU 只跑「验证过的东西」。

## 2. 数据

训练数据 `code-r1-lc2k/train.parquet`（2k 题），评估数据 LCB v5 官方 880 题：

```bash
# LCB 数据（main 分支 5 文件 = 880 题；2025-06-05 后冻结）
cd /root/autodl-tmp
for f in test.jsonl test2.jsonl test3.jsonl test4.jsonl test5.jsonl; do
  aria2c -x 16 -s 16 -k 1M --file-allocation=falloc \
    -o "$f" "https://hf-mirror.com/datasets/livecodebench/code_generation_lite/resolve/main/$f"
done
# 校验（HF API ?blobs=true 拿官方 sha256）
sha256sum -c lcb.sha
```

⚠️ 三个必踩的坑：
- 历史 commit 冷对象下载极慢（60KB/s）vs main 热缓存（31MB/s）——用 main
- aria2c 必须 `--file-allocation=falloc`（none 会留下 NUL 空洞，文件大小却"正确"）
- 下完 sha256 校验（大小一致 ≠ 内容完整）

## 3. E1：GRPO 基线（~¥70 含学费）

```bash
bash run_e1.sh   # 核心配置：GRPO + 执行奖励，n=4，batch 8，total_epochs=2（不要 4！）
```

**关键结论**：`total_epochs=2` 是峰值（LCB 25.2）；`4ep` 过训在难题上灾难性退化（19.8）。
GRPO 过训 = entropy collapse，文献见 TS-OPSD（arXiv 2606.00755）。

## 4. E5：OPD 蒸馏（~¥45）

```bash
bash run_e5.sh   # 核心：distillation.enabled=True + loss_mode=low_var_kl + use_task_rewards=False
```

**红线：单卡 OPD 是死刑**。7B teacher 的 vLLM 引擎有 ~24GB 不透明固定开销 +
worker 37GB > 任何单卡容量。必须双卡（student 卡 0 / teacher 卡 1）。

## 5. Verifier 全链路（~¥13）

```bash
# 5.1 数据：沙箱执行自动标注（correct + 子集 k/N 标签）
python3 gen_verifier_data.py

# 5.2 训练 1.5B 生成式 verifier（6 分类预测 k/N）
# 5.3 校准：温度 + Platt 双校准，目标 ECE < 0.01
python3 calibrate_verifier.py   # 本仓库最终 ECE = 0.0067
```

## 6. E7/E9p2：GRPO + verifier 集成（~¥45）

奖励函数：`R = result_bin + λ_v × V_cal`（λ_v=0.1，执行结果主导防黑客）。

```bash
bash run_e7.sh        # 起点 = Qwen 基模
bash run_e9_phase2.sh # 起点 = e5-final-3b（warm-start 对照）
```

### ⚠️ 本仓库最重要的教训：verifier 信号链路

veRL 的 `agent_loop` 写 `extra_fields`（dict）→ TQ 存储**不展平 dict 键** →
reward_fn 永远读不到 → **verifier 信号从未生效**。修复（`code/patch_verifier_topkey.py`）：

```python
# agent_loop_tq.py：TQ put 前的 field 构造处
_vsf = field["extra_fields"].get("verifier_score")
if _vsf is not None:
    field["verifier_score"] = float(_vsf)   # 提为顶层键
```

验证方法：训练日志无 `[rg-opd] WARN` + 门控打印 `gate frac < 1.0`。

## 7. E9p3：RG-OPD 门控蒸馏（~¥20）

```bash
bash run_e9_phase3.sh # OPD low_var_kl + verifier_gating=True（rlopd sign gate）
```

门控公式（RG-OPD, arXiv 2607.04037）：
`g = I[(V>0.5 ∧ L_T > L_S+δ) ∨ (V≤0.5 ∧ L_T < L_S−δ)]`

**边界条件发现**：teacher=student（自蒸馏）时 L_T≈L_S 是噪声比较，叠加 V 条件后
通过率仅 10-18%，蒸馏无新知识可传 → 无增量（LCB -0.4 噪声内）。
**teacher 必须强于 student**——用 7B 而非自蒸馏。

## 8. 评估体系

```bash
# LCB（880 题，与官方口径一致）
python3 gen_lcb_fast.py   # 两阶段：80 题串行自举 + 800 题装箱批量
python3 my_lcb_eval.py    # 官方 codegen_metrics 打分

# evalplus
python3 gen_eval.py && python3 eval_scores.py
```

评估口径三戒：
1. 数据 5 文件并集 = 880 题（缺一个文件都不行）
2. 打分在单卡实例跑（无卡 2GB 内存会 OOM；`/sys/fs/cgroup/memory.max` 是真实限额，不信 free）
3. 生成脚本坏行要报错退出（静默跳过 = 改变口径）

## 9. 监控与止损

```bash
# watchdog：训练完成/崩溃 → 自动关机（省钱的 24/7 监控）
bash remote_watchdog_e9p2.sh
# 外挂磁盘清理器（max_keep=1 在 resume 场景失效，必须兜底）
bash clean_old_ckpt.sh
```

预算纪律：无卡模式预演一切 → GPU 开机即跑 → watchdog 自动关机 → 每步记账。
本仓库 MISTAKES 账本记录的可避免浪费 ~¥25（磁盘事故 ¥10.5 最大）。

---

## 附：花费分解参考

| 环节 | 成本 |
|------|------|
| 环境+数据（无卡）| ~¥1 |
| 单实验训练（2ep 双卡 2.5-3h）| ~¥29-35 |
| 单实验评估（LCB+evalplus 单卡）| ~¥8-15 |
| verifier 全链路 | ~¥13 |
