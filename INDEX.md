# 03-code-r1 文档索引（INDEX）

> 项目已完结（2026-09-07）。按用途选入口。

## 教程（tutorials/，面试/学习核心）

| 文档 | 用途 | 规模 |
|------|------|------|
| [tutorials/INTERVIEW-MASTER.md](tutorials/INTERVIEW-MASTER.md) | **面试弹药库**：公式→代码→工程→设计 + 100 题问答 + 跨项目对比 + 速查卡 | 9 章 |
| [tutorials/ENGINEERING-DEEP-DIVE.md](tutorials/ENGINEERING-DEEP-DIVE.md) | **工程深度**：veRL 架构/显存账/监控体系/迁移清单 | 10 章 |
| [tutorials/CODE-WALKTHROUGH.md](tutorials/CODE-WALKTHROUGH.md) | **代码带读**：逐文件逐段精读 + 自查清单 | 7 章 |

## 核心文档

| 文档 | 用途 |
|------|------|
| [2026-09-07-FINAL-REPORT.md](2026-09-07-FINAL-REPORT.md) | 最终报告：六模型全指标 + 5 条结论 + 花费账 |
| [PITFALLS.md](PITFALLS.md) | 踩坑手册（15+ 条，后续实验必读）|

## 研究过程文档

| 文档 | 内容 |
|------|------|
| [2026-09-06-research.md](2026-09-06-research.md) | 方案全集 A-F + ¥100 预算分配 + 论文调研综合 |
| [research/papers/](research/papers/) | 7 篇关键论文中文摘要（GLM-5/SeqBeatsJoint/RG-OPD/SG-OPD/OPDVR/RWOPD/E7 理论框架/veRL OPD 实现）|
| [archive/](archive/) | 每日 CONTINUE/work-log 等过程记录（历史存档）|

## 本地目录结构

```
03-code-r1/
├── docs/                  # 全部文档（核心 5 篇 + 研究 8 篇 + 归档）
├── github_release/        # GitHub 发布目录（git 仓库，与远程同步）
├── backups/               # 22G：E5 ckpt 13.6G + models + 服务器资产 + 日志
├── archive/               # 归档：dev 调试脚本 / dl 下载脚本 / reports 旧报告 / figures 图表
├── eval_results_e5|e7/    # E5/E7 评估原始结果
├── run_*.sh               # 各实验训练入口（e1/e2/e5/e7/e9p2/e9p3）
├── reward_fn.py           # 奖励函数
├── verifier_server.py     # verifier 推理服务（Ray actor + 攒批）
├── patch_*.py             # 全部 verl patch（含 RG-OPD 三版 + 链路修复）
├── gen_*.py / my_lcb_eval.py  # 评估管线
├── convert_ckpt_*.py      # ckpt → hf 转换（每实验一份）
└── costs.jsonl            # 花费流水
```

## GitHub 仓库（Bowen-mantou/code-r1-repro）

README / TUTORIAL / REPORT / PITFALLS / INTERVIEW-MASTER /
ENGINEERING-DEEP-DIVE / CODE-WALKTHROUGH / papers / code / results

## 推送命令备忘

```bash
cd C:/new/intern/plan/projects/03-code-r1/github_release
cp ../docs/INTERVIEW-MASTER.md .
git add -A && git commit -m "docs: ..."
git -c http.proxy=http://127.0.0.1:7897 -c https.proxy=http://127.0.0.1:7897 push
```
