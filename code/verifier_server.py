#!/usr/bin/env python3
"""VerifierInference：agent_loop worker 内常驻推理器（E7/E6' 共用）。

设计取舍：不搞独立 vLLM 服务（复杂度高），在 AgentLoopWorkerTQ 初始化时
加载一份 1.5B 模型（~4GB）到 rollout GPU，postprocess 里 batch 推理。
单卡单 worker → 仅一份模型。

推理：batch 构造「题目+代码+指令」→ 取第一个生成 token 的 logits
     → softmax 6 类（0-5）→ E[k]/5 = V_raw → Platt 校准 → V_cal
"""
import json
import math
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_LEN = 4096
INSTR = "这段代码能通过几个测试用例？只回答一个数字 (0-5)。"
DIGITS = ["0", "1", "2", "3", "4", "5"]


def truncate_prompt(text: str, max_chars: int = 1024) -> str:
    return text[:max_chars]


def truncate_code(code: str, max_chars: int = 2048) -> str:
    return code[-max_chars:]


class VerifierInference:
    def __init__(self, model_path: str, calibration_path: str | None = None,
                 device: str = "cuda"):
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
        self.model.eval()
        self.device = device
        self.digit_ids = self.tok.convert_tokens_to_ids(DIGITS)
        # 校准参数
        self.a, self.b = 1.0, 0.0
        if calibration_path and os.path.exists(calibration_path):
            cal = json.load(open(calibration_path, encoding="utf-8"))
            self.a = cal.get("platt_coef", 1.0)
            self.b = cal.get("platt_intercept", 0.0)
            print(f"[verifier] Platt 校准加载: a={self.a:.3f} b={self.b:.3f} "
                  f"(ECE={cal.get('ece_platt', '?')})")

    def build_inputs(self, prompts, codes):
        texts = []
        for p, c in zip(prompts, codes):
            p = truncate_prompt(p)
            c = truncate_code(c)
            texts.append(f"{p}\n\n```python\n{c}\n```\n\n{INSTR}")
        return texts

    @torch.no_grad()
    def score_batch(self, prompts, codes) -> list[float]:
        """返回校准后的 V_cal ∈ [0,1]（期望通过率）"""
        texts = self.build_inputs(prompts, codes)
        enc = self.tok(texts, truncation=True, max_length=MAX_LEN,
                       padding=True, return_tensors="pt").to(self.device)
        out = self.model(**enc, use_cache=False)
        # 最后一个有效 token 位置的 logits
        last_pos = enc["attention_mask"].sum(dim=1) - 1
        logits = out.logits[torch.arange(len(texts)), last_pos]  # (B, V)
        # 6 类分布
        digit_logits = logits[:, self.digit_ids]  # (B, 6)
        probs = torch.softmax(digit_logits.float(), dim=-1)  # (B, 6)
        e_k = (probs * torch.arange(6, device=self.device)).sum(dim=-1)  # E[k]
        v_raw = (e_k / 5.0).cpu().numpy()  # E[k]/5 ∈ [0,1]
        # Platt 校准：logit(V) = log(V/(1-V))，clamp 防除零
        eps = 1e-6
        logit_v = [math.log(max(v, eps) / max(1 - v, eps)) for v in v_raw]
        v_cal = [1 / (1 + math.exp(-(self.a * lv + self.b))) for lv in logit_v]
        return v_cal


# ---- Ray 单例 actor（8 个 agent loop worker 共享 1 份模型，防 OOM）----
_ACTOR_NAME = "e7_verifier_actor"


def get_or_create_verifier_actor(model_path, calibration_path, gpu_id=1):
    """获取/创建全局单例 verifier actor（默认放卡 1，student 独占卡 0）

    并发安全：8 个 worker 同时调用时，check-then-act 竞态由
    ActorAlreadyExistsError 兜底（捕获后 get_actor）。
    """
    import ray
    try:
        return ray.get_actor(_ACTOR_NAME)
    except ValueError:
        pass
    Actor = ray.remote(_VerifierActor)
    try:
        return Actor.options(
            name=_ACTOR_NAME,
            runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": str(gpu_id)}},
        ).remote(model_path, calibration_path)
    except ray.exceptions.ActorAlreadyExistsError:
        return ray.get_actor(_ACTOR_NAME)


class _VerifierActor:
    """Ray actor 包装：内部攒批（per-sample 调用合并为 batch 推理，防串行瓶颈）"""
    def __init__(self, model_path, calibration_path):
        self.verifier = VerifierInference(model_path, calibration_path)
        import queue as _q
        import threading as _th
        self._q = _q.Queue()
        _th.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        import queue as _q
        import time as _t
        while True:
            items = [self._q.get()]  # 阻塞等第一条
            deadline = _t.time() + 0.02  # 20ms 攒批窗口
            while _t.time() < deadline:
                try:
                    items.append(self._q.get(timeout=deadline - _t.time()))
                except _q.Empty:
                    break
            try:
                vs = self.verifier.score_batch(
                    [i[1] for i in items], [i[2] for i in items])
            except Exception as e:
                vs = [0.0] * len(items)
                print(f"[verifier] batch 推理失败: {e}", flush=True)
            for item, v in zip(items, vs):
                item[0].put(float(v))

    def score_batch(self, prompts, codes):
        """兼容 per-sample 调用（agent loop 每次 1 条）——内部合并 batch"""
        import queue as _q
        fut = _q.Queue(maxsize=1)
        self._q.put((fut, prompts[0], codes[0]))
        return [fut.get()]
