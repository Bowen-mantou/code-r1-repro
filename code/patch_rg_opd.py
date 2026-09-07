"""RG-OPD 门控 patch（阶段 3 核心，2026-09-07）

1. losses.py：清掉 5 块重复的 E6' 软门控代码 → 1 块 RG-OPD sign gate
   g = I[(A>0 ∧ L_T > L_S+δ) ∨ (A≤0 ∧ L_T < L_S−δ)]
   A = verifier_score（V_cal，>0.5 视为正）；L_T/L_S = teacher/student 轨迹级 log-lik
   链路不通时 fallback 无条件蒸馏（全 1 gate）+ 警告——训练不会崩
2. workers/config/distillation.py：DistillationLossConfig 加 4 个字段
"""
p = "/root/autodl-tmp/verl/verl/trainer/distillation/losses.py"
src = open(p, encoding="utf-8").read()

anchor_start = ("    # E6' verifier 软门控：w = V^gamma 加权"
                "（data 里 verifier_score 为每样本标量）")
anchor_end = "    # Since k1 can be negative, log the mean absolute loss."

i = src.index(anchor_start)
j = src.index(anchor_end)
assert i < j, "anchors not found"

new_block = '''    # RG-OPD 门控（2026-09-07）：verifier 决定「这步蒸不蒸」
    # g = I[(A>0 ∧ L_T > L_S+δ) ∨ (A≤0 ∧ L_T < L_S−δ)]，A=V_cal>0.5
    # gating_mode: "rlopd"（sign gate，默认）/ "soft"（V^γ 加权，E6' 原版）
    # verifier_score 缺失/形状不符时 fallback 全 1（无条件蒸馏）+ 警告
    if getattr(loss_config, "verifier_gating", False):
        vs = data.get("verifier_score", None)
        if vs is not None:
            if hasattr(vs, "data"):
                vs = vs.data
            vs_t = torch.as_tensor(vs, dtype=student_log_probs.dtype,
                                   device=student_log_probs.device)
            if vs_t.dim() == 0:
                vs_t = vs_t.unsqueeze(0)
            if vs_t.shape[0] != student_log_probs.shape[0]:
                print(f"[rg-opd] WARN: verifier_score bsz {vs_t.shape[0]} != "
                      f"batch {student_log_probs.shape[0]}, gate disabled",
                      flush=True)
            else:
                mode = getattr(loss_config, "gating_mode", "rlopd")
                if mode == "soft":
                    w = vs_t ** getattr(loss_config, "gating_gamma", 1.0)
                    distillation_losses = distillation_losses * w.unsqueeze(-1)
                else:
                    delta = getattr(loss_config, "gating_delta", 0.0)
                    mask_t = response_mask_bool.to(student_log_probs.dtype)
                    lt = (teacher_log_probs * mask_t).sum(dim=-1)
                    ls = (student_log_probs * mask_t).sum(dim=-1)
                    pos = (vs_t > 0.5) & (lt > ls + delta)
                    neg = (vs_t <= 0.5) & (lt < ls - delta)
                    gate = (pos | neg).to(student_log_probs.dtype)
                    distillation_losses = distillation_losses * gate.unsqueeze(-1)
        else:
            print("[rg-opd] WARN: verifier_score not in data, "
                  "unconditional distillation (gate=1)", flush=True)
'''
src = src[:i] + new_block + src[j:]
open(p, "w", encoding="utf-8").write(src)
print("losses.py patched (5 dup blocks -> 1 RG-OPD gate)")

# 2. config 字段
p2 = "/root/autodl-tmp/verl/verl/workers/config/distillation.py"
src2 = open(p2, encoding="utf-8").read()
anchor = "    loss_settings (DistillationLossSettings, optional):"
assert anchor in src2
insert = '''    verifier_gating (bool):
        RG-OPD gate for distillation loss (2026-09-07). Default False.
    gating_mode (str):
        "rlopd" (sign gate, default) or "soft" (V^gamma weighting).
    gating_gamma (float):
        Exponent for soft mode.
    gating_delta (float):
        Confidence margin for sign gate (default 0.0).
    loss_settings (DistillationLossSettings, optional):'''
src2 = src2.replace(anchor, insert)

anchor2 = "    # Chunked top-K log-probs (opt-in, avoids [B, T, V] log_softmax buffer"
assert anchor2 in src2
fields = '''    # RG-OPD 门控（2026-09-07）
    verifier_gating: bool = False
    gating_mode: str = "rlopd"
    gating_gamma: float = 1.0
    gating_delta: float = 0.0

    # Chunked top-K log-probs (opt-in, avoids [B, T, V] log_softmax buffer'''
src2 = src2.replace(anchor2, fields)
open(p2, "w", encoding="utf-8").write(src2)
print("distillation.py config patched (verifier_gating/gating_mode/gating_gamma/gating_delta)")
