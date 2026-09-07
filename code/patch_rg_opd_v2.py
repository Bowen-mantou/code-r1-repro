"""RG-OPD gate v2 修复：NonTensorStack -> list -> tensor（v1 的 as_tensor 得到空张量）。"""
p = "/root/autodl-tmp/verl/verl/trainer/distillation/losses.py"
src = open(p, encoding="utf-8").read()

old = """    if getattr(loss_config, "verifier_gating", False):
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
            else:"""

new = """    if getattr(loss_config, "verifier_gating", False):
        vs = data.get("verifier_score", None)
        if vs is not None:
            # NonTensorStack -> list -> tensor（v1 直接 as_tensor 得空张量）
            try:
                vs_l = vs.tolist() if hasattr(vs, "tolist") else list(vs)
            except Exception:
                vs_l = None
            if vs_l is None or len(vs_l) != student_log_probs.shape[0]:
                print(f"[rg-opd] WARN: verifier_score bad shape "
                      f"({len(vs_l) if vs_l else '?'} vs "
                      f"{student_log_probs.shape[0]}), gate disabled",
                      flush=True)
            else:
                vs_t = torch.as_tensor(vs_l, dtype=student_log_probs.dtype,
                                       device=student_log_probs.device)"""

assert old in src, "v1 gate block not found"
src = src.replace(old, new)
open(p, "w", encoding="utf-8").write(src)
print("losses.py patched v2: NonTensorStack -> list -> tensor")
