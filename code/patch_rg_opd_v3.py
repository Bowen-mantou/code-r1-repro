"""RG-OPD gate v3：加门控统计打印（每批打印 gate 通过率——verifier 门控的观测证据）。"""
p = "/root/autodl-tmp/verl/verl/trainer/distillation/losses.py"
src = open(p, encoding="utf-8").read()

old = """                    gate = (pos | neg).to(student_log_probs.dtype)
                    distillation_losses = distillation_losses * gate.unsqueeze(-1)"""

new = """                    gate = (pos | neg).to(student_log_probs.dtype)
                    distillation_losses = distillation_losses * gate.unsqueeze(-1)
                    # v3 观测：门控通过率（<1.0 说明有轨迹被 verifier 过滤）
                    gate_frac = (gate > 0).float().mean().item()
                    print(f"[rg-opd] gate frac: {gate_frac:.3f} "
                          f"(bsz={gate.shape[0]})", flush=True)"""

assert old in src, "v2 gate block not found"
src = src.replace(old, new)
open(p, "w", encoding="utf-8").write(src)
print("losses.py patched v3: gate frac logging added")
