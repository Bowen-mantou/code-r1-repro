"""E9 阶段 3：veRL FSDP checkpoint (global_step_236) → HF 格式（bf16），评估用"""
import os
import shutil

import torch
from transformers import AutoModelForCausalLM

CKPT = ("/root/autodl-tmp/checkpoints/code-r1/e9-phase3-online-distill/"
        "global_step_236/actor/model_world_size_1_rank_0.pt")
CFG_DIR = ("/root/autodl-tmp/checkpoints/code-r1/e9-phase3-online-distill/"
           "global_step_236/actor/huggingface")
DST = "/root/autodl-tmp/models/e9-phase3-final"

sd = torch.load(CKPT, map_location="cpu")
if "model" in sd and isinstance(sd["model"], dict):
    sd = sd["model"]

print("state_dict keys sample:", list(sd.keys())[:5])
print("n tensors:", len(sd))

from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(CFG_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_config(
    cfg, torch_dtype=torch.bfloat16, trust_remote_code=True)

hf_keys = set(model.state_dict().keys())
if not hf_keys.issubset(sd.keys()):
    stripped = {}
    for k, v in sd.items():
        nk = k
        for prefix in ("_fsdp_wrapped_module.", "module.", "model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
                break
        stripped[nk] = v
    sd = stripped

missing = hf_keys - sd.keys()
unexpected = sd.keys() - hf_keys
print(f"missing: {len(missing)}, unexpected: {len(unexpected)}")
if missing:
    print("missing sample:", list(missing)[:5])

model.load_state_dict(sd)
model.save_pretrained(DST)

for f in os.listdir(CFG_DIR):
    if not f.endswith(".safetensors"):
        src = os.path.join(CFG_DIR, f)
        dst = os.path.join(DST, f)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

print(f"converted to {DST}")
print("size:", sum(
    os.path.getsize(os.path.join(dp, f))
    for dp, _, fns in os.walk(DST) for f in fns) / 1e9, "GB")
