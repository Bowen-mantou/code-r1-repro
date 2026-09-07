"""E1 模型 HumanEval+/MBPP+ 生成（transformers，官方 prompt 口径）。

绕开 evalplus 的 decoder（其 attn_implementation 参数触发 transformers 5.10 bug），
prompt 构造与 evalplus 官方完全一致；评估仍用 evalplus.evaluate。
"""
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/root/autodl-tmp")
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.provider.utility import make_raw_chat_prompt

MODEL = "/root/autodl-tmp/models/e9-phase3-final"
ROOT = "/root/autodl-tmp/evalplus_results"

INSTRUCTION_PREFIX = ("Please provide a self-contained Python script that "
                      "solves the following problem in a markdown code block:")
RESPONSE_PREFIX = ("Below is a Python script with a self-contained function "
                   "that solves the problem and passes corresponding tests:")


def build_prompt(task_prompt, tokenizer):
    """与 evalplus make_raw_chat_prompt 一致（Qwen2 chat template 路径）。"""
    return make_raw_chat_prompt(task_prompt, INSTRUCTION_PREFIX,
                                RESPONSE_PREFIX, tokenizer)


def main():
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16).cuda()
    model.eval()
    eos = tok.eos_token_id

    for ds in ["humaneval", "mbpp"]:
        data = get_human_eval_plus() if ds == "humaneval" else get_mbpp_plus()
        out_path = os.path.join(ROOT, f"{ds}.jsonl")
        os.makedirs(ROOT, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for i, (task_id, prob) in enumerate(data.items()):
                prompt = build_prompt(prob["prompt"], tok)
                inputs = tok(prompt, return_tensors="pt").to("cuda")
                out = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    eos_token_id=eos,
                    pad_token_id=tok.pad_token_id or eos,
                )
                gen = out[0][inputs.input_ids.shape[1]:]
                text = tok.decode(gen, skip_special_tokens=True)
                f.write(json.dumps({"task_id": task_id,
                                    "solution": text}) + "\n")
                if (i + 1) % 40 == 0:
                    print(f"{ds} {i+1}/{len(data)}", flush=True)
        print(f"{ds} done -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
