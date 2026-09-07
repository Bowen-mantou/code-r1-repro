"""E1 模型 LiveCodeBench v5 生成（transformers，LCB 官方 CodeQwen prompt 口径）。

完全自包含：prompt 构造与提取逻辑照抄 LCB 官方实现（避免其 anthropic 依赖问题）。
输入: test5.jsonl (release_v5, 880 题)
输出: lcb_custom_output.json (custom_evaluator 格式)
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/root/autodl-tmp/models/e7-final-3b"
DATA_FILES = ["/root/autodl-tmp/test.jsonl",
              "/root/autodl-tmp/test2.jsonl",
              "/root/autodl-tmp/test3.jsonl",
              "/root/autodl-tmp/test4.jsonl",
              "/root/autodl-tmp/test5.jsonl"]
OUT = "/root/autodl-tmp/lcb_custom_output.json"

SYSTEM_CODEQWEN = ("<|im_start|>system\nYou are a helpful assistant."
                   "<|im_end|>\n<|im_start|>user")
FORMATTING_WITH_STARTER = ("You will use the following starter code to write "
                           "the solution to the problem and enclose your code "
                           "within delimiters.")
FORMATTING_WITHOUT_STARTER = ("Read the inputs from stdin solve the problem "
                              "and write the answer to stdout (do not directly "
                              "test on the sample inputs). Enclose your code "
                              "within delimiters as follows. Ensure that when "
                              "the python program runs, it reads the inputs, "
                              "runs the algorithm and writes output to STDOUT.")


def build_prompt(q_content, starter_code):
    p = (f"{SYSTEM_CODEQWEN}\n\n"
         "You will be given a question (problem specification) and will "
         "generate a correct Python program that matches the specification "
         "and passes all tests. You will NOT return anything except for the "
         f"program.\n\nQuestion: {q_content}\n\n")
    if starter_code:
        p += f"{FORMATTING_WITH_STARTER}\n"
        p += f"```python\n{starter_code}\n```\n\n<|im_end|>\n"
    else:
        p += f"{FORMATTING_WITHOUT_STARTER}\n"
        p += "```python\n# YOUR CODE HERE\n```\n\n<|im_end|>\n"
    p += "<|im_start|>assistant\n"
    return p


def extract_code(text):
    """照抄 LCB extract_code 的 else 分支（CodeQwen style）。"""
    lines = text.split("\n")
    idx = [i for i, line in enumerate(lines) if "```" in line]
    if len(idx) < 2:
        return ""
    return "\n".join(lines[idx[-2] + 1: idx[-1]])


def main():
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16).cuda()
    model.eval()

    # 断点续跑：加载已有结果，跳过已生成的题
    done_ids = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            done_ids = {r["question_id"] for r in json.load(f)}
    results = []
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            results = json.load(f)

    problems = []
    for path in DATA_FILES:
        if not os.path.exists(path):
            print(f"skip missing {path}", flush=True)
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    prob = json.loads(line)
                except Exception as e:
                    raise RuntimeError(f"bad JSON in {path}: {e}")
                if prob["question_id"] not in done_ids:
                    problems.append(prob)

    print(f"to generate: {len(problems)} (done: {len(done_ids)})", flush=True)
    for i, prob in enumerate(problems):
        prompt = build_prompt(prob["question_content"],
                              prob.get("starter_code", ""))
        inputs = tok(prompt, return_tensors="pt").to("cuda")
        out = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        gen = out[0][inputs.input_ids.shape[1]:]
        text = tok.decode(gen, skip_special_tokens=True)
        results.append({"question_id": prob["question_id"],
                        "code_list": [extract_code(text)]})
        if (i + 1) % 50 == 0:
            print(f"lcb {i+1}/{len(problems)}", flush=True)
            with open(OUT, "w", encoding="utf-8") as f:
                json.dump(results, f)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"lcb done -> {OUT} ({len(results)} problems)", flush=True)


if __name__ == "__main__":
    main()
