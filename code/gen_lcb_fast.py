"""LCB 快速生成（E7 评估）：两阶段自举先验 + 长度装箱批量。

阶段 1：串行生成前 BOOTSTRAP 题，用真实输出 token 数建立长度先验
阶段 2：剩余题用 BatchGenerator 装箱批量（ratio<=1.5 组批 + 动态 max_new_tokens）
输出与 gen_lcb.py 完全一致：lcb_custom_output.json（question_id + code_list），
支持断点续跑（跳过 done_ids）。
用法: python3 gen_lcb_fast.py
"""
import json
import math
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/root/autodl-tmp/models/e9-phase3-final"
DATA_FILES = ["/root/autodl-tmp/test.jsonl",
              "/root/autodl-tmp/test2.jsonl",
              "/root/autodl-tmp/test3.jsonl",
              "/root/autodl-tmp/test4.jsonl",
              "/root/autodl-tmp/test5.jsonl"]
OUT = "/root/autodl-tmp/lcb_custom_output.json"
BOOTSTRAP = 80      # 阶段 1 串行题数（自举先验）
BATCH_MAX = 8       # 批量上限
RATIO = 1.5         # 装箱长度比上限
MARGIN = 1.5        # 动态 max_new_tokens 裕度
PAD = 64
CAP = 2048          # max_new_tokens 硬上限

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
    lines = text.split("\n")
    idx = [i for i, line in enumerate(lines) if "```" in line]
    if len(idx) < 2:
        return ""
    return "\n".join(lines[idx[-2] + 1: idx[-1]])


def schedule(items, est, default=1024):
    """长度分组装箱：批内预估比 <= RATIO。items: [(qid, prompt)]"""
    scored = sorted(items, key=lambda x: est.get(x[0], default), reverse=True)
    batches = []
    for item in scored:
        e = est.get(item[0], default)
        placed = False
        for batch in batches:
            be = est.get(batch[0][0], default)
            if len(batch) < BATCH_MAX and max(be, e) <= min(be, e) * RATIO:
                batch.append(item)
                placed = True
                break
        if not placed:
            batches.append([item])
    return batches


def gen_one(model, tok, prompt):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=CAP, do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    gen = out[0][inputs.input_ids.shape[1]:]
    text = tok.decode(gen, skip_special_tokens=True)
    return text, gen.shape[0]


def gen_batch(model, tok, batch, est, max_new):
    """批量生成。返回 [(text, hit_cap)]——hit_cap=True 表示撞到 max_new
    上限（可能截断），调用方必须用 CAP=2048 单独重跑，保证口径一致。"""
    prompts = [p for _, p in batch]
    inputs = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    outs = model.generate(
        **inputs, max_new_tokens=max_new, do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    results = []
    for j, out in enumerate(outs):
        plen = inputs["attention_mask"][j].sum().item()
        gen = out[plen:]
        hit_cap = gen.shape[0] >= max_new  # 撞顶 = 可能截断
        results.append((tok.decode(gen, skip_special_tokens=True), hit_cap))
    return results


def main():
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16).cuda()
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # 断点续跑
    done = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            done = {r["question_id"]: r for r in json.load(f)}

    # 严格加载全部题目（坏行直接崩，不静默丢题）
    problems = []
    for path in DATA_FILES:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    prob = json.loads(line)
                except Exception as e:
                    raise RuntimeError(f"bad JSON in {path}: {e}")
                if prob["question_id"] not in done:
                    problems.append(prob)
    print(f"to generate: {len(problems)} (done: {len(done)})", flush=True)

    todo = [(p["question_id"], build_prompt(p["question_content"],
                                             p.get("starter_code", "")))
            for p in problems]
    est = {}  # qid -> 真实生成 token 数

    # 阶段 1：均匀采样自举 BOOTSTRAP 题（串行）——先验覆盖全长分布
    n_boot = min(BOOTSTRAP, len(todo))
    boot_ids = {todo[i * len(todo) // n_boot][0]
                for i in range(n_boot)}
    print(f"phase1 bootstrap: {len(boot_ids)} serial", flush=True)
    i = 0
    for qid, prompt in todo:
        if qid not in boot_ids:
            continue
        text, n_tok = gen_one(model, tok, prompt)
        est[qid] = n_tok
        done[qid] = {"question_id": qid, "code_list": [extract_code(text)]}
        i += 1
        if i % 20 == 0:
            print(f"boot {i}/{len(boot_ids)}", flush=True)
            with open(OUT, "w", encoding="utf-8") as f:
                json.dump(list(done.values()), f)

    # 阶段 2：批量装箱（未知题用阶段 1 长度中位数做默认先验）
    rest = [item for item in todo if item[0] not in boot_ids]
    if rest:
        default = int(sorted(est.values())[len(est) // 2]) if est else 1024
        print(f"phase2 batch: {len(rest)} items, default_len={default}, "
              f"{len(schedule(rest, est, default))} batches", flush=True)
        n = 0
        retry = 0
        for batch in schedule(rest, est, default):
            est_max = max(est.get(q, default) for q, _ in batch)
            max_new = min(CAP, int(math.ceil(est_max * MARGIN)) + PAD)
            results = gen_batch(model, tok, batch, est, max_new)
            for (qid, prompt), (text, hit_cap) in zip(batch, results):
                if hit_cap:
                    # 撞顶重跑：单独用 CAP=2048，与 E1/E2/E5 口径一致
                    text, _ = gen_one(model, tok, prompt)
                    retry += 1
                done[qid] = {"question_id": qid,
                             "code_list": [extract_code(text)]}
            n += len(batch)
            if n % 50 == 0 or n == len(rest):
                print(f"batch {n}/{len(rest)}", flush=True)
                with open(OUT, "w", encoding="utf-8") as f:
                    json.dump(list(done.values()), f)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(list(done.values()), f)
    print(f"lcb done -> {OUT} ({len(done)} problems, "
          f"cap_retries={retry if rest else 0})", flush=True)


if __name__ == "__main__":
    main()
