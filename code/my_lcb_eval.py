"""LCB v5 评估（自研入口：本地 jsonl 构造 benchmark + LCB 官方 codegen_metrics）。

绕开 LCB 的 load_dataset（新版 datasets 不支持脚本数据集）。
"""
import json
import sys

sys.path.insert(0, "/root/autodl-tmp/LiveCodeBench")

from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

DATA_FILES = ["/root/autodl-tmp/test.jsonl",
              "/root/autodl-tmp/test2.jsonl",
              "/root/autodl-tmp/test3.jsonl",
              "/root/autodl-tmp/test4.jsonl",
              "/root/autodl-tmp/test5.jsonl"]
OUT = "/root/autodl-tmp/lcb_custom_output.json"


def main():
    problems = []
    for path in DATA_FILES:
        with open(path, encoding="utf-8") as f:
            for line in f:
                problems.append(CodeGenerationProblem(**json.loads(line)))
    problems.sort(key=lambda p: str(p.question_id))
    print(f"problems: {len(problems)}", flush=True)

    with open(OUT, encoding="utf-8") as f:
        outs = json.load(f)
    outs.sort(key=lambda r: str(r["question_id"]))
    generations = [r["code_list"] for r in outs]
    assert len(generations) == len(problems), \
        f"{len(generations)} != {len(problems)}"
    print(f"generations: {len(generations)}", flush=True)

    eval_samples = [p.get_evaluation_sample() for p in problems]
    metrics = codegen_metrics(
        eval_samples, generations,
        num_process_evaluate=16, timeout=6,
    )
    print("pass@1:", metrics[0]["pass@1"], flush=True)
    print("LCB-EVAL-DONE", flush=True)


if __name__ == "__main__":
    main()
