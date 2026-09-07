"""evalplus 官方评估：HumanEval+ / MBPP+（生成结果已在 evalplus_results/）。

官方流程：先 sanitize（提取代码块/目标代码）再 evaluate。
"""
import sys

sys.path.insert(0, "/root/autodl-tmp")
from evalplus.sanitize import script as sanitize_script
from evalplus.evaluate import evaluate

for ds in ["humaneval", "mbpp"]:
    f = f"/root/autodl-tmp/evalplus_results/{ds}.jsonl"
    sanitize_script(f, inplace=True)
    evaluate(dataset=ds, samples=f)
print("EVAL-DONE")
