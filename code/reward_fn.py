"""veRL-0.10-compatible reward function for Code-R1 experiments.

E1 (binary): the original code-r1 reward — format +0.1, all tests pass +1.0,
             bad format -1.1 (identical semantics to verl 0.2.0.dev coder1).
E2/E6 (PRM): the 4-layer Code PRM from code_prm.reward.CodePRM.

Wired via veRL main config:
  reward.custom_reward_function.path=<abs path to this file>
  reward.custom_reward_function.name=compute_score
  reward.custom_reward_function.reward_kwargs.use_prm=True   # E2/E6; omit for E1

Field access notes (veRL 0.10 v1 trainer, from project-1 E6-B debugging):
  - the reward fn receives a DataProto whose non_tensor_batch carries the
    dataset columns; `responses` text may arrive under different keys —
    verify on the instance with DEBUG_REWARD_FN=1 and adjust below.
"""
import json
import os

from concurrent.futures import ThreadPoolExecutor


def _binary_score(solution_str, ground_truth, extra_info):
    from verl.utils.reward_score.coder1 import compute_score
    # identical to the original code-r1 reward (format 0.1 + answer 1.0)
    return compute_score(solution_str, ground_truth, extra_info,
                         format_reward=0.1, answer_reward=1.0)


_PRM = None


def _prm_score(solution_str, ground_truth, extra_info, gold_code=None):
    global _PRM
    if _PRM is None:
        from code_prm.reward import CodePRM
        _PRM = CodePRM()
    total, _ = _PRM.compute_reward(solution_str, ground_truth,
                                   extra_info=extra_info, gold_code=gold_code)
    return total


def _get_completions(data):
    """Extracts decoded completion strings from whatever shape veRL passes."""
    if hasattr(data, "non_tensor_batch"):
        ntb = data.non_tensor_batch
        if "responses" in ntb:
            return list(ntb["responses"])
        if "response_text" in ntb:
            return list(ntb["response_text"])
    if isinstance(data, dict):
        for key in ("responses", "response_text", "completions"):
            if key in data:
                return list(data[key])
    raise KeyError("cannot locate completion texts in reward input")


def _get_column(data, name):
    if hasattr(data, "non_tensor_batch") and name in data.non_tensor_batch:
        return list(data.non_tensor_batch[name])
    if isinstance(data, dict) and name in data:
        return list(data[name])
    return None


def compute_score(data_source, solution_str, ground_truth, extra_info,
                  use_prm=False, use_verifier=False, lambda_v=0.1,
                  num_processes=8, **kwargs):
    """veRL 0.10 reward-manager entry point.

    NaiveRewardManager calls this per-sample:
        compute_score(data_source=..., solution_str=..., ground_truth=...,
                      extra_info=...)
    BatchRewardManager calls with plural lists (handled below recursively).

    E7 verifier 模式：R = Result_bin(0/1) + lambda_v * V_cal
      - Result 来自沙箱执行（binary 归一到 0/1）
      - V_cal 来自 agent loop 注入 extra_info["verifier_score"]（已校准）
      - 核心原则：Result 主导，λ 小（0.1 起步，通过率下降优先降 λ）
    """
    if os.environ.get("DEBUG_REWARD_FN"):
        print("[reward_fn] use_prm:", use_prm, "| use_verifier:", use_verifier,
              "| ds:", data_source, flush=True)

    # BatchRewardManager plural form -> recurse per sample
    if isinstance(solution_str, list):
        n = len(solution_str)
        dss = data_source if isinstance(data_source, list) else [data_source] * n
        gts = ground_truth if isinstance(ground_truth, list) else [ground_truth] * n
        eis = extra_info if isinstance(extra_info, list) else [extra_info] * n
        return [compute_score(dss[i], solution_str[i], gts[i], eis[i],
                              use_prm=use_prm, use_verifier=use_verifier,
                              lambda_v=lambda_v) for i in range(n)]

    gt = ground_truth.get("ground_truth") if isinstance(ground_truth, dict) \
        else ground_truth
    try:
        if use_verifier:
            # E7：结果门控主导 + verifier 置信度辅助
            raw = float(_binary_score(solution_str, gt, extra_info))
            result_bin = 1.0 if raw >= 1.0 else 0.0  # 1.1 满分制 → 0/1
            # verifier_score 可能在 extra_info 顶层或嵌套在 extra_fields
            v = extra_info.get("verifier_score") if isinstance(extra_info, dict) else None
            if v is None and isinstance(extra_info.get("extra_fields"), dict):
                v = extra_info["extra_fields"].get("verifier_score")
            v = float(v or 0.0)
            return result_bin + lambda_v * v
        if use_prm:
            return float(_prm_score(solution_str, gt, extra_info))
        return float(_binary_score(solution_str, gt, extra_info))
    except Exception:
        return 0.0
