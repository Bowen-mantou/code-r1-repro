"""修复 verifier_score 链路（2026-09-06 深夜根因修复）

根因：agent_loop 写 final_output.extra_fields["verifier_score"]，但 TQ 把 extra_fields
作为 dict 整体存储（list_of_dict_to_tensordict 不展平 dict 键）→
non_tensor_batch 里没有独立 verifier_score 键 → reward_fn 和蒸馏 gate 都读不到。

修复：TQ put 前的 field 构造处，把 verifier_score 提取为顶层键。
一处修复同时接通 E7 奖励链路（extra_info["verifier_score"]）和阶段 3 门控链路。
"""
p = "/root/autodl-tmp/verl/verl/trainer/ppo/v1/agent_loop_tq.py"
src = open(p, encoding="utf-8").read()

anchor = """            field["multi_modal_inputs"] = multi_modal_inputs
            fields.append(field)"""

assert anchor in src, "anchor not found"

new = """            field["multi_modal_inputs"] = multi_modal_inputs
            # E9 修复：extra_fields(dict) 不自动展平，verifier_score 必须提为顶层键
            # （reward_fn extra_info 与 RG-OPD gate 都从 non_tensor_batch 顶层读）
            _vsf = field["extra_fields"].get("verifier_score")
            if _vsf is not None:
                field["verifier_score"] = float(_vsf)
            fields.append(field)"""

src = src.replace(anchor, new)
open(p, "w", encoding="utf-8").write(src)
print("agent_loop_tq.py patched: verifier_score -> top-level field")
