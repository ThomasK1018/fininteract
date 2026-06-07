"""
Paraphrase-consistency metrics for FinInteract.

Motivation: a robust agent should recognize that a question is ambiguous and clarify it
*regardless of how the question is phrased*. We probe this by generating paraphrase GROUPS
(the same instance worded several ways) and measuring whether the model's behavior is
invariant across the group. This mirrors nvBench 2.0's finding that LLM behavior "swings
dramatically with minor wording changes" — here we quantify that swing for ambiguity
recognition specifically.

IMPORTANT: this is a *diagnostic* over the test instances, not an addition to the headline
benchmark count. The probe groups are derived from (not added to) the 53-instance test set.

Given, for each paraphrase group, the per-variant trajectories (from env.run_episode), we
report four group-level consistency metrics plus the accuracy for reference.

A `trajectory` needs: correct (bool), n_asks (int), first_ask_hit (bool|None).
`groups` maps group_id -> list[trajectory].
"""

from __future__ import annotations
from statistics import mean


def consistency_metrics(groups: dict[str, list[dict]]) -> dict:
    groups = {g: ts for g, ts in groups.items() if len(ts) >= 2}
    if not groups:
        return {"n_groups": 0}

    decision_consistent = []   # unanimous ask / no-ask within the group
    answer_consistent   = []   # unanimous correct / wrong within the group
    axis_consistent     = []   # among asking variants, unanimous axis-hit verdict
    mixed_accuracy       = []   # group is brittle: some variants correct, some not
    swing                = []   # within-group accuracy spread (max - min, 0 or 1 per group)
    overall_correct      = []

    for ts in groups.values():
        asked = [t.get("n_asks", 0) > 0 for t in ts]
        correct = [bool(t.get("correct")) for t in ts]
        hits = [bool(t.get("first_ask_hit")) for t in ts if t.get("n_asks", 0) > 0]

        decision_consistent.append(all(asked) or not any(asked))
        answer_consistent.append(all(correct) or not any(correct))
        mixed_accuracy.append(any(correct) and not all(correct))
        swing.append(1.0 if (any(correct) and not all(correct)) else 0.0)
        if hits:
            axis_consistent.append(all(hits) or not any(hits))
        overall_correct.extend(correct)

    return {
        "n_groups":                 len(groups),
        "decision_consistency":     mean(decision_consistent),   # ↑ better: clarifies regardless of wording
        "answer_consistency":       mean(answer_consistent),     # ↑ better: same correctness across wordings
        "axis_consistency":         mean(axis_consistent) if axis_consistent else None,
        "brittle_group_rate":       mean(mixed_accuracy),        # ↓ better: fraction of groups that flip
        "accuracy_swing":           mean(swing),                 # ↓ better: within-group correctness spread
        "mean_accuracy":            mean(overall_correct),       # reference
    }


def print_report(name: str, m: dict) -> None:
    print(f"\n=== Paraphrase-Consistency: {name}  ({m.get('n_groups',0)} groups) ===")
    if not m.get("n_groups"):
        print("  (no multi-variant groups)")
        return
    print(f"  Decision consistency : {m['decision_consistency']:.1%}   (clarify-or-not stable across wordings; ↑)")
    print(f"  Answer consistency   : {m['answer_consistency']:.1%}   (same correctness across wordings; ↑)")
    if m.get("axis_consistency") is not None:
        print(f"  Axis consistency     : {m['axis_consistency']:.1%}   (asks same axis across wordings; ↑)")
    print(f"  Brittle group rate   : {m['brittle_group_rate']:.1%}   (groups that flip correctness; ↓)")
    print(f"  Accuracy swing       : {m['accuracy_swing']:.1%}   (within-group correctness spread; ↓)")
    print(f"  Mean accuracy        : {m['mean_accuracy']:.1%}   (reference)")


if __name__ == "__main__":
    # tiny self-test
    demo = {
        "g1": [{"correct": True, "n_asks": 1, "first_ask_hit": True},
               {"correct": True, "n_asks": 1, "first_ask_hit": True},
               {"correct": False, "n_asks": 0, "first_ask_hit": None}],  # brittle
        "g2": [{"correct": True, "n_asks": 1, "first_ask_hit": True},
               {"correct": True, "n_asks": 1, "first_ask_hit": True}],   # consistent
    }
    print_report("demo", consistency_metrics(demo))
