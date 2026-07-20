"""Render the Exp-1 (Axis-Aware ReAct vs baselines) and Exp-2 (elicitation/grounding 2x2)
tables from a finished evaluation run. No API calls -- pure re-aggregation via
evaluate.compute_metrics, so every metric (acc, IR, AxisHit@1, wrong-axis, generic-ask,
turns, ECE) is computed identically to the harness.

Usage:
  python scripts/report_exp12.py --results data/results/exp1_gpt5.jsonl [--model openai/gpt-5]
"""
import argparse, json, collections, sys
from pathlib import Path

sys.path.insert(0, "scripts")
from evaluate import compute_metrics

# Exp-1: proposed method + the four baselines the reviewer asked for.
EXP1_ORDER = [
    ("answer+search+interact", "Standard ReAct"),
    ("axis-aware",             "Axis-Aware ReAct (ours)"),
    ("always-ask",             "Always-Ask"),
    ("axis-oracle",            "Axis-Oracle"),
    ("template-oracle",        "Template-Oracle"),
]
# Exp-2: {C absent/present} x {evidence absent/oracle}
EXP2_GRID = {
    ("C absent",  "evidence absent"): "answer-only",
    ("C absent",  "oracle evidence"): "answer+search",
    ("C present", "evidence absent"): "interp-oracle",
    ("C present", "oracle evidence"): "context-oracle",
}


def load(pattern):
    import glob
    seen = {}
    for f in glob.glob(pattern):
        for l in open(f, encoding="utf-8"):
            l = l.strip()
            if l:
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    continue
                seen[(r.get("model"), r.get("mode"), r.get("instance_id"), r.get("forced_n", 0))] = r
    return list(seen.values())


def pct(x):
    return f"{100*x:5.1f}" if isinstance(x, (int, float)) else "  -  "


def main(a):
    rows = [r for r in load(a.results) if (not a.model or r.get("model") == a.model)]
    by_mode = collections.defaultdict(list)
    for r in rows:
        by_mode[r["mode"]].append(r)
    M = {m: compute_metrics(rs) for m, rs in by_mode.items()}
    model = a.model or (rows[0]["model"] if rows else "?")
    print(f"model: {model}   (modes present: {sorted(by_mode)})\n")

    print("=" * 100)
    print("EXP 1 -- Axis-Aware ReAct vs baselines")
    print("=" * 100)
    print(f"{'condition':26s} {'n':>3s} {'Acc':>6s} {'IR':>6s} {'AHit@1':>7s} "
          f"{'WrongAx':>8s} {'Generic':>8s} {'Turns':>6s} {'ECE':>6s}")
    for mode, label in EXP1_ORDER:
        m = M.get(mode)
        if not m:
            print(f"{label:26s}  --  (mode not in run)")
            continue
        ece = m.get("ece")
        print(f"{label:26s} {m['n']:3d} {pct(m['accuracy'])} {pct(m['ir_rate'])} "
              f"{pct(m['axis_hit_at1'])} {pct(m['wrong_axis_rate'])} {pct(m['generic_ask_rate'])} "
              f"{m['avg_turns']:6.1f} {(f'{ece:.3f}' if ece is not None else '  -  '):>6s}")

    print("\n" + "=" * 100)
    print("EXP 2 -- Elicitation vs grounding (2x2): accuracy (%)")
    print("=" * 100)
    def acc_of(mode):
        m = M.get(mode)
        return m["accuracy"] if m else None
    print(f"{'':12s} {'evidence absent':>18s} {'oracle evidence':>18s}")
    for crow in ("C absent", "C present"):
        cells = []
        for ecol in ("evidence absent", "oracle evidence"):
            mode = EXP2_GRID[(crow, ecol)]
            cells.append(acc_of(mode))
        print(f"{crow:12s} {pct(cells[0]):>18s} {pct(cells[1]):>18s}")
    # deltas
    aa = acc_of("answer-only"); asrch = acc_of("answer+search")
    io = acc_of("interp-oracle"); co = acc_of("context-oracle")
    def d(x, y):
        return f"{100*(x-y):+.1f}" if (x is not None and y is not None) else "  -  "
    print("\nGrounding value (add oracle evidence):")
    print(f"  C absent:   answer+search - answer-only     = {d(asrch, aa)}")
    print(f"  C present:  context-oracle - interp-oracle   = {d(co, io)}")
    print("Elicitation value (add resolved interpretation C):")
    print(f"  evidence absent: interp-oracle - answer-only = {d(io, aa)}")
    print(f"  oracle evidence: context-oracle - answer+search = {d(co, asrch)}")
    print("\nInterpretation: if adding C (elicitation) moves accuracy far more than adding "
          "oracle evidence (grounding), elicitation -- not evidence access -- is the bottleneck.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--model", default=None)
    main(p.parse_args())
