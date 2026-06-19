"""Roll a served model's result files into a paste-ready tab:main LaTeX row.

Reads data/results/eval_open_<label>.jsonl (answer-only / +search / +interact)
and data/results/eval_ceiling_<label>.jsonl (context-oracle), and prints:
  - Ans-only / +Search / +Interact accuracy
  - IR (% asked), AxisHit@1, ECE (5-bin) for the interact mode
  - the context-oracle ceiling
plus a LaTeX line matching the column order of tab:main.

Usage:
  python experiments/gpu_eval/rollup_open.py qwen3-8b [qwen3-14b ...]
"""
import json, sys, glob
from pathlib import Path

def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] if Path(path).exists() else []

def acc(rows):
    n = len(rows); return (100*sum(1 for r in rows if r.get("correct"))/n if n else float("nan")), n

def ece(rows, bins=5):
    pts = [(r["confidence"]/100.0, bool(r.get("correct"))) for r in rows
           if isinstance(r.get("confidence"), (int, float))]
    if not pts: return float("nan")
    tot = len(pts); e = 0.0
    for b in range(bins):
        lo, hi = b/bins, (b+1)/bins
        bucket = [(c, ok) for c, ok in pts if (lo < c <= hi or (b == 0 and c == 0))]
        if not bucket: continue
        conf = sum(c for c, _ in bucket)/len(bucket)
        a = sum(1 for _, ok in bucket if ok)/len(bucket)
        e += (len(bucket)/tot)*abs(a - conf)
    return e

def row(label):
    open_rows = load(f"data/results/eval_open_{label}.jsonl")
    ceil_rows = load(f"data/results/eval_ceiling_{label}.jsonl")
    by = {m: [r for r in open_rows if r.get("mode") == m] for m in
          ("answer-only", "answer+search", "answer+search+interact")}
    ao, _   = acc(by["answer-only"])
    sr, _   = acc(by["answer+search"])
    it_rows = by["answer+search+interact"]
    ir_acc, n_it = acc(it_rows)
    asked   = [r for r in it_rows if (r.get("n_asks") or 0) > 0]
    IR      = 100*len(asked)/n_it if n_it else float("nan")
    ahit    = (sum(r.get("axis_hit_rate", 0.0) for r in asked)/len(asked)) if asked else float("nan")
    e       = ece(it_rows)
    cl, _   = acc(ceil_rows)
    print(f"\n== {label} (n_interact={n_it}) ==")
    print(f"  Ans-only={ao:.1f}  +Search={sr:.1f}  +Interact={ir_acc:.1f}  "
          f"IR={IR:.0f}  AxisHit@1={ahit:.2f}  ECE={e:.2f}  Ceiling={cl:.1f}")
    def ah(x): return f"{x:.2f}".lstrip("0") if x == x else "--"
    print(f"  LaTeX: {label:14s} & {ao:.1f} & {sr:.1f} & {ir_acc:.1f} & {IR:.0f} & {ah(ahit)} & {ah(e)} \\\\")

if __name__ == "__main__":
    labels = sys.argv[1:] or [Path(p).stem.replace("eval_open_", "")
                              for p in glob.glob("data/results/eval_open_*.jsonl")]
    for lb in labels: row(lb)
