"""Per-axis performance matrix (paper/fig_axisperf.png).

Two heatmaps over model x ambiguity-axis:
  (a) Targeting  = AxisHit@1 (touch form), from Table tab:skills.
  (b) Resolution = +interact accuracy, computed from data/results/eval_*.jsonl.

The juxtaposition shows the paper's central dissociation at model x axis granularity:
models target the correct axis almost always yet resolve it rarely, and recognition
policy collapses on both. filing_vintage has no corpus instances (n=0) and is marked
n/a. Resolution is computed here; targeting values are the published tab:skills numbers.

Usage:
  python scripts/plot_axis_performance.py
"""
import argparse, glob, json, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

MODELS = ["gpt-5", "gpt-4o", "gpt-5-mini", "qwen3-30b-a3b", "qwen3p5-35b-a3b"]
MLABEL = ["GPT-5", "GPT-4o", "GPT-5-mini", "Qwen3-30B", "Qwen3.5-35B"]
AXES = ["entity_scope", "metric_definition", "recognition_policy", "temporal_scope", "filing_vintage"]
ALABEL = ["Entity", "Metric", "Recognition", "Temporal", "Filing\nvintage"]
# Targeting (touch AxisHit@1) from Table tab:skills; filing_vintage has no data.
TARGET = {
    "entity_scope":       [0.89, 1.00, 0.99, 0.93, 0.85],
    "metric_definition":  [0.92, 0.97, 1.00, 0.96, 0.87],
    "recognition_policy": [0.00, 0.00, 0.00, 0.00, 0.00],
    "temporal_scope":     [1.00, 1.00, 1.00, 1.00, 1.00],
    "filing_vintage":     [None]*5,
}
TEAL = LinearSegmentedColormap.from_list("teal", ["#f5faf8", "#2a9d8f", "#14503f"])


def load(pats):
    seen = {}
    for pat in pats:
        for f in glob.glob(pat):
            for l in open(f, encoding="utf-8"):
                l = l.strip()
                if not l:
                    continue
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    continue
                seen[(r.get("model"), r.get("mode"), r.get("instance_id"), r.get("forced_n", 0))] = r
    return list(seen.values())


def resolution_matrix(rows):
    cell = collections.defaultdict(lambda: {"c": 0, "n": 0})
    for r in rows:
        if r.get("model") not in MODELS or r.get("mode") != "answer+search+interact" or r.get("forced_n", 0):
            continue
        ax = r.get("axes") or []
        if not ax:
            continue
        d = cell[(r["model"], ax[0])]
        d["n"] += 1
        d["c"] += bool(r.get("correct"))
    res, ncol = {}, {}
    for a in AXES:
        res[a] = []
        ncol[a] = 0
        for m in MODELS:
            d = cell.get((m, a))
            ncol[a] = d["n"] if d else 0
            res[a].append(100 * d["c"] / d["n"] if d and d["n"] else None)
    return res, ncol


def draw(ax, mat, ncol, title, vmax, fmt, note_sparse):
    grid = np.array([[np.nan if mat[a][j] is None else mat[a][j] for a in AXES]
                     for j in range(len(MODELS))], dtype=float)
    ax.imshow(np.ma.masked_invalid(grid), cmap=TEAL, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(AXES)))
    heads = []
    for a, lab in zip(AXES, ALABEL):
        n = ncol[a]
        tag = f"\n(n={n})" if n else "\n(n=0)"
        heads.append(lab + tag)
    ax.set_xticklabels(heads, fontsize=8)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MLABEL, fontsize=8.5)
    for j in range(len(MODELS)):
        for i, a in enumerate(AXES):
            v = mat[a][j]
            if v is None:
                ax.text(i, j, "n/a", ha="center", va="center", fontsize=7.5, color="#999")
                ax.add_patch(plt.Rectangle((i-0.5, j-0.5), 1, 1, color="#eee", zorder=0))
            else:
                tc = "white" if v > vmax*0.55 else "#222"
                ax.text(i, j, fmt(v), ha="center", va="center", fontsize=8, color=tc)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(np.arange(-.5, len(AXES), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(MODELS), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)


def main(a):
    rows = load(a.results)
    res, ncol = resolution_matrix(rows)
    print("Resolution (+interact accuracy %):")
    for ax_ in AXES:
        print(f"  {ax_:20s}", ["--" if v is None else f"{v:.1f}" for v in res[ax_]])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 3.0))
    draw(axL, TARGET, ncol, "(a) Targeting: AxisHit@1 (touch)", 1.0, lambda v: f"{v:.2f}", True)
    draw(axR, res, ncol, "(b) Resolution: +interact accuracy", 100.0, lambda v: f"{v:.0f}", True)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", default=["data/results/eval_*.jsonl"])
    p.add_argument("--out", default="paper/fig_axisperf.png")
    main(p.parse_args())
