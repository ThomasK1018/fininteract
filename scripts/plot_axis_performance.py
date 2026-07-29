"""Per-axis performance matrix (paper/fig_axisperf.png).

Two heatmaps over model x ambiguity-axis:
  (a) Targeting  = AC@1 (touch form).
  (b) Resolution = +interact accuracy.

Two model blocks, separated in the figure:
  - full scale (n=173): GPT ladder + Qwen MoE; targeting from Table tab:skills.
  - proprietary pilot (n=50): the seven-vendor panel; targeting recomputed into
    data/results/axishit_peraxis_cv.json (scripts/recompute_axishit_peraxis.py).
Resolution is computed here from data/results/{eval_*,or_eval_20}.jsonl. For the
pilot block only entity and metric have enough instances (n=29/19); recognition,
temporal, and filing_vintage are n<=1 and shown n/a.

Usage:
  python scripts/plot_axis_performance.py
"""
import argparse, glob, json, collections, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
from matplotlib.colors import LinearSegmentedColormap

AXES = ["entity_scope", "metric_definition", "recognition_policy", "temporal_scope", "filing_vintage"]
ALABEL = ["Entity", "Metric", "Recognition", "Temporal", "Filing\nvintage"]

FULL = ["gpt-5", "gpt-4o", "gpt-5-mini", "qwen3-30b-a3b", "qwen3p5-35b-a3b"]
FULL_L = ["GPT-5", "GPT-4o", "GPT-5-mini", "Qwen3-30B", "Qwen3.5-35B"]
CV = ["anthropic/claude-sonnet-5", "google/gemini-3.5-flash", "x-ai/grok-4.5",
      "deepseek/deepseek-v4-flash", "z-ai/glm-5.2", "moonshotai/kimi-k3", "openai/gpt-5.6-sol"]
CV_L = ["Claude-Sonnet-5", "Gemini-3.5", "Grok-4.5", "DeepSeek-V4", "GLM-5.2", "Kimi-K3", "GPT-5.6"]
MODELS = FULL + CV
MLABEL = FULL_L + CV_L
# full-scale targeting (touch AC@1) from Table tab:skills
TARGET_FULL = {
    "entity_scope":       [0.89, 1.00, 0.99, 0.93, 0.85],
    "metric_definition":  [0.92, 0.97, 1.00, 0.96, 0.87],
    "recognition_policy": [0.00, 0.00, 0.00, 0.00, 0.00],
    "temporal_scope":     [1.00, 1.00, 1.00, 1.00, 1.00],
    "filing_vintage":     [None]*5,
}
# axes with enough pilot-block instances to report (others -> n/a)
CV_RELIABLE = {"entity_scope", "metric_definition"}
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


def resolution(rows):
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
    res = {}
    for a in AXES:
        res[a] = []
        for m in MODELS:
            d = cell.get((m, a))
            ok = (m in FULL) or (a in CV_RELIABLE)
            res[a].append(100 * d["c"] / d["n"] if (d and d["n"] and ok) else None)
    return res


def targeting(cv_json):
    tgt = {a: list(TARGET_FULL[a]) for a in AXES}
    cvd = json.loads(open(cv_json, encoding="utf-8").read()) if os.path.exists(cv_json) else {}
    for a in AXES:
        for m in CV:
            v = cvd.get(m, {}).get(a, {}).get("touch") if a in CV_RELIABLE else None
            tgt[a].append(v)
    return tgt


def draw(ax, mat, title, vmax, fmt):
    grid = np.array([[np.nan if mat[a][j] is None else mat[a][j] for a in AXES]
                     for j in range(len(MODELS))], dtype=float)
    ax.imshow(np.ma.masked_invalid(grid), cmap=TEAL, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(AXES)))
    ax.set_xticklabels(ALABEL, fontsize=9)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MLABEL, fontsize=9)
    for j in range(len(MODELS)):
        for i, a in enumerate(AXES):
            v = mat[a][j]
            if v is None:
                ax.text(i, j, "n/a", ha="center", va="center", fontsize=9, color="#aaa")
                ax.add_patch(plt.Rectangle((i-0.5, j-0.5), 1, 1, color="#eee", zorder=0))
            else:
                tc = "white" if v > vmax*0.55 else "#222"
                ax.text(i, j, fmt(v), ha="center", va="center", fontsize=9, color=tc)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(np.arange(-.5, len(AXES), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(MODELS), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.0)
    ax.tick_params(which="minor", length=0)
    ax.axhline(len(FULL)-0.5, color="#333", lw=1.8)  # divider: full-scale | pilot


def main(a):
    rows = load(a.results)
    res = resolution(rows)
    tgt = targeting(a.cv_json)
    print("Resolution (+interact accuracy %):")
    for ax_ in AXES:
        print(f"  {ax_:20s}", ["--" if v is None else f"{v:.0f}" for v in res[ax_]])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.8, 3.05))
    draw(axL, tgt, "(a) Targeting: AC@1 (touch)", 1.0, lambda v: f"{v:.2f}")
    draw(axR, res, "(b) Resolution: +interact accuracy", 100.0, lambda v: f"{v:.0f}")
    axR.set_yticklabels([])
    fig.text(0.5, 0.965, "Full scale (n=173, top)   |   Proprietary pilot (n=50, bottom)",
             ha="center", fontsize=9, color="#444")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(a.out, dpi=200, bbox_inches="tight"); fig.savefig(str(a.out).replace(".png", ".pdf"), bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", default=["data/results/eval_*.jsonl", "data/results/or_eval_20.jsonl"])
    p.add_argument("--cv-json", default="data/results/axishit_peraxis_cv.json")
    p.add_argument("--out", default="paper/fig_axisperf.png")
    main(p.parse_args())
