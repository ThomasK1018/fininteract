"""Render the FinInteract dataset-distribution figure (paper/fig_dataset_stats.png).

Two panels:
  (a) primary-axis share (bar), the task/type distribution.
  (b) disambiguation-entropy H0 distribution (bar over the discrete H0 values),
      the difficulty/ambiguity-level distribution.

Reads the frozen snapshot data/final/fininteract_v1.jsonl. No API calls.

Usage:
  python scripts/plot_dataset_stats.py
"""
import json, collections, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AXIS_ORDER = ["entity_scope", "metric_definition", "recognition_policy",
              "temporal_scope", "filing_vintage"]
AXIS_LABEL = {"entity_scope": "Entity\nscope", "metric_definition": "Metric\ndefinition",
              "recognition_policy": "Recognition\npolicy", "temporal_scope": "Temporal\nscope",
              "filing_vintage": "Filing\nvintage"}
NAVY, TEAL = "#1f3a5f", "#2a9d8f"


def main(a):
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    n = len(rows)
    prim = collections.Counter(r["axes"][0] for r in rows if r.get("axes"))
    h0 = [r["h0"] for r in rows if isinstance(r.get("h0"), (int, float))]
    h0c = collections.Counter(round(x, 2) for x in h0)

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.1))

    # (a) primary-axis share
    counts = [prim.get(ax, 0) for ax in AXIS_ORDER]
    labels = [AXIS_LABEL[ax] for ax in AXIS_ORDER]
    bars = ax1.bar(range(len(AXIS_ORDER)), counts, color=NAVY, width=0.66)
    for b, c in zip(bars, counts):
        ax1.text(b.get_x() + b.get_width() / 2, c + 1.2,
                 f"{c}\n{100*c/n:.0f}%", ha="center", va="bottom", fontsize=9, linespacing=0.95)
    ax1.set_xticks(range(len(AXIS_ORDER)))
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylabel("Instances")
    ax1.set_ylim(0, max(counts) * 1.28)
    ax1.set_title("(a) Primary ambiguity category", fontsize=11)

    # (b) H0 distribution
    xs = sorted(h0c)
    ys = [h0c[x] for x in xs]
    bars2 = ax2.bar([str(x) for x in xs], ys, color=TEAL, width=0.6)
    for b, c in zip(bars2, ys):
        ax2.text(b.get_x() + b.get_width() / 2, c + 1.2, str(c), ha="center", va="bottom", fontsize=9)
    ax2.set_xlabel("Disambiguation entropy $H_0$ (bits)")
    ax2.set_ylabel("Instances")
    ax2.set_ylim(0, max(ys) * 1.2)
    ax2.set_title("(b) Difficulty distribution", fontsize=11)

    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"n={n}  axes={dict(prim)}  H0={dict(h0c)}")
    print("wrote", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/final/fininteract_v1.jsonl")
    p.add_argument("--out", default="paper/fig_dataset_stats.png")
    main(p.parse_args())
