"""Per-axis clarification-capability radar (paper/fig_radar.png).

Multi-dimensional capability profile: AxisHit@1 (validated touch form) on each of the
four populated ambiguity axes, for a frontier closed model, a weaker closed model, and
the strongest open model. The shape makes the universal recognition-policy blind spot
immediate (all models collapse to the center on that axis while saturating the others).

Values are the touch AxisHit@1 reported in Table~\ref{tab:skills} (computed by
scripts/recompute_axishit_peraxis.py). No API calls.

Usage:
  python scripts/plot_capability_radar.py
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

AXES = ["Entity\nscope", "Metric\ndefinition", "Recognition\npolicy", "Temporal scope"]
# touch AxisHit@1 per axis, from Table tab:skills
SERIES = {
    "GPT-5":            [0.89, 0.92, 0.00, 1.00],
    "GPT-4o":           [1.00, 0.97, 0.00, 1.00],
    "Qwen3.5-35B-A3B":  [0.85, 0.87, 0.00, 1.00],
}
COLORS = {"GPT-5": "#1f3a5f", "GPT-4o": "#e76f51", "Qwen3.5-35B-A3B": "#2a9d8f"}


def main(a):
    n = len(AXES)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(3.25, 3.25), subplot_kw=dict(polar=True))
    for name, vals in SERIES.items():
        v = vals + vals[:1]
        ax.plot(angles, v, "o-", linewidth=2, markersize=4, color=COLORS[name], label=name)
        ax.fill(angles, v, color=COLORS[name], alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(AXES, fontsize=9.5)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "0.5", "", "1.0"], fontsize=9, color="#555")
    ax.set_rlabel_position(135)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3,
              fontsize=9, frameon=False, columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight"); fig.savefig(str(a.out).replace(".png", ".pdf"), bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="paper/fig_radar.png")
    main(p.parse_args())
