"""Per-category clarification-capability bar chart (paper/fig_radar.png).

Grouped bar chart of per-category targeting (touch AC@1) on the four populated
ambiguity categories, for a frontier closed model, a weaker closed model, and the
strongest open model. The zero-height recognition-policy group makes the universal
blind spot immediate (all models collapse there while saturating the others).

Values are the touch AC@1 reported in Table~\ref{tab:skills} (computed by
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

AXES = ["Entity scope", "Metric definition", "Recognition policy", "Temporal scope"]
# touch AC@1 per category, from Table tab:skills
SERIES = {
    "GPT-5":            [0.89, 0.92, 0.00, 1.00],
    "GPT-4o":           [1.00, 0.97, 0.00, 1.00],
    "Qwen3.5-35B-A3B":  [0.85, 0.87, 0.00, 1.00],
}
COLORS = {"GPT-5": "#1f3a5f", "GPT-4o": "#e76f51", "Qwen3.5-35B-A3B": "#2a9d8f"}


def main(a):
    models = list(SERIES.keys())
    y = np.arange(len(AXES))
    height = 0.26

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.3, 2.8))
    for i, name in enumerate(models):
        offs = ((len(models) - 1) / 2 - i) * height
        ax.barh(y + offs, SERIES[name], height, color=COLORS[name], label=name)

    # make the shared recognition-policy blind spot explicit (all bars are 0 there)
    ax.text(0.015, y[2], "0", ha="left", va="center", fontsize=8.5, color="#555")

    ax.set_yticks(y)
    ax.set_yticklabels(AXES, fontsize=9.5)
    ax.invert_yaxis()  # first category on top
    ax.set_xlabel("Targeting (AC@1)", fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=3,
              fontsize=8.5, frameon=False, columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    fig.savefig(str(a.out).replace(".png", ".pdf"), bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="paper/fig_radar.png")
    main(p.parse_args())
