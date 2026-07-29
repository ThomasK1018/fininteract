"""Statistical figures for the headline findings (paper/fig_bottleneck.png,
paper/fig_lever.png, paper/fig_humanllm.png).

All values are the published numbers in the paper tables (tab:ceiling, tab:main,
tab:crossvendor, Finding 8), so the figures match the tables exactly. No API calls.

Usage:
  python scripts/plot_findings.py
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

NAVY, TEAL, ORANGE, GREY = "#1f3a5f", "#2a9d8f", "#e76f51", "#8a8f98"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def fig_bottleneck(out):
    # tab:ceiling: free-interaction accuracy vs context-oracle ceiling
    models = ["GPT-5", "GPT-4o", "GPT-5-mini", "Qwen3-30B", "Qwen3.5-35B"]
    interact = [20.2, 4.6, 0.0, 11.6, 28.9]
    oracle = [95.4, 93.1, 95.4, 90.2, 91.9]
    y = np.arange(len(models))[::-1]
    fig, ax = plt.subplots(figsize=(3.25, 2.6))
    for yi, a, o in zip(y, interact, oracle):
        ax.plot([a, o], [yi, yi], color=GREY, lw=2, zorder=1)
        ax.text((a + o) / 2, yi + 0.16, f"+{o-a:.0f}", ha="center", va="bottom",
                fontsize=9, color="#444")
    ax.scatter(interact, y, s=70, color=ORANGE, zorder=3, label="Free interaction")
    ax.scatter(oracle, y, s=70, color=TEAL, zorder=3, label="Context-oracle ceiling")
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlim(-3, 105)
    ax.set_xlabel("Accuracy (\\%)".replace("\\%", "%"))
    ax.set_title("The elicitation gap", fontsize=10.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
              fontsize=9, frameon=False, handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); fig.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight"); print("wrote", out)


def fig_lever(out):
    # Delta = +Interact minus +Search accuracy (tab:main full-scale + tab:crossvendor)
    data = [("GPT-5", 13.8), ("Gemini-3.5", 24.0), ("GPT-5-mini", 0.0), ("GLM-5.2", 0.0),
            ("Claude-Sonnet-5", -2.0), ("DeepSeek-V4", -4.0), ("Kimi-K3", -4.0),
            ("GPT-5.6", -6.0), ("Qwen3.5-35B", -6.9), ("GPT-4o", -7.5),
            ("Grok-4.5", -8.0), ("Qwen3-30B", -16.7)]
    data.sort(key=lambda t: t[1])
    names = [d[0] for d in data]
    vals = [d[1] for d in data]
    colors = [TEAL if v > 0 else (GREY if v == 0 else ORANGE) for v in vals]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(3.25, 3.7))
    ax.barh(y, vals, color=colors, height=0.66)
    ax.axvline(0, color="#333", lw=0.8)
    for yi, v in zip(y, vals):
        ax.text(v + (0.4 if v >= 0 else -0.4), yi, f"{v:+.0f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Interaction lever (points)")
    ax.set_xlim(-21, 29)
    ax.set_title("Interaction helps only 2 of 12 models", fontsize=10.5)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); fig.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight"); print("wrote", out)


def fig_humanllm(out):
    # Accuracy vs AC@1 (interact): tab:main + Finding 8 human row
    pts = [("GPT-5", 0.86, 20.2, NAVY), ("GPT-4o", 0.94, 4.6, NAVY),
           ("GPT-5-mini", 0.94, 0.0, NAVY), ("Qwen3-30B", 0.90, 11.6, NAVY),
           ("Qwen3.5-35B", 0.81, 28.9, NAVY), ("Human", 0.68, 30.0, ORANGE)]
    fig, ax = plt.subplots(figsize=(3.25, 2.9))
    for name, ah, acc, col in pts:
        marker = "*" if name == "Human" else "o"
        size = 320 if name == "Human" else 90
        ax.scatter(ah, acc, s=size, color=col, marker=marker, zorder=3,
                   edgecolor="white", linewidth=0.8)
        dx = 0.006 if name != "GPT-4o" else -0.006
        ha = "left" if name != "GPT-4o" else "right"
        ax.annotate(name, (ah, acc), xytext=(ah + dx, acc + 1.1), fontsize=9, ha=ha)
    ax.set_xlabel("Targeting: AC@1")
    ax.set_ylabel("Resolution: accuracy (\\%)".replace("\\%", "%"))
    ax.set_title("Asking on target does not imply resolving", fontsize=10.5)
    ax.set_xlim(0.6, 1.0); ax.set_ylim(-2, 34)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); fig.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight"); print("wrote", out)


def main(a):
    fig_bottleneck(a.bottleneck)
    fig_lever(a.lever)
    fig_humanllm(a.humanllm)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bottleneck", default="paper/fig_bottleneck.png")
    p.add_argument("--lever", default="paper/fig_lever.png")
    p.add_argument("--humanllm", default="paper/fig_humanllm.png")
    main(p.parse_args())
