"""v2 headline figure: the AxisHit VAL/TEST trajectory per round (non-degenerate, unlike v1),
showing entity learns+transfers while recognition stalls, and the overfit guard preserves
human transfer. Elo curves are produced separately by coevolve_elo.py."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [("entity",         "entity  w1 (single-round)", "#1f77b4", "-o"),
        ("entity_w3",      "entity  w3 (accumulate)",   "#2b6cb0", "--s"),
        ("metric",         "metric  w1 (single-round)", "#8c564b", "-o"),
        ("metric_w3",      "metric  w3 (accumulate)",   "#2ca02c", "--s"),
        ("recognition_v15","recognition (val15, train26)","#d62728", "-o")]
ROOT = "data/coevolve/mr_v2"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
for tag, label, col, style in RUNS:
    d = json.load(open(f"{ROOT}/{tag}/state.json"))
    rs = sorted(d["rounds"], key=lambda r: r["round"])
    xs = [r["round"] + 1 for r in rs]                       # S_1..S_3
    ax1.plot(xs, [r["val_axishit"] for r in rs], style, color=col, label=label, linewidth=2)
    ax2.plot(xs, [r["test_human_axishit"] for r in rs], style, color=col, label=label, linewidth=2)

ax1.axhline(0.05, color="gray", ls=":", lw=1); ax1.annotate("base ≈ 0.05", (1, 0.07), color="gray", fontsize=8)
ax1.set_title("VAL CategoryHit@1 per round (selection metric)", fontsize=11)
ax1.set_xlabel("solver checkpoint S_i"); ax1.set_ylabel("val CategoryHit@1")
ax1.set_ylim(0, 1); ax1.set_xticks([1, 2, 3]); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

ax2.axhline(0.0, color="gray", ls=":", lw=1)
ax2.set_title("TEST human-probe CategoryHit@1 (held out; guard preserves transfer)", fontsize=11)
ax2.set_xlabel("solver checkpoint S_i"); ax2.set_ylabel("human-probe CategoryHit@1")
ax2.set_ylim(0, 1); ax2.set_xticks([1, 2, 3]); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

plt.tight_layout(); plt.savefig("data/results/multiround_v2_axishit.png", dpi=130)
print("wrote data/results/multiround_v2_axishit.png")
