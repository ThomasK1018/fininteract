"""Headline figure for the multi-round co-evolution study: the FRONTIER-YIELD trajectory
(the abundance ceiling) per axis, plus the held-out human-probe solver trajectory.
The Elo tournament was degenerate (0 solver wins everywhere -- battle=held-out-frontier is
unresolvable by construction), so frontier yield is the valid abundance instrument, not Elo."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AXES = [("entity", "entity_scope", 0.92, "#1f77b4"),
        ("metric", "metric_definition", 0.48, "#2ca02c"),
        ("temporal", "temporal_scope", 0.15, "#ff7f0e"),
        ("recognition", "recognition_policy", 0.04, "#d62728")]
ROOT = "data/coevolve/mr"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

for name, aid, sal, col in AXES:
    try:
        st = json.load(open(f"{ROOT}/{name}/state.json"))
    except FileNotFoundError:
        continue
    rounds = sorted(st.get("rounds", []), key=lambda r: r["round"])
    xs = [r["round"] for r in rounds]
    fy = [r["frontier_yield"].get("frontier", 0) for r in rounds]
    ax1.plot(xs, fy, "-o", color=col, label=f"{name} (salience {sal})", linewidth=2)

ax1.set_title("Frontier yield per round = the abundance ceiling", fontsize=11)
ax1.set_xlabel("co-evolution round"); ax1.set_ylabel("fresh frontier instances (solver fails)")
ax1.set_xticks([0, 1, 2]); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
ax1.annotate("abundant axes sustain frontier;\nscarce axes collapse to 0",
             xy=(0.5, 0.6), xycoords="axes fraction", fontsize=8, color="gray")

# right panel: held-out human-probe solver trajectory (where a probe exists) -- flat = no transfer
for name, aid, sal, col in AXES:
    try:
        st = json.load(open(f"{ROOT}/{name}/state.json"))
    except FileNotFoundError:
        continue
    rounds = sorted(st.get("rounds", []), key=lambda r: r["round"])
    traj = [(r["round"], r["human_axishit_next_solver"]) for r in rounds
            if r.get("human_axishit_next_solver")]
    if not traj:
        continue
    xs = [t[0] + 1 for t in traj]  # S_{i+1}
    acc = [t[1].get("resolve_acc") for t in traj]
    ax2.plot(xs, acc, "-s", color=col, label=f"{name}", linewidth=2)

ax2.set_title("Held-out HUMAN-probe resolution vs round (flat = no transfer)", fontsize=11)
ax2.set_xlabel("solver checkpoint S_i"); ax2.set_ylabel("human-probe resolve acc")
ax2.set_ylim(0, 0.6); ax2.set_xticks([1, 2, 3]); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("data/results/multiround_frontier_yield.png", dpi=130)
print("wrote data/results/multiround_frontier_yield.png")
