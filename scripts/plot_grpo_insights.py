"""GRPO/RLVR training-insight figure, built from the aggregates in
results/grpo/RESULTS_REPORT.md (raw per-step logs live on the GPU box).

Panel (a): the RLVR ladder -- separates the LEAK-PROOF learning signal
           (AxisHit@1, Interaction) from leak-confounded Accuracy across
           base -> SFT -> KTO -> GRPO (n=51 held-out).
Panel (b): true multi-turn GRPO (verl, 4B, 6 GPUs) -- reward and clarifying
           turns rise together over 15 steps -> the axis-aware reward is
           teaching the policy to ask more.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 160, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

# --- Panel (a): RLVR ladder (RESULTS_REPORT.md, n=51) ---
stages = ["base", "SFT", "KTO", "GRPO"]
acc      = [84.3, 88.2, 84.3, 88.2]   # leak-confounded (search returns gold span)
axishit  = [25.0, 68.6, 70.6, 66.7]   # leak-proof
interact = [54.9, 100.0, 100.0, 100.0]  # leak-proof

# --- Panel (b): verl 4B multi-turn run (logged steps) ---
step       = [1, 7, 13, 15]
reward_mean= [0.55, 1.08, 1.11, 1.08]
turns_mean = [5.7, 6.1, 8.6, 7.3]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.6))

x = np.arange(len(stages)); w = 0.27
a1.bar(x - w, axishit,  w, label="AxisHit@1 (leak-proof)", color="C0")
a1.bar(x,      interact, w, label="Interaction % (leak-proof)", color="C2")
a1.bar(x + w, acc,      w, label="Accuracy (leak-confounded)", color="C3", alpha=0.55, hatch="//")
a1.set_xticks(x); a1.set_xticklabels(stages)
a1.set_ylabel("%"); a1.set_ylim(0, 105)
a1.set_title("(a) RLVR ladder: SFT does the heavy lifting")
a1.legend(fontsize=7, loc="lower right")

l1, = a2.plot(step, reward_mean, "o-", color="C0", lw=1.8, label="reward mean")
a2.set_xlabel("GRPO step"); a2.set_ylabel("reward mean", color="C0")
a2.tick_params(axis="y", labelcolor="C0"); a2.set_ylim(0, 1.4)
a2b = a2.twinx()
l2, = a2b.plot(step, turns_mean, "s--", color="C1", lw=1.8, label="clarifying turns")
a2b.set_ylabel("turns / episode", color="C1"); a2b.tick_params(axis="y", labelcolor="C1")
a2b.set_ylim(4, 9); a2b.grid(False)
a2.set_title("(b) True multi-turn GRPO (verl, 4B): reward & asking rise")
a2.legend([l1, l2], ["reward mean", "clarifying turns"], fontsize=7, loc="lower right")

fig.tight_layout()
out = "paper/fig_grpo_insights.png"
fig.savefig(out, bbox_inches="tight"); print("wrote", out)

# --- printed insights ---
print("\n=== TRAINING INSIGHTS (from reported aggregates) ===")
print(f"AxisHit@1  base->SFT: {axishit[0]}->{axishit[1]}  (+{axishit[1]-axishit[0]:.1f}); "
      f"SFT->GRPO: {axishit[1]}->{axishit[3]} ({axishit[3]-axishit[1]:+.1f})")
print(f"Interaction base->SFT: {interact[0]}->{interact[1]}  (+{interact[1]-interact[0]:.1f}); saturates at 100 thereafter")
print(f"Accuracy base->GRPO: {acc[0]}->{acc[3]} (+{acc[3]-acc[0]:.1f})  <- tiny gap = leak-confounded, ignore as learning signal")
r = np.corrcoef(reward_mean, turns_mean)[0, 1]
print(f"verl 4B: reward {reward_mean[0]}->{reward_mean[-1]}, turns {turns_mean[0]}->{turns_mean[-1]}; "
      f"reward-vs-turns corr = {r:.2f}")
