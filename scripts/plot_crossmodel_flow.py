"""Cross-model interpretability figure (four models, 4B -> 35B, dense + MoE).

Panel (a): depth-flow separation (ambiguous - disambiguated), normalized to each model's peak
           and plotted vs depth fraction -> the late-layer peak is architecture/scale invariant.
Panel (b): axis-decoding linear-probe accuracy vs depth for the two MoE models (the leak-proof,
           quantitative "model decodes the ambiguity TYPE" signal), against the majority baseline.

Reads results/flow/<model>/flow_report*.json (depth-flow, all four) and
results/flow/<model>/probes_bare.json (axis-decoding accuracy, MoE models only).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 160, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

ROOT = Path(__file__).resolve().parent.parent / "results" / "flow"

# (label, depth-flow report, color)
DEPTH = [
    ("Qwen3-4B (dense, 37L)",        ROOT / "baseline_qwen3-4b-instruct-2507/flow_report.json", "C0"),
    ("Qwen3.5-4B (hybrid, 33L)",     ROOT / "qwen3.5-4b/flow_report_qwen35.json",                "C1"),
    ("Qwen3-30B-A3B (MoE, 49L)",     ROOT / "qwen3-30b-a3b-instruct-2507/flow_report.json",      "C2"),
    ("Qwen3.5-35B-A3B (MoE, 41L)",   ROOT / "qwen3.5-35b-a3b/flow_report.json",                  "C3"),
]
# (label, probes_bare.json, color) — only the MoE runs produced linear-probe accuracy
PROBE = [
    ("Qwen3-30B-A3B (MoE, 49L)",   ROOT / "qwen3-30b-a3b-instruct-2507/probes_bare.json", "C2"),
    ("Qwen3.5-35B-A3B (MoE, 41L)", ROOT / "qwen3.5-35b-a3b/probes_bare.json",             "C3"),
]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.4))

for name, path, c in DEPTH:
    df = json.load(open(path))["depth_flow"]
    sep = np.array(df["ambiguous_vs_disambiguated_separation_by_layer"], dtype=float)
    x = np.linspace(0, 1, len(sep))
    a1.plot(x, sep / sep.max(), color=c, lw=1.8, label=name)
    a1.axvline(df["peak_separation_layer"] / (len(sep) - 1), color=c, ls=":", lw=1, alpha=0.6)
a1.set_xlabel("depth (fraction of layers)")
a1.set_ylabel("ambiguous − disambiguated\n(normalized to peak)")
a1.set_title("(a) Ambiguity emerges in the final layers")
a1.legend(fontsize=7, loc="upper left"); a1.axhline(0, color="k", lw=0.5)

baseline = None
for name, path, c in PROBE:
    ad = json.load(open(path))["axis_decoding"]
    acc = np.array(ad["per_layer_acc"], dtype=float)
    baseline = ad["baseline"]
    x = np.linspace(0, 1, len(acc))
    pk = int(np.argmax(acc))
    a2.plot(x, acc, color=c, lw=1.8, label=f"{name} (peak {acc[pk]:.2f})")
    a2.scatter([x[pk]], [acc[pk]], color=c, s=20, zorder=5)
a2.axhline(baseline, color="gray", ls="--", lw=1, label=f"majority baseline ({baseline:.2f})")
a2.set_xlabel("depth (fraction of layers)")
a2.set_ylabel("entity-vs-metric probe accuracy")
a2.set_title("(b) The ambiguity TYPE is linearly decodable")
a2.legend(fontsize=7, loc="lower right"); a2.set_ylim(0.5, 0.85)

fig.tight_layout()
out = Path(__file__).resolve().parent.parent / "paper" / "fig_crossmodel_depthflow.png"
fig.savefig(out, bbox_inches="tight"); print("wrote", out)
