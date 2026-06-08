"""Cross-model depth-flow overlay: the robust, replicated finding.
Normalizes x by depth fraction (different layer counts) and y by per-model peak
(different magnitudes) so the two architectures are directly comparable."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 160, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

BASE = Path("/tmp/flow_results/fininteract_flow_results/results")
MODELS = [
    ("Qwen3-4B-Instruct (dense, 37L)", BASE / "baseline_qwen3-4b-instruct-2507/flow_report.json", "C0"),
    ("Qwen3.5-4B (hybrid-attn, 33L)", BASE / "qwen3.5-4b/flow_report_qwen35.json", "C3"),
]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.3))

for name, path, c in MODELS:
    df = json.load(open(path))["depth_flow"]
    sep = np.array(df["ambiguous_vs_disambiguated_separation_by_layer"])
    x = np.linspace(0, 1, len(sep))
    a1.plot(x, sep / sep.max(), color=c, lw=1.8, label=name)
    pk = df["peak_separation_layer"] / (len(sep) - 1)
    a1.axvline(pk, color=c, ls=":", lw=1, alpha=0.7)

    ad = df["axis_distinguishability_entity_vs_metric"]
    e = np.array(ad["entity_curve"]); m = np.array(ad["metric_curve"])
    norm = max(abs(e).max(), abs(m).max())
    xx = np.linspace(0, 1, len(e))
    a2.plot(xx, e / norm, color=c, lw=1.8, label=f"{name.split(' (')[0]} — entity")
    a2.plot(xx, m / norm, color=c, lw=1.8, ls="--", label=f"{name.split(' (')[0]} — metric")

a1.set_xlabel("depth (fraction of layers)"); a1.set_ylabel("ambiguous − disambiguated\n(normalized to peak)")
a1.set_title("(a) Ambiguity emerges in the final layers"); a1.legend(fontsize=7.5, loc="upper left")
a1.axhline(0, color="k", lw=0.5)

a2.set_xlabel("depth (fraction of layers)"); a2.set_ylabel("axis-discriminative projection\n(normalized)")
a2.set_title("(b) Entity vs. metric split, opposite sign"); a2.legend(fontsize=7, loc="upper left")
a2.axhline(0, color="k", lw=0.5)

fig.tight_layout()
out = Path("/tmp/flow_results/fig_crossmodel_depthflow.png")
fig.savefig(out, bbox_inches="tight"); print("wrote", out)
