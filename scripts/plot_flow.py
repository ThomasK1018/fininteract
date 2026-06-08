"""
Paper-ready figures for the flow studies (CPU). Reads the report produced by analyze_flow.py
(and optionally the raw genflow.npz for the onset histogram), writes PNGs.

Figures:
  fig_depth_separation.png   line: ambiguous-vs-disambiguated projection gap per layer (peak = where
                             the model separates under-specified from disambiguated questions)
  fig_axis_depth_curves.png  per-axis ambiguity-projection curves across layers
  fig_entity_vs_metric.png   entity vs metric projection onto the axis-discriminative direction
  fig_generation_heatmap.png layer x generated-token heatmap of the ambiguity score (+ onset line)
  fig_recognition_onset.png  histogram: tokens by which the internal ambiguity peak precedes the
                             spoken clarification (needs --genflow)

Usage:
  python src/plot_flow.py --report data/interp/flow_report.json \
      --genflow data/interp/genflow.npz --outdir data/interp/figs
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight"})


def _save(fig, path):
    fig.savefig(path); plt.close(fig); print(f"  wrote {path}")


def depth_figs(df: dict, outdir: Path):
    sep = df.get("ambiguous_vs_disambiguated_separation_by_layer")
    if sep:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(range(len(sep)), sep, marker="o", ms=3)
        pk = df.get("peak_separation_layer", int(np.argmax(sep)))
        ax.axvline(pk, color="crimson", ls="--", lw=1, label=f"peak L{pk}")
        ax.set_xlabel("layer"); ax.set_ylabel("ambiguous − disambiguated\nprojection")
        ax.set_title("Depth-flow: where ambiguity emerges"); ax.legend()
        _save(fig, outdir / "fig_depth_separation.png")

    pac = df.get("per_axis_depth_curves") or {}
    if pac:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for ax_name, d in pac.items():
            ax.plot(range(len(d["curve"])), d["curve"], label=f"{ax_name} (n={d['n']})", lw=1.5)
        ax.set_xlabel("layer"); ax.set_ylabel("ambiguity projection")
        ax.set_title("Per-axis depth curves"); ax.legend(fontsize=8)
        _save(fig, outdir / "fig_axis_depth_curves.png")

    ad = df.get("axis_distinguishability_entity_vs_metric")
    if ad:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(ad["entity_curve"], label="entity_scope", lw=1.5)
        ax.plot(ad["metric_curve"], label="metric_definition", lw=1.5)
        ax.axvline(ad["peak_separation_layer"], color="crimson", ls="--", lw=1,
                   label=f"peak L{ad['peak_separation_layer']}")
        ax.set_xlabel("layer"); ax.set_ylabel("axis-discriminative projection")
        ax.set_title("Axis distinguishability: entity vs metric"); ax.legend(fontsize=8)
        _save(fig, outdir / "fig_entity_vs_metric.png")


def gen_figs(gf: dict, genflow_path: Path | None, outdir: Path):
    heat = gf.get("mean_flow_heatmap_layer_by_token")
    if heat:
        H = np.array(heat)
        fig, ax = plt.subplots(figsize=(6, 3.4))
        im = ax.imshow(H, aspect="auto", origin="lower", cmap="magma")
        onset = gf.get("mean_clarification_onset_token")
        if onset is not None:
            ax.axvline(onset, color="cyan", ls="--", lw=1.2, label=f"mean clarify onset (t={onset:.0f})")
            ax.legend(fontsize=8, loc="upper right")
        ax.set_xlabel("generated token position"); ax.set_ylabel("layer")
        ax.set_title("Generation-flow: ambiguity score over tokens")
        fig.colorbar(im, ax=ax, label="ambiguity projection")
        _save(fig, outdir / "fig_generation_heatmap.png")

    if genflow_path and genflow_path.exists():
        z = np.load(genflow_path, allow_pickle=True)
        flow, onsets = z["flow"], z["onsets"]
        L = flow.shape[1]
        band = flow[:, int(0.66 * L):, :].mean(1)
        peak = np.nanargmax(np.where(np.isnan(band), -np.inf, band), axis=1)
        m = onsets >= 0
        if m.sum() > 0:
            lead = onsets[m] - peak[m]
            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.hist(lead, bins=range(int(lead.min()) - 1, int(lead.max()) + 2), color="steelblue")
            ax.axvline(0, color="crimson", ls="--", lw=1, label="peak = ask")
            ax.set_xlabel("tokens: clarify-onset − ambiguity-peak  (>0: recognition precedes ask)")
            ax.set_ylabel("instances"); ax.set_title("Recognition onset vs. clarification")
            ax.legend(fontsize=8)
            _save(fig, outdir / "fig_recognition_onset.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--genflow", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=Path("data/interp/figs"))
    args = ap.parse_args()

    report = json.loads(args.report.read_text())
    args.outdir.mkdir(parents=True, exist_ok=True)
    if "depth_flow" in report:
        depth_figs(report["depth_flow"], args.outdir)
    if "generation_flow" in report:
        gen_figs(report["generation_flow"], args.genflow, args.outdir)
    print(f"Figures -> {args.outdir}")


if __name__ == "__main__":
    main()
