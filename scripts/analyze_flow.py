"""
Flow analyses (CPU): depth-flow (across layers) + generation-flow (across tokens).

DEPTH-FLOW (from probe_activations.py output, no GPU):
  - Per-layer ambiguity-direction projection of the final-prompt-token activation, separately
    for ambiguous vs disambiguated prompts -> shows at WHICH DEPTH the model separates an
    under-specified question from a disambiguated one (the layer where the curves diverge).
  - Per-axis depth curves: does temporal vs entity vs metric ambiguity emerge at different
    layers? (entity-vs-metric is statistically supported; rarer axes shown for reference.)
  - Axis-distinguishability: an entity-minus-metric direction per layer; projection of entity
    vs metric instances -> the depth at which the model distinguishes ambiguity TYPES.

GENERATION-FLOW (from flow_generate.py output, GPU produced it):
  - Mean flow heatmap (layer x generated-token-position), and the recognition-onset analysis:
    where (token position) the ambiguity projection peaks, and whether that aligns with the
    first clarification token (clarify_onset). Tests the "moment of recognition" hypothesis.

Usage:
  python src/analyze_flow.py --acts data/interp/acts_bare.npz \
      --genflow data/interp/genflow.npz --out data/interp/flow_report.json
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np


def _dirs(amb, dis):
    mu_a, mu_d = amb.mean(0), dis.mean(0)
    d = mu_a - mu_d
    d /= (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-8)
    mid = 0.5 * (mu_a + mu_d)
    return d, mid


def depth_flow(acts_path: Path) -> dict:
    z = np.load(acts_path, allow_pickle=True)
    amb, dis, axes = z["amb_last"], z["dis_last"], z["axes"]   # [N,L,H], [N,L,H]
    d, mid = _dirs(amb, dis)
    proj = lambda X: ((X - mid) * d).sum(-1)                    # [N, L]
    pa, pd = proj(amb), proj(dis)
    L = amb.shape[1]

    # where do ambiguous vs disambiguated separate across depth?
    sep = (pa.mean(0) - pd.mean(0))                             # [L]
    peak_layer = int(np.argmax(sep))

    # per-axis ambiguous depth curves
    per_axis = {}
    for ax in sorted(set(axes.tolist())):
        m = axes == ax
        if m.sum() >= 1:
            per_axis[ax] = {"n": int(m.sum()),
                            "curve": np.round(pa[m].mean(0), 3).tolist()}

    # axis-distinguishability: entity vs metric direction per layer
    me = axes == "entity_scope"; mm = axes == "metric_definition"
    axis_sep = None
    if me.sum() >= 10 and mm.sum() >= 10:
        ad = amb[me].mean(0) - amb[mm].mean(0)                  # [L,H]
        ad /= (np.linalg.norm(ad, axis=-1, keepdims=True) + 1e-8)
        pe = ((amb[me] - amb.mean(0)) * ad).sum(-1).mean(0)     # [L]
        pm = ((amb[mm] - amb.mean(0)) * ad).sum(-1).mean(0)
        axis_sep = {"entity_curve": np.round(pe, 3).tolist(),
                    "metric_curve": np.round(pm, 3).tolist(),
                    "peak_separation_layer": int(np.argmax(np.abs(pe - pm)))}

    return {
        "n_layers": L,
        "ambiguous_vs_disambiguated_separation_by_layer": np.round(sep, 3).tolist(),
        "peak_separation_layer": peak_layer,
        "per_axis_depth_curves": per_axis,
        "axis_distinguishability_entity_vs_metric": axis_sep,
    }


def generation_flow(genflow_path: Path) -> dict:
    z = np.load(genflow_path, allow_pickle=True)
    flow, axes, onsets, lengths = z["flow"], z["axes"], z["onsets"], z["lengths"]
    # flow: [N, L, T] padded with nan. Mean over instances (nan-aware) -> [L, T]
    mean_heat = np.nanmean(flow, axis=0)
    L, T = mean_heat.shape

    # recognition onset: per instance, the generation-token position of peak ambiguity
    # (use the peak-separation layer band: mean over last third of layers)
    band = flow[:, int(0.66 * L):, :].mean(1)                   # [N, T]
    peak_tok = np.nanargmax(np.where(np.isnan(band), -np.inf, band), axis=1)

    # does peak ambiguity precede / align with the clarification token?
    has_clar = onsets >= 0
    align = None
    if has_clar.sum() > 0:
        lead = (onsets[has_clar] - peak_tok[has_clar])         # >0: peak precedes clarification
        align = {"n_with_clarification": int(has_clar.sum()),
                 "mean_tokens_peak_before_clarify": round(float(np.nanmean(lead)), 2),
                 "frac_peak_before_clarify": round(float((lead >= 0).mean()), 3)}

    return {
        "n": int(flow.shape[0]), "n_layers": L, "max_tokens": T,
        "mean_flow_heatmap_layer_by_token": np.round(mean_heat, 3).tolist(),
        "mean_clarification_onset_token": round(float(onsets[onsets >= 0].mean()), 2)
                                          if (onsets >= 0).any() else None,
        "recognition_onset_alignment": align,
        "interpretation": "frac_peak_before_clarify near 1 = ambiguity signal peaks BEFORE the "
                          "model emits its clarifying question (internal recognition precedes the ask).",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", type=Path, help="probe_activations.py output (depth-flow)")
    ap.add_argument("--genflow", type=Path, help="flow_generate.py output (generation-flow)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    report = {}
    if args.acts:
        df = depth_flow(args.acts)
        report["depth_flow"] = df
        print("=== DEPTH-FLOW ===")
        print(f"  peak ambiguous-vs-disambiguated separation at layer {df['peak_separation_layer']}/{df['n_layers']}")
        if df["axis_distinguishability_entity_vs_metric"]:
            print(f"  entity-vs-metric distinguishable; peak-separation layer "
                  f"{df['axis_distinguishability_entity_vs_metric']['peak_separation_layer']}")
        print(f"  per-axis curves for: {list(df['per_axis_depth_curves'])}")
    if args.genflow:
        gf = generation_flow(args.genflow)
        report["generation_flow"] = gf
        print("\n=== GENERATION-FLOW ===")
        print(f"  instances={gf['n']}  layers={gf['n_layers']}  max_tokens={gf['max_tokens']}")
        print(f"  mean clarification onset token: {gf['mean_clarification_onset_token']}")
        if gf["recognition_onset_alignment"]:
            a = gf["recognition_onset_alignment"]
            print(f"  peak ambiguity precedes clarification in {a['frac_peak_before_clarify']:.0%} "
                  f"of asking cases (mean lead {a['mean_tokens_peak_before_clarify']} tokens)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nReport -> {args.out}")


if __name__ == "__main__":
    main()
