"""
Linear-probe analyses over extracted activations (output of probe_activations.py).

Three studies:
  1. AMBIGUITY DETECTION (within-item): per layer, can a linear probe separate the bare
     ambiguous question from the disambiguated (question+context) one? Reports per-layer
     test accuracy + the peak layer. Robust to axis imbalance (balanced 1:1 pairs).
     A high peak = the model linearly represents "this question is under-specified."

  2. AXIS DECODING: among ambiguous activations, can a probe decode the axis? Restricted to
     entity_scope vs metric_definition (only classes with enough samples). Honest scope.

  3. BEHAVIORAL-INTERNAL MISMATCH (the headline): using the SAME open model's behavioral
     eval results (evaluate.py output for that model), among instances the probe scores as
     "ambiguous", how often did the model still answer directly / land on the default?
     High mismatch = the model REPRESENTS ambiguity but does not ACT on it
     (bottleneck = action, not recognition).

Pure CPU (numpy + scikit-learn); no GPU needed.

Usage:
  python scripts/analyze_probes.py --acts data/interp/acts_qwen_bare.npz \
      --rep last \
      --behavior data/results/eval_qwen.jsonl   # optional, same model, interact mode
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, GroupKFold


def detection_probe(amb, dis, n_layers):
    """Per-layer CV accuracy separating ambiguous (1) from disambiguated (0)."""
    n = amb.shape[0]
    y = np.r_[np.ones(n), np.zeros(n)]
    # group by instance so a pair never splits across folds
    groups = np.r_[np.arange(n), np.arange(n)]
    accs = []
    for L in range(n_layers):
        X = np.r_[amb[:, L, :], dis[:, L, :]]
        clf = LogisticRegression(max_iter=2000, C=1.0)
        gkf = GroupKFold(n_splits=5)
        score = cross_val_score(clf, X, y, groups=groups, cv=gkf, scoring="accuracy").mean()
        accs.append(round(float(score), 3))
    return accs


def axis_probe(amb, axes, n_layers, classes=("entity_scope", "metric_definition")):
    mask = np.isin(axes, classes)
    if mask.sum() < 20:
        return None
    y = (axes[mask] == classes[0]).astype(int)
    accs = []
    for L in range(n_layers):
        X = amb[mask, L, :]
        clf = LogisticRegression(max_iter=2000, C=1.0)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        score = cross_val_score(clf, X, y, cv=skf, scoring="accuracy").mean()
        accs.append(round(float(score), 3))
    return {"classes": list(classes), "n": int(mask.sum()),
            "baseline": round(float(max(y.mean(), 1 - y.mean())), 3),
            "per_layer_acc": accs}


def behavioral_mismatch(amb, dis, ids, behavior_path: Path, peak_layer: int):
    """Among instances the probe scores 'ambiguous', how many did the model NOT act on?"""
    beh = {}
    for l in behavior_path.open():
        if not l.strip():
            continue
        r = json.loads(l)
        beh[r.get("instance_id")] = r
    # fit detection probe at peak layer on all pairs, then score the ambiguous prompts
    n = amb.shape[0]
    X = np.r_[amb[:, peak_layer, :], dis[:, peak_layer, :]]
    y = np.r_[np.ones(n), np.zeros(n)]
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(X, y)
    amb_score = clf.predict_proba(amb[:, peak_layer, :])[:, 1]   # P(ambiguous) per instance

    rows = []
    for i, iid in enumerate(ids):
        r = beh.get(str(iid)) or beh.get(iid)
        if not r:
            continue
        rows.append({
            "iid": iid, "p_ambiguous": float(amb_score[i]),
            "asked": bool(r.get("interacted")),
            "default_captured": bool(r.get("default_captured")),
            "correct": bool(r.get("correct")),
        })
    if not rows:
        return {"available": False, "reason": "no instance_id overlap with behavior file"}
    hi = [r for r in rows if r["p_ambiguous"] >= 0.5]   # internally flagged ambiguous
    if not hi:
        return {"available": True, "n_flagged": 0}
    return {
        "available": True,
        "n_flagged_ambiguous": len(hi),
        "of_those_did_not_ask": round(sum(1 for r in hi if not r["asked"]) / len(hi), 3),
        "of_those_default_captured": round(sum(1 for r in hi if r["default_captured"]) / len(hi), 3),
        "interpretation": "high 'did_not_ask' among internally-ambiguous instances = the model "
                          "represents ambiguity but does not act on it (bottleneck is action).",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", type=Path, required=True)
    ap.add_argument("--rep", choices=["last", "mean"], default="last")
    ap.add_argument("--behavior", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    d = np.load(args.acts, allow_pickle=True)
    amb = d[f"amb_{args.rep}"]; dis = d[f"dis_{args.rep}"]
    axes = d["axes"]; ids = d["ids"]
    n_layers = amb.shape[1]
    print(f"Loaded {amb.shape[0]} instances, {n_layers} layers, rep={args.rep}, "
          f"model={str(d['model'])}, prompt_mode={str(d['prompt_mode'])}")

    det = detection_probe(amb, dis, n_layers)
    peak = int(np.argmax(det))
    print(f"\n[1] Ambiguity-detection probe (per-layer CV acc):")
    print("    " + " ".join(f"{a:.2f}" for a in det))
    print(f"    peak layer {peak}: acc={det[peak]:.3f}  (0.50 = chance)")

    ax = axis_probe(amb, axes, n_layers)
    print(f"\n[2] Axis-decoding probe (entity vs metric):")
    if ax:
        apk = int(np.argmax(ax["per_layer_acc"]))
        print(f"    n={ax['n']} baseline={ax['baseline']}  peak layer {apk}: "
              f"acc={ax['per_layer_acc'][apk]:.3f}")
    else:
        print("    insufficient samples")

    report = {"detection_per_layer": det, "detection_peak_layer": peak,
              "detection_peak_acc": det[peak], "axis_decoding": ax}

    if args.behavior:
        bm = behavioral_mismatch(amb, dis, ids, args.behavior, peak)
        report["behavioral_mismatch"] = bm
        print(f"\n[3] Behavioral-internal mismatch (peak layer {peak}):")
        if bm.get("available") and bm.get("n_flagged_ambiguous"):
            print(f"    {bm['n_flagged_ambiguous']} instances internally flagged ambiguous")
            print(f"    of those, did NOT ask:       {bm['of_those_did_not_ask']:.1%}")
            print(f"    of those, default-captured:  {bm['of_those_default_captured']:.1%}")
            print(f"    -> represents-but-doesn't-act signal")
        else:
            print(f"    {bm}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nReport -> {args.out}")


if __name__ == "__main__":
    main()
