"""Exp-9 neural-probe CONTROLS (reviewer response). Extends the axis-decoding probe with the
controls a linear-probe claim needs, all in AUROC so they compare to the BoW lexical floor:
  - neural axis probe (entity vs metric), StratifiedKFold AUROC + best layer
  - shuffled-label NULL (permutation): must collapse to ~0.5
  - company-disjoint (GroupKFold by company) and source-disjoint AUROC: does it survive when no
    train/test company or source overlaps?
  - cross-language transfer: fit EN -> test ZH and fit ZH -> test EN AUROC
The neural probe must beat the BoW floor (~0.73 combined / ~0.80 company-disjoint) to support a
"the axis is represented beyond surface lexicon" claim. Pure CPU.

Usage:
  python scripts/analyze_probes_controls.py --acts data/interp/acts_qwen3-4b.npz \
      --instances data/final/fininteract_v1.jsonl --out data/results/probe_controls_qwen3-4b.json
"""
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score

CLASSES = ("entity_scope", "metric_definition")   # only classes with enough n (94 / 63)


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))


def auroc_cv(X, y, seed=0):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return float(cross_val_score(clf(), X, y, cv=skf, scoring="roc_auc").mean())


def auroc_group(X, y, groups):
    ng = len(np.unique(groups))
    if ng < 3:
        return None
    gkf = GroupKFold(n_splits=min(5, ng))
    scores = cross_val_score(clf(), X, y, groups=groups, cv=gkf, scoring="roc_auc")
    scores = scores[~np.isnan(scores)]            # drop folds where a group has one class only
    return float(scores.mean()) if len(scores) else None


def auroc_transfer(Xtr, ytr, Xte, yte):
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return None
    m = clf().fit(Xtr, ytr)
    return float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))


def main(a):
    d = np.load(a.acts, allow_pickle=True)
    amb = d[f"amb_{a.rep}"]; axes = d["axes"]; langs = d["langs"]; ids = d["ids"]
    n_layers = amb.shape[1]
    # join company/source by instance_id
    meta = {json.loads(l)["instance_id"]: json.loads(l)
            for l in open(a.instances) if l.strip()}
    company = np.array([meta.get(str(i), {}).get("company", "?") for i in ids])
    source = np.array([meta.get(str(i), {}).get("source", "?") for i in ids])

    mask = np.isin(axes, CLASSES)
    y = (axes[mask] == CLASSES[0]).astype(int)
    lang_m = langs[mask]; comp_m = company[mask]; src_m = source[mask]
    print(f"model={str(d['model'])} rep={a.rep} | axis probe {CLASSES} n={mask.sum()} "
          f"(pos={y.sum()}) | EN={int((lang_m=='en').sum())} ZH={int((lang_m=='zh').sum())}")

    # 1. per-layer neural AUROC -> best layer
    per_layer = [auroc_cv(amb[mask, L, :], y) for L in range(n_layers)]
    best = int(np.argmax(per_layer))
    Xb = amb[mask, best, :]
    print(f"[neural] best layer {best}/{n_layers-1}  AUROC={per_layer[best]:.3f}")

    # 2. shuffled-label null (permutation) at best layer
    rng = np.random.RandomState(0)
    nulls = [auroc_cv(Xb, rng.permutation(y), seed=s) for s in range(5)]
    null_mean = float(np.mean(nulls))
    print(f"[null]   shuffled-label AUROC={null_mean:.3f} (should be ~0.5)")

    # 3. disjoint-group splits at best layer
    comp_auroc = auroc_group(Xb, y, comp_m)
    src_auroc = auroc_group(Xb, y, src_m)
    print(f"[disjoint] company-disjoint AUROC={comp_auroc}  source-disjoint AUROC={src_auroc}")

    # 4. cross-language transfer at best layer
    en = lang_m == "en"; zh = lang_m == "zh"
    en2zh = auroc_transfer(Xb[en], y[en], Xb[zh], y[zh])
    zh2en = auroc_transfer(Xb[zh], y[zh], Xb[en], y[en])
    print(f"[cross-lang] EN->ZH AUROC={en2zh}  ZH->EN AUROC={zh2en}")

    BOW = {"combined": 0.73, "company_disjoint": 0.78, "length_only": 0.64,
           "en_overfit": 0.97, "zh": 0.57,
           "note": "scripts/probe_bow_control.py on fininteract_v1 (173): BoW AUROC combined 0.73 "
                   "(EN 0.97 n=37 overfit, ZH 0.57~chance), length 0.64, company-disjoint 0.78"}
    report = {
        "model": str(d["model"]), "rep": a.rep, "classes": list(CLASSES),
        "n": int(mask.sum()), "best_layer": best,
        "neural_auroc_cv": round(per_layer[best], 3),
        "per_layer_auroc": [round(x, 3) for x in per_layer],
        "shuffled_null_auroc": round(null_mean, 3),
        "company_disjoint_auroc": round(comp_auroc, 3) if comp_auroc else None,
        "source_disjoint_auroc": round(src_auroc, 3) if src_auroc else None,
        "cross_lang_en2zh_auroc": round(en2zh, 3) if en2zh else None,
        "cross_lang_zh2en_auroc": round(zh2en, 3) if zh2en else None,
        "bow_floor": BOW,
        "beats_bow_combined": bool(per_layer[best] > BOW["combined"]),
        "beats_bow_company_disjoint": bool(comp_auroc and comp_auroc > BOW["company_disjoint"]),
    }
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=2))
        print(f"-> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--acts", type=Path, required=True)
    p.add_argument("--instances", type=Path, default=Path("data/final/fininteract_v1.jsonl"))
    p.add_argument("--rep", choices=["last", "mean"], default="last")
    p.add_argument("--out", type=Path, default=None)
    main(p.parse_args())
