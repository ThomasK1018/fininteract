"""Lexical control baselines for the ambiguity probe (reviewer point 9, CPU-only part).

The reviewer's concern: the neural axis probe "may exploit lexical or source cues," and
Q-vs-Q+C detection "can largely detect context presence." This script quantifies exactly
those null baselines from raw TEXT (no activations, no GPU), so the neural probe can be
reported ABOVE these lexical floors:

  (1) AXIS DECODING from bag-of-words on the BARE question (entity_scope vs
      metric_definition) -- char n-gram LogReg, cross-validated. If BoW matches the
      neural probe, axis decoding is lexical.
  (2) LENGTH-ONLY axis baseline: a single feature (question length). Controls for the
      trivial "longer questions are a different axis" confound.
  (3) DETECTION from bag-of-words (Q vs Q+C). Expected near-ceiling because Q+C appends
      text -- this demonstrates the reviewer's point that raw detection is lexical, so the
      paper's claim must rest on axis decoding / behavioral mismatch, not detection alone.
  (4) Company/source-disjoint axis decoding: GroupKFold by company so no company's
      instances straddle the split -- tests whether decoding rides source cues.

Reported per language (EN, ZH) and combined, vs the majority-class floor.

Usage:
    python scripts/probe_bow_control.py --instances data/final/fininteract_v1.jsonl
"""
import argparse, json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score

AXIS_A, AXIS_B = "entity_scope", "metric_definition"


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            axes = d.get("axes") or []
            rows.append({
                "id": d.get("instance_id"),
                "q": d.get("question", ""),
                "c": d.get("context", "") or d.get("intended_interpretation", ""),
                "axis": axes[0] if axes else None,
                "lang": d.get("language"),
                "company": d.get("company") or d.get("ticker") or d.get("instance_id"),
            })
    return rows


def cv_scores(X_text, y, groups=None, vectorizer=None, folds=5):
    """Cross-validated accuracy + AUROC for a text classifier. Returns (acc, auroc, maj)."""
    y = np.asarray(y)
    maj = max(np.mean(y == 0), np.mean(y == 1))
    vec = vectorizer or CountVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)
    X = vec.fit_transform(X_text)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    if groups is not None:
        n_splits = min(folds, len(set(groups)))
        splitter = GroupKFold(n_splits=n_splits)
        pred = cross_val_predict(clf, X, y, cv=splitter, groups=groups)
        proba = cross_val_predict(clf, X, y, cv=splitter, groups=groups, method="predict_proba")[:, 1]
    else:
        n_splits = min(folds, int(min(np.sum(y == 0), np.sum(y == 1))))
        if n_splits < 2:
            return float("nan"), float("nan"), maj
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        pred = cross_val_predict(clf, X, y, cv=splitter)
        proba = cross_val_predict(clf, X, y, cv=splitter, method="predict_proba")[:, 1]
    auroc = roc_auc_score(y, proba) if len(set(y)) == 2 else float("nan")
    return accuracy_score(y, pred), auroc, maj


def length_scores(texts, y, folds=5):
    """Axis decoding from a single feature: character length of the question."""
    y = np.asarray(y)
    X = np.array([[len(t)] for t in texts], dtype=float)
    maj = max(np.mean(y == 0), np.mean(y == 1))
    n_splits = min(folds, int(min(np.sum(y == 0), np.sum(y == 1))))
    if n_splits < 2:
        return float("nan"), float("nan"), maj
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    sk = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    pred = cross_val_predict(clf, X, y, cv=sk)
    proba = cross_val_predict(clf, X, y, cv=sk, method="predict_proba")[:, 1]
    return accuracy_score(y, pred), roc_auc_score(y, proba), maj


def subset(rows, lang=None):
    ab = [r for r in rows if r["axis"] in (AXIS_A, AXIS_B)]
    if lang:
        ab = [r for r in ab if r["lang"] == lang]
    return ab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="data/final/fininteract_v1.jsonl")
    args = ap.parse_args()
    rows = load(args.instances)
    print(f"loaded {len(rows)} instances | "
          f"{AXIS_A}={sum(r['axis']==AXIS_A for r in rows)} "
          f"{AXIS_B}={sum(r['axis']==AXIS_B for r in rows)}\n")

    print("=" * 92)
    print(f"(1)+(2) AXIS DECODING FROM TEXT: {AXIS_A} vs {AXIS_B}  (bare question, CV)")
    print("=" * 92)
    print(f"{'split':16s} {'n':>4s} {'maj':>5s} | {'BoW acc':>8s} {'BoW AUROC':>10s} | "
          f"{'len acc':>8s} {'len AUROC':>10s}")
    for lang in [None, "en", "zh"]:
        ab = subset(rows, lang)
        if len(ab) < 10:
            continue
        y = [0 if r["axis"] == AXIS_A else 1 for r in ab]
        q = [r["q"] for r in ab]
        b_acc, b_auc, maj = cv_scores(q, y)
        l_acc, l_auc, _ = length_scores(q, y)
        name = {None: "combined", "en": "EN", "zh": "ZH"}[lang]
        print(f"{name:16s} {len(ab):4d} {maj:5.2f} | {b_acc:8.2f} {b_auc:10.2f} | "
              f"{l_acc:8.2f} {l_auc:10.2f}")

    print("\n" + "=" * 92)
    print("(4) COMPANY-DISJOINT AXIS DECODING (GroupKFold by company; BoW)")
    print("=" * 92)
    ab = subset(rows)
    y = [0 if r["axis"] == AXIS_A else 1 for r in ab]
    q = [r["q"] for r in ab]
    groups = [r["company"] for r in ab]
    d_acc, d_auc, maj = cv_scores(q, y, groups=groups)
    print(f"company-disjoint   n={len(ab)}  maj={maj:.2f}  BoW acc={d_acc:.2f}  AUROC={d_auc:.2f}  "
          f"({len(set(groups))} companies)")

    print("\n" + "=" * 92)
    print("(3) DETECTION FROM TEXT: bare Q (0) vs Q+context (1)  -- expected near-ceiling")
    print("=" * 92)
    detect_rows = [r for r in rows if r["c"]]
    texts = [r["q"] for r in detect_rows] + [f"{r['q']}\n\nClarifying context: {r['c']}" for r in detect_rows]
    yd = [0] * len(detect_rows) + [1] * len(detect_rows)
    groups_d = [r["id"] for r in detect_rows] * 2  # pair never splits
    acc_d, auc_d, maj_d = cv_scores(texts, yd, groups=groups_d)
    print(f"detection (paired GroupKFold)   n={len(texts)}  maj={maj_d:.2f}  "
          f"BoW acc={acc_d:.2f}  AUROC={auc_d:.2f}")
    print("\nInterpretation: BoW detection near 1.0 confirms Q-vs-Q+C is lexically trivial "
          "(reviewer's point). The load-bearing claim is AXIS decoding (1) and the\n"
          "behavioral mismatch -- report the neural axis probe ABOVE the BoW/length floors "
          "here, and prefer the company-disjoint number (4) to rule out source cues.")


if __name__ == "__main__":
    main()
