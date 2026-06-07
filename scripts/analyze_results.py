"""
Post-hoc analyses over FinInteract evaluation logs (output of evaluate.py).

Pure-Python (no API) — consumes the per-instance result records and produces the
FinInteract-specific analyses, grouped per (model, mode):

  1. Default-capture       — when WRONG, how often does the model return the DEFAULT
                             (naive) interpretation's answer? High = ambiguity blindness.
  2. IR-vs-H0 calibration  — does interaction rate rise with ambiguity magnitude H0?
                             Reports IR per H0 bin + Pearson r(H0, interacted).
  3. First-action mix      — what does the model do first (answer / search / interact)?
                             Answer-first = ambiguity-blind.
  4. Search-vs-ask         — mean searches vs asks; retrieval-as-crutch signal (E4).
  5. Confidence calibration— if confidence logged: Expected Calibration Error (ECE) +
                             mean confidence vs accuracy (overconfidence gap).
  6. Item difficulty       — per-instance p-value (fraction of model/mode runs correct);
                             accuracy-vs-H0 curve; hardest/easiest items.

Usage:
  python scripts/analyze_results.py --results data/results/eval_results.jsonl
  python scripts/analyze_results.py --results data/results/eval_results.jsonl --out data/results/analysis.json
"""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def default_capture(results: list[dict]) -> dict:
    wrong = [r for r in results if not r.get("correct")]
    if not wrong:
        return {"n_wrong": 0, "default_capture_rate": None}
    cap = sum(1 for r in wrong if r.get("default_captured"))
    return {
        "n_wrong": len(wrong),
        "default_capture_rate": round(cap / len(wrong), 3),
        "interpretation": "of wrong answers, fraction equal to the DEFAULT interpretation",
    }


def ir_vs_h0(results: list[dict], bins=(0.5, 1.5, 2.5, 3.5)) -> dict:
    pairs = [(r.get("h0", 0.0), 1 if r.get("interacted") else 0) for r in results]
    if not pairs:
        return {}
    binned = defaultdict(list)
    edges = [0] + list(bins) + [99]
    for h0, it in pairs:
        for lo, hi in zip(edges, edges[1:]):
            if lo <= h0 < hi:
                binned[f"[{lo},{hi})"].append(it)
                break
    per_bin = {k: {"n": len(v), "ir": round(statistics.mean(v), 3)}
               for k, v in sorted(binned.items())}
    r = _pearson([p[0] for p in pairs], [p[1] for p in pairs])
    return {"ir_by_h0_bin": per_bin,
            "pearson_r_h0_interact": round(r, 3) if r is not None else None,
            "interpretation": "r>0 means the model asks more when ambiguity (H0) is higher"}


def first_action_mix(results: list[dict]) -> dict:
    c = defaultdict(int)
    for r in results:
        c[r.get("first_action", "none")] += 1
    n = sum(c.values()) or 1
    return {k: round(v / n, 3) for k, v in sorted(c.items(), key=lambda x: -x[1])}


def search_vs_ask(results: list[dict]) -> dict:
    asks = [r.get("n_asks", 0) for r in results]
    srch = [r.get("n_searches", 0) for r in results]
    return {
        "mean_asks": round(statistics.mean(asks), 2) if asks else 0,
        "mean_searches": round(statistics.mean(srch), 2) if srch else 0,
        "search_no_ask_rate": round(
            sum(1 for r in results if r.get("n_searches", 0) > 0 and r.get("n_asks", 0) == 0)
            / max(len(results), 1), 3),
        "interpretation": "high search_no_ask_rate = retrieval used as a substitute for asking (E4)",
    }


def confidence_calibration(results: list[dict], n_bins: int = 5) -> dict:
    have = [r for r in results if isinstance(r.get("confidence"), (int, float))]
    if not have:
        return {"available": False}
    # ECE
    bins = defaultdict(list)
    for r in have:
        conf = r["confidence"] / 100.0
        b = min(n_bins - 1, int(conf * n_bins))
        bins[b].append((conf, 1 if r.get("correct") else 0))
    ece = 0.0
    total = len(have)
    for b, items in bins.items():
        conf_avg = statistics.mean(c for c, _ in items)
        acc_avg = statistics.mean(a for _, a in items)
        ece += (len(items) / total) * abs(conf_avg - acc_avg)
    mean_conf = statistics.mean(r["confidence"] for r in have) / 100.0
    acc = statistics.mean(1 if r.get("correct") else 0 for r in have)
    return {
        "available": True,
        "n": total,
        "mean_confidence": round(mean_conf, 3),
        "accuracy": round(acc, 3),
        "overconfidence_gap": round(mean_conf - acc, 3),
        "ece": round(ece, 3),
        "interpretation": "overconfidence_gap>0 means stated confidence exceeds accuracy",
    }


def item_difficulty(all_results: list[dict]) -> dict:
    by_item = defaultdict(list)
    for r in all_results:
        by_item[r.get("instance_id")].append(1 if r.get("correct") else 0)
    pvals = {iid: statistics.mean(v) for iid, v in by_item.items()}
    # accuracy vs H0
    h0_acc = defaultdict(list)
    h0_of = {r.get("instance_id"): r.get("h0", 0.0) for r in all_results}
    for iid, p in pvals.items():
        h0_acc[round(h0_of.get(iid, 0.0), 1)].append(p)
    acc_by_h0 = {k: round(statistics.mean(v), 3) for k, v in sorted(h0_acc.items())}
    hardest = sorted(pvals.items(), key=lambda x: x[1])[:10]
    return {
        "n_items": len(pvals),
        "mean_pvalue": round(statistics.mean(pvals.values()), 3) if pvals else None,
        "universally_hard": [iid for iid, p in pvals.items() if p == 0.0][:20],
        "hardest_10": [(iid, round(p, 2)) for iid, p in hardest],
        "accuracy_by_h0": acc_by_h0,
        "interpretation": "p-value = fraction of (model,mode) runs correct; lower = harder",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    results = _load(args.results)
    print(f"Loaded {len(results)} result records")

    # Per (model, mode) analyses
    groups = defaultdict(list)
    for r in results:
        groups[(r.get("model"), r.get("mode"))].append(r)

    report = {"per_group": {}, "item_difficulty": item_difficulty(results)}
    for (model, mode), rs in sorted(groups.items()):
        key = f"{model} | {mode}"
        acc = statistics.mean(1 if r.get("correct") else 0 for r in rs)
        g = {
            "n": len(rs),
            "accuracy": round(acc, 3),
            "default_capture": default_capture(rs),
            "ir_vs_h0": ir_vs_h0(rs),
            "first_action": first_action_mix(rs),
            "search_vs_ask": search_vs_ask(rs),
            "confidence": confidence_calibration(rs),
        }
        report["per_group"][key] = g
        dc = g["default_capture"]["default_capture_rate"]
        r_h0 = g["ir_vs_h0"].get("pearson_r_h0_interact")
        conf = g["confidence"]
        print(f"\n=== {key}  (n={len(rs)}, acc={acc:.1%}) ===")
        print(f"  default-capture (of wrong): {dc}")
        print(f"  IR~H0 correlation: {r_h0}   first-action: {g['first_action']}")
        print(f"  search_no_ask_rate: {g['search_vs_ask']['search_no_ask_rate']}")
        if conf.get("available"):
            print(f"  overconfidence_gap: {conf['overconfidence_gap']}  ECE: {conf['ece']}")

    idiff = report["item_difficulty"]
    print(f"\n=== Item difficulty ({idiff['n_items']} items) ===")
    print(f"  mean p-value: {idiff['mean_pvalue']}  universally-hard items: {len(idiff['universally_hard'])}")
    print(f"  accuracy by H0: {idiff['accuracy_by_h0']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nFull report -> {args.out}")


if __name__ == "__main__":
    main()
