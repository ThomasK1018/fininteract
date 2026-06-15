"""Score two filled instance-validation sheets (Task 1).

Computes, for the binary items H1-H6 (and H7 treated as ok/not-ok):
  - per-item percent agreement, Gwet's AC1 (PRIMARY), and Cohen's kappa.
    AC1 is the headline statistic because the labels are heavily skewed toward
    "Yes" (>85%): under that skew Cohen's kappa hits the Feinstein-Cicchetti
    "prevalence paradox" (kappa ~0 or negative DESPITE 90%+ agreement), so kappa
    is reported only for completeness and flagged as unreliable here.
  - CONSENSUS rejection rate: an instance is genuinely contested only if BOTH
    annotators flag the SAME item No. (The earlier 'union' rule -- either
    annotator, any item -- inflates the count by accumulating rare,
    non-overlapping single flags, and is reported as a secondary diagnostic.)

Usage:
  python annotation/score_validation.py \
      --a annotation/sheets/validation_annotatorA.csv \
      --b annotation/sheets/validation_annotatorB.csv \
      --out annotation/validation_stats.json
"""
import argparse, csv, json
from pathlib import Path

BINARY = ["H1_ambiguous", "H2_default_plausible", "H3_C_unique",
          "H4_C_not_answer", "H5_A_correct", "H6_Ad_correct"]

def yn(v):
    v = (v or "").strip().lower()
    if v in ("y", "yes", "1", "true", "t"): return 1
    if v in ("n", "no", "0", "false", "f"): return 0
    return None

def agreement_stats(a, b):
    """Return (pct_agree, cohen_kappa, gwet_ac1, n) for paired binary labels."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n == 0: return None, None, None, 0
    po = sum(1 for x, y in pairs if x == y) / n
    pa1 = sum(x for x, _ in pairs) / n
    pb1 = sum(y for _, y in pairs) / n
    # Cohen's kappa (chance from each annotator's marginals)
    pe_k = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    kappa = (po - pe_k) / (1 - pe_k) if pe_k < 1 else 1.0
    # Gwet's AC1 (chance from overall prevalence; robust to skew)
    p1 = (pa1 + pb1) / 2
    pe_g = 2 * p1 * (1 - p1)
    ac1 = (po - pe_g) / (1 - pe_g) if pe_g < 1 else 1.0
    return po, kappa, ac1, n

def load(path):
    rows = {r["instance_id"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}
    return rows

def main(a):
    A, B = load(a.a), load(a.b)
    ids = [i for i in A if i in B]
    print(f"{len(ids)} instances labeled by both annotators\n")
    stats = {"n_instances": len(ids), "items": {}}
    print(f"{'item':22s} {'%agree':>7s} {'AC1':>7s} {'kappa':>7s} {'n':>4s}   (AC1 = primary)")
    for col in BINARY + ["H7_axis_ok_or_fix"]:
        if col == "H7_axis_ok_or_fix":
            va = [1 if (A[i].get(col, "").strip().lower() in ("ok", "", "yes")) else 0 for i in ids]
            vb = [1 if (B[i].get(col, "").strip().lower() in ("ok", "", "yes")) else 0 for i in ids]
        else:
            va = [yn(A[i].get(col)) for i in ids]
            vb = [yn(B[i].get(col)) for i in ids]
        po, k, ac1, n = agreement_stats(va, vb)
        stats["items"][col] = {"pct_agree": po, "ac1": ac1, "kappa": k, "n": n}
        f = lambda v: f"{v:.3f}" if v is not None else "n/a"
        print(f"{col:22s} {f(po):>7s} {f(ac1):>7s} {f(k):>7s} {n:>4d}")

    # CONSENSUS rejection: BOTH annotators flag the SAME item No (real concern).
    # UNION (either annotator, any item) is reported too but inflates via scattered
    # single flags -> those go to adjudication, not auto-reject.
    consensus, union, consensus_ids = 0, 0, []
    for i in ids:
        a_no = {c for c in BINARY if yn(A[i].get(c)) == 0}
        b_no = {c for c in BINARY if yn(B[i].get(c)) == 0}
        if a_no or b_no: union += 1
        both = a_no & b_no
        if both:
            consensus += 1
            consensus_ids.append({"instance_id": i, "items": sorted(both)})
    stats["consensus_rejection_rate"] = consensus / len(ids) if ids else None
    stats["union_flag_rate"] = union / len(ids) if ids else None
    stats["consensus_flagged"] = consensus_ids
    print(f"\nCONSENSUS rejection (both flag same item): {consensus}/{len(ids)} = "
          f"{100*consensus/max(len(ids),1):.1f}%   <- real concern level")
    print(f"union flag (either annotator, any item):   {union}/{len(ids)} = "
          f"{100*union/max(len(ids),1):.1f}%   <- send to adjudication, not auto-reject")
    if consensus_ids:
        print("consensus-flagged (adjudicate):", consensus_ids)

    # headline agreement = mean AC1 over the 6 core items (kappa is paradox-degenerate here)
    ac = [stats["items"][c]["ac1"] for c in BINARY if stats["items"][c]["ac1"] is not None]
    if ac:
        stats["mean_ac1_H1_H6"] = sum(ac) / len(ac)
        print(f"\nmean Gwet's AC1 (H1-H6): {stats['mean_ac1_H1_H6']:.3f}  (primary agreement statistic)")
    Path(a.out).write_text(json.dumps(stats, indent=2))
    print("wrote", a.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="annotator A filled CSV")
    p.add_argument("--b", required=True, help="annotator B filled CSV")
    p.add_argument("--out", default="annotation/validation_stats.json")
    main(p.parse_args())
