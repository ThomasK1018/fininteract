"""Score two filled instance-validation sheets (Task 1).

Computes, for the binary items H1-H6 (and H7 treated as ok/not-ok):
  - per-item percent agreement and Cohen's kappa between the two annotators
  - the pre-adjudication rejection rate (an instance is 'rejected' if either
    annotator answered No to any of H1-H6 -> fails the acceptance rule)
These are the numbers reviewers ask for to back the human-validation claim.

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

def cohen_kappa(a, b):
    """Cohen's kappa for paired binary labels (drops rows with missing)."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n == 0: return None, 0, None
    po = sum(1 for x, y in pairs if x == y) / n
    # expected agreement from marginals
    pa1 = sum(x for x, _ in pairs) / n
    pb1 = sum(y for _, y in pairs) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return kappa, n, po

def load(path):
    rows = {r["instance_id"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}
    return rows

def main(a):
    A, B = load(a.a), load(a.b)
    ids = [i for i in A if i in B]
    print(f"{len(ids)} instances labeled by both annotators\n")
    stats = {"n_instances": len(ids), "items": {}}
    print(f"{'item':22s} {'%agree':>7s} {'kappa':>7s} {'n':>4s}")
    for col in BINARY + ["H7_axis_ok_or_fix"]:
        if col == "H7_axis_ok_or_fix":
            va = [1 if (A[i].get(col, "").strip().lower() in ("ok", "", "yes")) else 0 for i in ids]
            vb = [1 if (B[i].get(col, "").strip().lower() in ("ok", "", "yes")) else 0 for i in ids]
        else:
            va = [yn(A[i].get(col)) for i in ids]
            vb = [yn(B[i].get(col)) for i in ids]
        k, n, po = cohen_kappa(va, vb)
        stats["items"][col] = {"kappa": k, "pct_agree": po, "n": n}
        ks = f"{k:.3f}" if k is not None else "n/a"
        ps = f"{po:.3f}" if po is not None else "n/a"
        print(f"{col:22s} {ps:>7s} {ks:>7s} {n:>4d}")

    # rejection rate: instance rejected if EITHER annotator said No to any H1-H6
    rej = 0
    for i in ids:
        bad = any(yn(A[i].get(c)) == 0 or yn(B[i].get(c)) == 0 for c in BINARY)
        rej += int(bad)
    stats["pre_adjudication_rejection_rate"] = rej / len(ids) if ids else None
    print(f"\npre-adjudication rejection rate: {rej}/{len(ids)} = "
          f"{100*rej/max(len(ids),1):.1f}%")

    # mean kappa over the 6 core binary items (headline number)
    ks = [stats["items"][c]["kappa"] for c in BINARY if stats["items"][c]["kappa"] is not None]
    if ks:
        stats["mean_kappa_H1_H6"] = sum(ks) / len(ks)
        print(f"mean Cohen's kappa (H1-H6): {stats['mean_kappa_H1_H6']:.3f}")
    Path(a.out).write_text(json.dumps(stats, indent=2))
    print("wrote", a.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="annotator A filled CSV")
    p.add_argument("--b", required=True, help="annotator B filled CSV")
    p.add_argument("--out", default="annotation/validation_stats.json")
    main(p.parse_args())
