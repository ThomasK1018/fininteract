"""Export the instance-validation annotation sheet (Task 1: quality control).

Each annotator independently labels a sample of instances on the H1-H8 protocol.
This sheet shows the annotator EVERYTHING (question, context, both answers, both
spans, axis) because the task is to *judge the construction*, not to answer.

Output: a CSV with one row per instance and blank H1-H8 columns to fill in
(Yes/No, except H7 = correct axis or 'ok', H8 = free-text natural question).

Usage:
  python annotation/make_validation_sheet.py --n 60 --seed 7 \
      --out annotation/sheets/validation_blank.csv
  # one sheet per annotator (same instances, independent labels):
  cp validation_blank.csv validation_annotatorA.csv
  cp validation_blank.csv validation_annotatorB.csv
"""
import argparse, json, csv, random
from pathlib import Path

H_COLS = ["H1_ambiguous", "H2_default_plausible", "H3_C_unique",
          "H4_C_not_answer", "H5_A_correct", "H6_Ad_correct",
          "H7_axis_ok_or_fix", "H8_natural_question"]

def stratified(rows, n, seed):
    """Sample n, stratified by (language, primary axis), deterministic."""
    rng = random.Random(seed)
    buckets = {}
    for r in rows:
        key = (r["language"], (r.get("axes") or ["?"])[0])
        buckets.setdefault(key, []).append(r)
    for b in buckets.values():
        rng.shuffle(b)
    # round-robin across buckets until we have n (preserves axis/lang spread)
    out, keys = [], sorted(buckets, key=lambda k: -len(buckets[k]))
    while len(out) < n and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k] and len(out) < n:
                out.append(buckets[k].pop())
    return out

def main(a):
    rows = [json.loads(l) for l in open(a.instances)]
    sample = stratified(rows, min(a.n, len(rows)), a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cols = ["instance_id", "language", "primary_axis", "company", "question",
            "context_C", "intended_answer_A", "default_answer_Ad",
            "intended_span", "default_span"] + H_COLS
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sample:
            w.writerow({
                "instance_id": r["instance_id"],
                "language": r["language"],
                "primary_axis": (r.get("axes") or ["?"])[0],
                "company": r.get("company", ""),
                "question": r["question"],
                "context_C": r["context"],
                "intended_answer_A": r["answer"],
                "default_answer_Ad": r.get("default_answer", ""),
                "intended_span": r.get("intended_evidence_span", ""),
                "default_span": r.get("default_evidence_span", ""),
                **{c: "" for c in H_COLS},
            })
    print(f"wrote {a.out}  ({len(sample)} instances)")
    print("Give each annotator a COPY; they fill H1-H6 with Yes/No, "
          "H7 with 'ok' or the corrected axis, H8 with the yes/no question "
          "they would naturally ask. Work independently.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--instances", default="data/final/fininteract_v1.jsonl")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="annotation/sheets/validation_blank.csv")
    main(p.parse_args())
