"""Blind default-interpretation annotation (reviewer point 4a).

Tests the single-gold-illusion PREMISE empirically: would a conventional benchmark
builder, seeing only the question and the passage (NOT the resolving context C, and
UNAWARE the item is ambiguous), record the constructed default answer A_d as gold?

Annotator sees: question + passage (both candidate values present but unlabeled, exactly
what a benchmark author scraping the filing would face). Task: write the single answer you
would record as the gold answer -- as if building an ordinary QA benchmark. The word
"ambiguous" never appears; we do NOT ask them to resolve anything. A separate optional
column asks, only AFTER they commit, whether they noticed more than one defensible answer.

Scoring (score_default.py) reports: % of commitments matching A_d, % matching A_i,
inter-annotator agreement, and per-item answer entropy. High A_d-match + low entropy =>
a single-gold benchmark would indeed encode the default => the illusion premise holds.

Produces annotator sheets (per language, csv+xlsx) + a held-back answer key.

Usage:
  python annotation/make_default_sheet.py --n 60 --seed 23 \
      --outdir annotation/sheets/default
"""
import argparse, json, csv, random
from pathlib import Path

# reuse the exact conventions from the baseline sheet builder
import sys
sys.path.insert(0, str(Path(__file__).parent))
from make_baseline_sheet import write_xlsx, nonleaky_passage, stratified

HEADER = ["instance_id", "company", "question", "passage",
          "annotator_id", "your_gold_answer", "confidence_1to5",
          "noticed_second_answer_y_n"]


def main(a):
    rows = [json.loads(l) for l in open(a.instances, encoding="utf-8")]
    sample = stratified(rows, min(a.n, len(rows)), a.seed)
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    for lang in sorted(set(r["language"] for r in sample)):
        sub = [r for r in sample if r["language"] == lang]
        sheet_rows = []
        for idx, r in enumerate(sub):
            rng = random.Random(20_000 + idx)
            sheet_rows.append([r["instance_id"], r.get("company", ""), r["question"],
                               nonleaky_passage(r, rng), "", "", "", ""])
        csv_path = outdir / f"default_questions_{lang}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(HEADER); w.writerows(sheet_rows)
        write_xlsx(outdir / f"default_questions_{lang}.xlsx", HEADER, sheet_rows)
        print(f"wrote {csv_path}(+.xlsx)  ({len(sub)} {lang}) -- blind: Q + passage, "
              f"'write the gold answer you'd record'. No mention of ambiguity.")

    keypath = outdir / "default_answerkey.csv"
    with open(keypath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "language", "primary_axis", "question",
                    "intended_answer", "default_answer", "context_C"])
        for r in sample:
            w.writerow([r["instance_id"], r["language"], (r.get("axes") or ["?"])[0],
                        r["question"], r["answer"], r.get("default_answer", ""), r["context"]])
    print(f"wrote {keypath}  -- EXPERIMENTER ONLY (A_i, A_d, C). Do not show annotators.")
    print("\nRecruit >=3 annotators/language; each fills annotator_id so agreement + entropy "
          "are computable. Give each the SAME items (blind replication).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--instances", default="data/final/fininteract_v1.jsonl")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--outdir", default="annotation/sheets/default")
    main(p.parse_args())
