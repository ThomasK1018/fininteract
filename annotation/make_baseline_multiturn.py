"""Turn-matched human baseline sheet (reviewer point 7).

The existing human baseline gives ONE yes/no question; agents get up to MAX_INTERACT=6
rounds. That is not directly comparable. This sheet gives the human the SAME budget: up to
--max-turns yes/no questions, each answered truthfully from C by the experimenter, then one
final answer. Scoring reports BOTH the 1-turn ceiling (first question only) and the
agent-matched multi-turn ceiling, so the Human row is comparable to +Interact.

Also carries an annotator_id column so several annotators per language can be run (the paper
notes the ZH result is confounded by a single bilingual annotator -- recruit >=2/language).

Usage:
  python annotation/make_baseline_multiturn.py --n 50 --max-turns 6 --seed 11 \
      --outdir annotation/sheets/baseline_mt
"""
import argparse, json, csv, random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from make_baseline_sheet import write_xlsx, nonleaky_passage, stratified


def main(a):
    rows = [json.loads(l) for l in open(a.instances, encoding="utf-8")]
    sample = stratified(rows, min(a.n, len(rows)), a.seed)
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    qa_cols = []
    for t in range(1, a.max_turns + 1):
        qa_cols += [f"q{t}_yesno", f"q{t}_reply"]
    header = ["instance_id", "company", "annotator_id", "question", "retrieved_passage"] \
        + qa_cols + ["final_answer"]

    for lang in sorted(set(r["language"] for r in sample)):
        sub = [r for r in sample if r["language"] == lang]
        sheet_rows = []
        for idx, r in enumerate(sub):
            rng = random.Random(30_000 + idx)
            sheet_rows.append([r["instance_id"], r.get("company", ""), "",
                               r["question"], nonleaky_passage(r, rng)]
                              + [""] * len(qa_cols) + [""])
        path = outdir / f"baseline_mt_questions_{lang}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(sheet_rows)
        write_xlsx(outdir / f"baseline_mt_questions_{lang}.xlsx", header, sheet_rows)
        print(f"wrote {path}(+.xlsx)  ({len(sub)} {lang}) -- up to {a.max_turns} yes/no rounds")

    keypath = outdir / "baseline_mt_answerkey.csv"
    with open(keypath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "language", "primary_axis", "question",
                    "context_C", "intended_answer", "default_answer"])
        for r in sample:
            w.writerow([r["instance_id"], r["language"], (r.get("axes") or ["?"])[0],
                        r["question"], r["context"], r["answer"], r.get("default_answer", "")])
    print(f"wrote {keypath}  -- EXPERIMENTER ONLY (answer each q{{t}}_reply from C).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--instances", default="data/final/fininteract_v1.jsonl")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--max-turns", type=int, default=6)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--outdir", default="annotation/sheets/baseline_mt")
    main(p.parse_args())
