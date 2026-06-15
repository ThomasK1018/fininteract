"""Export the human-baseline sheets (Task 2: performance ceiling).

Produces THREE files for a stratified subset (default 50, language-balanced):
  1. baseline_questions_<lang>.csv  -- what the ANNOTATOR sees: question only,
     plus blank columns for their answer and (interact mode) their one yes/no
     question. NO gold answer, NO context -> no leak.
  2. baseline_answerkey.csv         -- what the EXPERIMENTER holds: instance_id,
     intended/default answers, context C, and the per-axis interpretation, used
     to (a) answer the annotator's yes/no question from C, and (b) grade later.

Protocol (per instance, both conditions):
  - No-interaction: annotator answers the question with the source filing
    available but may NOT ask. Tests whether humans also default.
  - Interaction: annotator may ask ONE yes/no question; the experimenter answers
    it truthfully from C (Yes/No/IDK). Then the annotator commits an answer.
Record the answer, whether/what they asked, and time taken.

Usage:
  python annotation/make_baseline_sheet.py --n 50 --seed 11 \
      --outdir annotation/sheets/baseline
"""
import argparse, json, csv, random
from pathlib import Path

def stratified(rows, n, seed):
    rng = random.Random(seed)
    buckets = {}
    for r in rows:
        buckets.setdefault((r["language"], (r.get("axes") or ["?"])[0]), []).append(r)
    for b in buckets.values():
        rng.shuffle(b)
    out, keys = [], sorted(buckets, key=lambda k: -len(buckets[k]))
    while len(out) < n and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k] and len(out) < n:
                out.append(buckets[k].pop())
    return out

def main(a):
    rows = [json.loads(l) for l in open(a.instances)]
    sample = stratified(rows, min(a.n, len(rows)), a.seed)
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) annotator-facing sheets, split by language (question ONLY)
    for lang in sorted(set(r["language"] for r in sample)):
        sub = [r for r in sample if r["language"] == lang]
        path = outdir / f"baseline_questions_{lang}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["instance_id", "company", "question",
                        "noask_answer", "noask_seconds",
                        "interact_question_asked", "experimenter_reply",
                        "interact_answer", "interact_seconds"])
            for r in sub:
                w.writerow([r["instance_id"], r.get("company", ""), r["question"],
                            "", "", "", "", "", ""])
        print(f"wrote {path}  ({len(sub)} {lang} instances) -- annotator-facing, no answers")

    # 2) experimenter answer key (HOLD BACK from annotators)
    keypath = outdir / "baseline_answerkey.csv"
    with open(keypath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "language", "primary_axis", "question",
                    "context_C", "intended_answer", "default_answer",
                    "intended_interpretation", "default_interpretation"])
        for r in sample:
            w.writerow([r["instance_id"], r["language"], (r.get("axes") or ["?"])[0],
                        r["question"], r["context"], r["answer"],
                        r.get("default_answer", ""),
                        json.dumps(r.get("intended_interpretation", {}), ensure_ascii=False),
                        json.dumps(r.get("default_interpretation", {}), ensure_ascii=False)])
    print(f"wrote {keypath}  -- EXPERIMENTER ONLY (gold answers + C). Do not show annotators.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--instances", default="data/final/fininteract_v1.jsonl")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--outdir", default="annotation/sheets/baseline")
    main(p.parse_args())
