"""Export the human-baseline sheets (Task 2: performance ceiling).

The human task mirrors the MODELS' task: given the ambiguous question and a
retrieved passage containing BOTH candidate values (intended + default,
shuffled, neither marked correct -- exactly what the model's `search` returns),
do you recognize the ambiguity and resolve it? This measures DISAMBIGUATION, not
recall, and makes the Human row directly comparable to the +Search / +Interact
model columns.

Produces THREE files for a stratified subset (default 50, language-balanced):
  1. baseline_questions_<lang>.csv  -- what the ANNOTATOR sees: question +
     retrieved_passage (both values, unlabeled) + blank answer columns. The
     resolving context C and the intended/default LABELS are withheld -> no leak.
  2. baseline_answerkey.csv         -- what the EXPERIMENTER holds: intended/
     default answers, context C, interpretations, used to (a) answer the
     annotator's yes/no question from C, and (b) grade later.

Protocol (per instance, two conditions):
  - Search (no ask): annotator answers from the passage, may NOT ask. Tests
    whether humans also commit to the default reading.
  - Interaction: annotator may ask ONE yes/no question; the experimenter answers
    it truthfully from C (Yes/No/IDK). Then the annotator commits an answer.
Record the answer, whether/what they asked, and time taken.

Usage:
  python annotation/make_baseline_sheet.py --n 50 --seed 11 \
      --outdir annotation/sheets/baseline
"""
import argparse, json, csv, random
from pathlib import Path


def nonleaky_passage(inst, rng):
    """Both evidence spans, shuffled, presented as unlabeled retrieved excerpts."""
    spans = [s for s in (inst.get("intended_evidence_span", ""),
                         inst.get("default_evidence_span", "")) if s]
    if len(spans) < 2:
        return (inst.get("passage_text", "") or (spans[0] if spans else ""))[:1200]
    rng.shuffle(spans)
    return "Retrieved excerpts (the query is under-specified; more than one is consistent):\n" + \
           "\n".join(f"[Excerpt {i+1}] {s}" for i, s in enumerate(spans))

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

    # 1) annotator-facing sheets, split by language (question + non-leaky passage)
    for lang in sorted(set(r["language"] for r in sample)):
        sub = [r for r in sample if r["language"] == lang]
        path = outdir / f"baseline_questions_{lang}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["instance_id", "company", "question", "retrieved_passage",
                        "search_answer", "search_seconds",
                        "interact_question_asked", "experimenter_reply",
                        "interact_answer", "interact_seconds"])
            for idx, r in enumerate(sub):
                rng = random.Random(10_000 + idx)
                w.writerow([r["instance_id"], r.get("company", ""), r["question"],
                            nonleaky_passage(r, rng),
                            "", "", "", "", "", ""])
        print(f"wrote {path}  ({len(sub)} {lang} instances) -- annotator-facing, "
              f"question + both-value passage, no answer key")

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
