"""Judge-human agreement sheets (reviewer point 3a).

Primary accuracy and AxisHit depend on LLM judges (GPT-4o-mini grader + AxisHit
classifier). This exports blind human-annotation sheets over a stratified sample of ACTUAL
model outputs so we can measure judge-human agreement, and the judge's false-positive /
false-negative rates against human consensus.

Two sheets, both blind to the LLM's label:
  (A) answer grading: question + intended gold + model answer -> human marks correct 0/1.
  (B) AxisHit: the model's clarification question + the true primary axis -> human marks
      "does this question target that axis?" 0/1. Pulled from each row's stored `axis_hits`.

A held-back key stores the LLM judge's own labels (`correct`, `is_hit`) for comparison.
Sampling balances correct/incorrect (A) and hit/miss (B) so FP/FN are estimable.

Usage:
  python annotation/make_judge_sheet.py --results "data/results/eval_*.jsonl" \
      --instances data/final/fininteract_v1.jsonl --n-answers 80 --n-axis 80 \
      --outdir annotation/sheets/judge
"""
import argparse, json, glob, csv, random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from make_baseline_sheet import write_xlsx


def load_instances(path):
    m = {}
    for l in open(path, encoding="utf-8"):
        l = l.strip()
        if l:
            d = json.loads(l)
            m[d["instance_id"]] = d
    return m


def load_rows(pattern):
    seen = {}
    for f in sorted(glob.glob(pattern)):
        for l in open(f, encoding="utf-8"):
            l = l.strip()
            if not l:
                continue
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            seen[(r.get("model"), r.get("mode"), r.get("instance_id"), r.get("forced_n", 0))] = r
    return list(seen.values())


def balanced_sample(items, key_fn, n, seed):
    rng = random.Random(seed)
    pos = [x for x in items if key_fn(x)]
    neg = [x for x in items if not key_fn(x)]
    rng.shuffle(pos); rng.shuffle(neg)
    half = n // 2
    out = pos[:half] + neg[:n - half]
    rng.shuffle(out)
    return out


def main(a):
    inst = load_instances(a.instances)
    rows = [r for r in load_rows(a.results) if r.get("final_answer")]
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # ---- (A) answer-grading sheet ----
    ans_items = []
    for r in rows:
        d = inst.get(r["instance_id"])
        if not d:
            continue
        ans_items.append({
            "instance_id": r["instance_id"], "model": r["model"], "mode": r["mode"],
            "language": r.get("language"), "question": d["question"],
            "gold_answer": d["answer"], "model_answer": r["final_answer"],
            "llm_correct": int(bool(r.get("correct"))),
        })
    ans_sample = balanced_sample(ans_items, lambda x: x["llm_correct"], a.n_answers, a.seed)
    HDR_A = ["row_id", "instance_id", "language", "question", "gold_answer",
             "model_answer", "annotator_id", "human_correct_0_1"]
    sheet = [[i, x["instance_id"], x["language"], x["question"], x["gold_answer"],
              x["model_answer"], "", ""] for i, x in enumerate(ans_sample)]
    with open(outdir / "judge_answers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(HDR_A); w.writerows(sheet)
    write_xlsx(outdir / "judge_answers.xlsx", HDR_A, sheet)
    with open(outdir / "judge_answers_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["row_id", "instance_id", "model", "mode", "llm_correct"])
        for i, x in enumerate(ans_sample):
            w.writerow([i, x["instance_id"], x["model"], x["mode"], x["llm_correct"]])
    print(f"(A) wrote judge_answers.csv(+xlsx)  {len(ans_sample)} outputs "
          f"({sum(x['llm_correct'] for x in ans_sample)} judged-correct) + held-back key")

    # ---- (B) AxisHit sheet ----
    axis_items = []
    for r in rows:
        if r.get("mode") != "answer+search+interact":
            continue
        d = inst.get(r["instance_id"])
        if not d:
            continue
        primary = (d.get("axes") or ["?"])[0]
        for h in (r.get("axis_hits") or []):
            q = (h.get("question") or "").strip()
            if q:
                axis_items.append({
                    "instance_id": r["instance_id"], "language": r.get("language"),
                    "clarification_question": q, "true_primary_axis": primary,
                    "llm_is_hit": int(bool(h.get("is_hit"))),
                })
    axis_sample = balanced_sample(axis_items, lambda x: x["llm_is_hit"], a.n_axis, a.seed + 1)
    HDR_B = ["row_id", "instance_id", "language", "clarification_question",
             "true_primary_axis", "annotator_id", "human_is_hit_0_1"]
    sheetb = [[i, x["instance_id"], x["language"], x["clarification_question"],
               x["true_primary_axis"], "", ""] for i, x in enumerate(axis_sample)]
    with open(outdir / "judge_axishit.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(HDR_B); w.writerows(sheetb)
    write_xlsx(outdir / "judge_axishit.xlsx", HDR_B, sheetb)
    with open(outdir / "judge_axishit_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["row_id", "instance_id", "llm_is_hit"])
        for i, x in enumerate(axis_sample):
            w.writerow([i, x["instance_id"], x["llm_is_hit"]])
    print(f"(B) wrote judge_axishit.csv(+xlsx)  {len(axis_sample)} clarifications "
          f"({sum(x['llm_is_hit'] for x in axis_sample)} judged-hit) + held-back key")
    print("\nDouble-annotate blind (>=2 raters, fill annotator_id). Then score_judge_agreement.py.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="data/results/eval_*.jsonl")
    p.add_argument("--instances", default="data/final/fininteract_v1.jsonl")
    p.add_argument("--n-answers", type=int, default=80)
    p.add_argument("--n-axis", type=int, default=80)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--outdir", default="annotation/sheets/judge")
    main(p.parse_args())
