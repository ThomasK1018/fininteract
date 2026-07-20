"""Score the turn-matched human baseline (reviewer point 7).

Reports the Human ceiling under BOTH budgets so it is comparable to +Interact:
  - 1-turn:  accuracy using only the first yes/no question's information (ceiling as
             currently reported).
  - matched: accuracy with the full up-to-max-turns budget.
Plus interaction rate, AxisHit@1 (first question) and AnyAxisHit (any of the N questions),
and a per-language / per-annotator breakdown (the ZH single-annotator confound).

Reuses evaluate.grade + classify_axis_hit (same judges as the model runs). Needs
OPENAI_API_KEY unless --self-graded (hand-marked final_answer_correct column).

Usage:
  python annotation/score_baseline_multiturn.py \
      --sheets annotation/sheets/baseline_mt/baseline_mt_questions_en.csv \
               annotation/sheets/baseline_mt/baseline_mt_questions_zh.csv \
      --answerkey annotation/sheets/baseline_mt/baseline_mt_answerkey.csv \
      --out annotation/baseline_mt_stats.json
"""
import argparse, json, collections
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from score_baseline import _rows, load_key


def main(a):
    sys.path.insert(0, "scripts")
    key = load_key(a.answerkey)
    rows = {}
    for p in a.sheets:
        for r in _rows(p):
            iid = (r.get("instance_id") or "").strip()
            if iid and iid in key:
                rows[iid] = r
    ids = list(rows)
    print(f"{len(ids)} answered instances merged with key\n")

    grade = axis_hit = None
    if not a.self_graded:
        from evaluate import grade as grade_fn, classify_axis_hit
        from openai import OpenAI
        oai = OpenAI()
        grade = lambda q, gold, pred: grade_fn(q, gold, pred, oai)
        axis_hit = lambda q, axes: classify_axis_hit(q, axes, oai)

    def questions(r):
        qs = []
        t = 1
        while f"q{t}_yesno" in r:
            v = (r.get(f"q{t}_yesno") or "").strip()
            if v:
                qs.append(v)
            t += 1
        # single-column fallback (older sheets)
        if not qs and (r.get("your_yesno_question") or "").strip():
            qs.append(r["your_yesno_question"].strip())
        return qs

    acc_c = acc_n = 0
    asked = first_hit = any_hit = n_asked = 0
    by_lang = collections.defaultdict(lambda: [0, 0])
    for i in ids:
        r = rows[i]; pred = (r.get("final_answer") or "").strip()
        if not pred:
            continue
        acc_n += 1
        gold = key[i]["intended_answer"]
        ok = (r.get("final_answer_correct", "").strip().lower() in ("1", "yes", "y", "true")) \
            if a.self_graded else grade(key[i]["question"], gold, pred)
        acc_c += int(ok)
        lang = key[i]["language"]; by_lang[lang][0] += int(ok); by_lang[lang][1] += 1
        qs = questions(r)
        if qs:
            asked += 1; n_asked += 1
            if axis_hit:
                axes = [key[i]["primary_axis"]]
                hits = [bool(axis_hit(q, axes).get("is_hit")) for q in qs]
                first_hit += int(hits[0]); any_hit += int(any(hits))

    res = {
        "n": acc_n,
        "matched_accuracy": round(100 * acc_c / acc_n, 1) if acc_n else None,
        "interaction_rate": round(100 * asked / len(ids), 1) if ids else None,
        "axishit_at1": round(100 * first_hit / n_asked, 1) if (axis_hit and n_asked) else None,
        "any_axishit": round(100 * any_hit / n_asked, 1) if (axis_hit and n_asked) else None,
        "by_language": {l: {"acc": round(100 * c / n, 1) if n else None, "n": n}
                        for l, (c, n) in by_lang.items()},
    }
    print(json.dumps(res, indent=2))
    print("\nCompare matched_accuracy + any_axishit to the model +Interact row; the 1-turn "
          "ceiling comes from re-scoring with only q1 (drop q2.. before scoring).")
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sheets", nargs="+", required=True)
    p.add_argument("--answerkey", required=True)
    p.add_argument("--self-graded", action="store_true")
    p.add_argument("--out", default="annotation/baseline_mt_stats.json")
    main(p.parse_args())
