"""Score judge-human agreement (reviewer point 3a).

Compares human labels (blind) against the stored LLM-judge labels for (A) answer grading
and (B) AxisHit. Reports, per task:
  - n, % raw agreement, Cohen's kappa, Gwet AC1
  - judge FALSE-POSITIVE rate  (judge=1, human-consensus=0)
  - judge FALSE-NEGATIVE rate  (judge=0, human-consensus=1)
  - human-human agreement (if >=2 annotators) as a reliability ceiling
Human consensus = majority vote across annotators per row (ties dropped).

Usage:
  python annotation/score_judge_agreement.py \
      --answers annotation/sheets/judge/judge_answers.csv \
      --answers-key annotation/sheets/judge/judge_answers_key.csv \
      --axis annotation/sheets/judge/judge_axishit.csv \
      --axis-key annotation/sheets/judge/judge_axishit_key.csv \
      --out annotation/judge_agreement_stats.json
"""
import argparse, csv, json, collections, math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from score_baseline import _rows


def to01(v):
    v = (v or "").strip().lower()
    if v in ("1", "yes", "y", "true", "correct", "hit"):
        return 1
    if v in ("0", "no", "n", "false", "wrong", "miss"):
        return 0
    return None


def cohen_kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa1 = sum(a) / n; pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def gwet_ac1(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pi = (sum(a) + sum(b)) / (2 * n)
    pe = 2 * pi * (1 - pi)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def consensus(labels_by_row):
    """row_id -> majority human label (None on tie / no labels)."""
    out = {}
    for rid, labs in labels_by_row.items():
        labs = [l for l in labs if l is not None]
        if not labs:
            continue
        ones = sum(labs)
        if 2 * ones == len(labs):
            continue  # tie
        out[rid] = int(2 * ones > len(labs))
    return out


def human_human_agreement(labels_by_row):
    """Mean pairwise raw agreement across rows with >=2 raters."""
    agr = tot = 0
    for labs in labels_by_row.values():
        labs = [l for l in labs if l is not None]
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                tot += 1; agr += int(labs[i] == labs[j])
    return (agr / tot) if tot else float("nan")


def score_task(sheet_paths, key_path, human_col, llm_col, name):
    key = {row["row_id"]: to01(row[llm_col]) for row in csv.DictReader(open(key_path, encoding="utf-8"))}
    labels_by_row = collections.defaultdict(list)
    for p in sheet_paths:
        for r in _rows(p):
            rid = (r.get("row_id") or "").strip()
            if rid == "":
                continue
            labels_by_row[rid].append(to01(r.get(human_col)))
    cons = consensus(labels_by_row)
    rows = [rid for rid in cons if rid in key and key[rid] is not None]
    h = [cons[rid] for rid in rows]
    m = [key[rid] for rid in rows]
    n = len(rows)
    fp = sum(1 for hi, mi in zip(h, m) if mi == 1 and hi == 0)
    fn = sum(1 for hi, mi in zip(h, m) if mi == 0 and hi == 1)
    judge_pos = sum(mi == 1 for mi in m); judge_neg = sum(mi == 0 for mi in m)
    res = {
        "task": name, "n": n,
        "raw_agreement": round(sum(hi == mi for hi, mi in zip(h, m)) / n, 3) if n else None,
        "cohen_kappa": round(cohen_kappa(h, m), 3) if n else None,
        "gwet_ac1": round(gwet_ac1(h, m), 3) if n else None,
        "judge_false_positive_rate": round(fp / judge_pos, 3) if judge_pos else None,
        "judge_false_negative_rate": round(fn / judge_neg, 3) if judge_neg else None,
        "human_human_agreement": round(human_human_agreement(labels_by_row), 3),
    }
    print(json.dumps(res, indent=2))
    return res


def main(a):
    out = {}
    if a.answers and a.answers_key:
        out["answer_grading"] = score_task(a.answers, a.answers_key,
                                            "human_correct_0_1", "llm_correct", "answer_grading")
    if a.axis and a.axis_key:
        out["axishit"] = score_task(a.axis, a.axis_key,
                                    "human_is_hit_0_1", "llm_is_hit", "axishit")
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("\nwrote", a.out)
    print("If judge FP/FN are low and kappa is high (near human-human agreement), the "
          "LLM-judged headline numbers stand under human labels.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--answers", nargs="+")
    p.add_argument("--answers-key")
    p.add_argument("--axis", nargs="+")
    p.add_argument("--axis-key")
    p.add_argument("--out", default="annotation/judge_agreement_stats.json")
    main(p.parse_args())
