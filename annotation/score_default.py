"""Score the blind default-interpretation sheets (reviewer point 4a) -> illusion premise.

For each committed gold answer, grade it against BOTH the default answer A_d and the
intended answer A_i (reusing evaluate.grade for finance tolerance). Reports:
  - % matching A_d  (the key number: does a blind builder default to the constructed A_d?)
  - % matching A_i
  - % matching neither
  - inter-annotator agreement (Gwet AC1 on the A_d-match label, if >=2 annotators/item)
  - mean per-item answer entropy (bits) across annotators (low => convergent default)
  - % who reported noticing a second defensible answer

Needs OPENAI_API_KEY (or --self-graded with a hand-marked *_matches_default column).

Usage:
  python annotation/score_default.py \
      --sheets annotation/sheets/default/default_questions_en.csv \
               annotation/sheets/default/default_questions_zh.csv \
      --answerkey annotation/sheets/default/default_answerkey.csv \
      --out annotation/default_stats.json
"""
import argparse, csv, json, math, sys, collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from score_baseline import _rows, load_key  # reuse csv/xlsx row reader + key loader


def entropy(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = collections.Counter(labels)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def gwet_ac1(item_labels):
    """Gwet's AC1 for binary labels over items with >=2 raters (chance-corrected)."""
    pos = tot = 0
    pe_terms = []
    n_agree = n_pairs = 0
    for labs in item_labels:
        labs = [l for l in labs if l is not None]
        m = len(labs)
        if m < 2:
            continue
        k1 = sum(labs); k0 = m - k1
        n_agree += k1 * (k1 - 1) + k0 * (k0 - 1)
        n_pairs += m * (m - 1)
        pos += k1; tot += m
    if n_pairs == 0:
        return float("nan")
    pa = n_agree / n_pairs
    pi = pos / tot if tot else 0.0
    pe = 2 * pi * (1 - pi)
    return (pa - pe) / (1 - pe) if pe < 1 else float("nan")


def main(a):
    sys.path.insert(0, "scripts")
    key = load_key(a.answerkey)
    # collect all rows (possibly multiple annotators per instance)
    per_item = collections.defaultdict(list)   # id -> [answer strings]
    per_item_ad = collections.defaultdict(list)  # id -> [matches-default bool]
    noticed = 0; noticed_n = 0
    grade = None
    if not a.self_graded:
        from evaluate import grade as grade_fn
        from openai import OpenAI
        oai = OpenAI()
        grade = lambda q, gold, pred: grade_fn(q, gold, pred, oai)

    n_ad = n_ai = n_neither = n_total = 0
    for p in a.sheets:
        for r in _rows(p):
            iid = (r.get("instance_id") or "").strip()
            pred = (r.get("your_gold_answer") or "").strip()
            if not iid or iid not in key or not pred:
                continue
            q = key[iid]["question"]; ad = key[iid]["default_answer"]; ai = key[iid]["intended_answer"]
            if a.self_graded:
                md = (r.get("your_gold_answer_matches_default", "").strip().lower() in ("1", "yes", "y", "true"))
                mi = (r.get("your_gold_answer_matches_intended", "").strip().lower() in ("1", "yes", "y", "true"))
            else:
                md = grade(q, ad, pred); mi = grade(q, ai, pred)
            n_total += 1
            n_ad += int(md); n_ai += int(mi and not md)
            n_neither += int(not md and not mi)
            per_item[iid].append(pred)
            per_item_ad[iid].append(md)
            nb = (r.get("noticed_second_answer_y_n") or "").strip().lower()
            if nb in ("y", "yes", "n", "no", "1", "0", "true", "false"):
                noticed_n += 1; noticed += int(nb in ("y", "yes", "1", "true"))

    ac1 = gwet_ac1(list(per_item_ad.values()))
    mean_ent = (sum(entropy(v) for v in per_item.values()) / len(per_item)) if per_item else float("nan")
    res = {
        "n_commitments": n_total,
        "n_items": len(per_item),
        "pct_match_default_Ad": round(100 * n_ad / n_total, 1) if n_total else None,
        "pct_match_intended_Ai": round(100 * n_ai / n_total, 1) if n_total else None,
        "pct_match_neither": round(100 * n_neither / n_total, 1) if n_total else None,
        "interannotator_AC1_on_Ad_match": round(ac1, 3) if ac1 == ac1 else None,
        "mean_item_answer_entropy_bits": round(mean_ent, 3) if mean_ent == mean_ent else None,
        "pct_noticed_second_answer": round(100 * noticed / noticed_n, 1) if noticed_n else None,
    }
    print(json.dumps(res, indent=2))
    print("\nReading: high pct_match_default_Ad + low entropy => a conventional single-gold "
          "benchmark would encode the default => the single-gold-illusion premise holds.")
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sheets", nargs="+", required=True)
    p.add_argument("--answerkey", required=True)
    p.add_argument("--self-graded", action="store_true")
    p.add_argument("--out", default="annotation/default_stats.json")
    main(p.parse_args())
