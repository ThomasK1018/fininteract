#!/usr/bin/env python3
"""Agreement between a LOCAL open-model grader and the stored GPT-4o-mini grades.

For the API-free replication (reviewer request) the correctness grader is an open model.
This script re-grades the stored final answers of the canonical runs with that local
grader and reports raw agreement and Cohen's kappa against the stored `correct` labels,
per model and overall, so the local stack's accuracy numbers can be read on the same
scale as the paper's.

Usage:
  python scripts/validate_local_grader.py --judge-base-url http://127.0.0.1:18001/v1 \
      --grader-model judge --out data/results/local_grader_agreement.json
"""
import argparse, glob, json, random, sys
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as ev

FILES = ["data/results/eval_gpt5_gpt4o.jsonl", "data/results/eval_gpt5mini_interact.jsonl",
         "data/results/eval_gpt5mini_search.jsonl", "data/results/eval_open_qwen3p5-35b-a3b.jsonl",
         "data/results/eval_open_qwen3-30b-a3b.jsonl", "data/results/eval_open_qwen3-8b.jsonl"]


def kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-base-url", required=True)
    ap.add_argument("--judge-api-key", default="EMPTY")
    ap.add_argument("--grader-model", default="judge")
    ap.add_argument("--files", nargs="+", default=FILES)
    ap.add_argument("--per-model", type=int, default=150, help="rows sampled per model (all modes)")
    ap.add_argument("--instances", default="data/final/fininteract_v1.jsonl")
    ap.add_argument("--out", default="data/results/local_grader_agreement.json")
    a = ap.parse_args()
    questions = {}
    for line in open(a.instances, encoding="utf-8"):
        if line.strip():
            x = json.loads(line)
            questions[x.get("instance_id", x.get("id"))] = x["question"]
    ev.GRADER_MODEL = a.grader_model
    client = OpenAI(base_url=a.judge_base_url, api_key=a.judge_api_key)
    rows = []
    for f in a.files:
        if not Path(f).exists():
            continue
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("final_answer") and not r.get("error") and not r.get("forced_n"):
                    rows.append(r)
    by = {}
    for r in rows:
        by.setdefault(r["model"], []).append(r)
    rng = random.Random(0)
    report, all_a, all_b = {}, [], []
    for m, rs in by.items():
        # balance correct / incorrect so kappa is informative
        pos = [r for r in rs if r["correct"]]
        neg = [r for r in rs if not r["correct"]]
        k = a.per_model // 2
        sample = rng.sample(pos, min(k, len(pos))) + rng.sample(neg, min(a.per_model - min(k, len(pos)), len(neg)))
        stored, local = [], []
        for r in sample:
            stored.append(int(bool(r["correct"])))
            local.append(int(bool(ev.grade(questions.get(r["instance_id"], ""), r["correct_answer"], r["final_answer"], client))))
        report[m] = dict(n=len(sample), raw_agreement=round(sum(x == y for x, y in zip(stored, local)) / len(sample), 3),
                         cohen_kappa=round(kappa(stored, local), 3),
                         local_says_correct_rate=round(sum(local) / len(sample), 3),
                         stored_correct_rate=round(sum(stored) / len(sample), 3))
        all_a += stored; all_b += local
        print(m, report[m], flush=True)
    report["overall"] = dict(n=len(all_a), raw_agreement=round(sum(x == y for x, y in zip(all_a, all_b)) / len(all_a), 3),
                             cohen_kappa=round(kappa(all_a, all_b), 3))
    print("overall", report["overall"])
    Path(a.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
