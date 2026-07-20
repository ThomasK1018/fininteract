"""Recompute AxisHit@1 on stored trajectories with the CORRECTED classifier.

The original single-label AxisHit classifier under-detected hits (validated agreement 0.53 vs
humans). evaluate.classify_axis_hit is now multi-label (agreement 0.75). This re-classifies the
FIRST stored clarification question per interact instance and reports old vs new AxisHit@1 per
model, without re-running any agents (questions are stored in axis_hits[].question).

No agent calls; only cheap gpt-4o-mini classifier calls (deduped by question text). Routes the
judge through OpenRouter.

Usage:
  python scripts/recompute_axishit.py --results "data/results/eval_*.jsonl" "data/results/or_eval_20.jsonl" \
      --mode answer+search+interact --config configs/openrouter.json --out data/results/axishit_recompute.json
"""
import argparse, glob, json, collections
from pathlib import Path
import sys
sys.path.insert(0, "scripts")
import evaluate as E
from openai import OpenAI


def load(patterns):
    seen = {}
    for pat in patterns:
        for f in glob.glob(pat):
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


def main(a):
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    oai = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
    E.GRADER_MODEL = "openai/gpt-4o-mini"

    rows = [r for r in load(a.results)
            if r.get("mode") == a.mode and r.get("forced_n", 0) == 0 and (r.get("axis_hits") or [])]

    cache = {}   # question -> classifier result (dedup identical questions)
    def classify(q, axes):
        if q not in cache:
            cache[q] = E.classify_axis_hit(q, axes, oai)
        info = cache[q]
        return bool(set(info["axis_pred"].split(",")) & set(axes)) if info["axis_pred"] not in ("generic", "none") else False

    by_model = collections.defaultdict(lambda: {"old": [], "new": []})
    for r in rows:
        first = (r.get("axis_hits") or [None])[0]
        if not first:
            continue
        q = first.get("question", "")
        axes = r.get("axes", [])
        by_model[r["model"]]["old"].append(bool(first.get("is_hit")))
        by_model[r["model"]]["new"].append(classify(q, axes))

    out = {}
    print(f"mode={a.mode} | {len(rows)} interact instances | {len(cache)} unique first-questions\n")
    print(f"{'model':26s} {'n':>4s} {'AHit@1 old':>11s} {'AHit@1 new':>11s} {'delta':>7s}")
    for m in sorted(by_model):
        d = by_model[m]; n = len(d["new"])
        old = sum(d["old"]) / n if n else float("nan")
        new = sum(d["new"]) / n if n else float("nan")
        out[m] = {"n": n, "ahit1_old": round(old, 4), "ahit1_new": round(new, 4)}
        print(f"{m:26s} {n:>4d} {old:>11.3f} {new:>11.3f} {new-old:>+7.3f}")

    Path(a.out).write_text(json.dumps(out, indent=2))
    print("\nwrote", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", default=["data/results/eval_*.jsonl", "data/results/or_eval_20.jsonl"])
    p.add_argument("--mode", default="answer+search+interact")
    p.add_argument("--config", default="configs/openrouter.json")
    p.add_argument("--out", default="data/results/axishit_recompute.json")
    main(p.parse_args())
