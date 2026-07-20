"""Strict 'central-axis' AxisHit + validation, alongside the lenient (any-overlap) variant.

The lenient multi-label classifier (evaluate.classify_axis_hit) matches humans (0.75) but
saturates: clarification questions are compound (name an entity AND a period AND a metric),
so "any predicted axis overlaps a true axis" is almost always true. This adds a STRICT
variant: classify the question by the SINGLE axis it most centrally asks the user to decide,
and count a hit only if that central axis is a true axis. This restores discrimination.

Outputs, per model: AHit@1 old (stored, biased single-label), lenient (any-overlap, human-
validated), central (strict). Also validates the central variant against the human labels.

Usage:
  python scripts/axishit_central.py --annot <dir with judge_axishit*.xlsx> \
      --prior data/results/axishit_recompute.json --out data/results/axishit_central.json
"""
import argparse, glob, json, csv, collections, sys, os
sys.path.insert(0, "scripts"); sys.path.insert(0, "annotation")
import evaluate as E
from openai import OpenAI
from score_baseline import _rows

CENTRAL_SYS = """What is the SINGLE ambiguity axis this clarifying question is PRIMARILY trying
to resolve? Choose the one axis the question most centrally asks the user to decide between,
not axes merely mentioned in passing. Options:
  temporal_scope, metric_definition, entity_scope, filing_vintage, recognition_policy, generic
Output EXACTLY ONE label. No explanation."""
AXES = {"temporal_scope", "metric_definition", "entity_scope", "filing_vintage", "recognition_policy"}


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
    cfg = json.loads(open(a.config, encoding="utf-8").read())
    oai = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
    MODEL = "openai/gpt-4o-mini"
    cache = {}
    def central(q):
        if q not in cache:
            r = E.chat(oai, MODEL, messages=[{"role": "system", "content": CENTRAL_SYS},
                                             {"role": "user", "content": f"Clarifying question: {q}"}],
                       max_tokens=12)
            t = (r or "").lower().replace("-", "_")
            hit = next((ax for ax in AXES if ax in t), None)
            cache[q] = hit
        return cache[q]

    # ---- validation against human labels ----
    lab = collections.defaultdict(list); meta = {}
    for p in glob.glob(f"{a.annot}/judge_axishit*.xlsx"):
        for r in _rows(p):
            rid = (r.get("row_id") or "").strip()
            if rid == "":
                continue
            v = (r.get("human_is_hit_0_1") or "").strip()
            if v in ("0", "1"):
                lab[rid].append(int(v))
            meta[rid] = (r.get("clarification_question"), r.get("true_primary_axis"))
    H, M = [], []
    for rid, vs in lab.items():
        q, ax = meta[rid]
        H.append(int(sum(vs) > len(vs) / 2))
        M.append(int(central(q) == ax))
    n = len(H); po = sum(x == y for x, y in zip(H, M)) / n
    p1, q1 = sum(H) / n, sum(M) / n; pe = p1 * q1 + (1 - p1) * (1 - q1)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    print(f"CENTRAL validation vs humans: n={n} agree={po:.3f} kappa={kappa:.3f} "
          f"(human-hit={p1:.2f} central-hit={q1:.2f})\n")

    # ---- recompute central AHit@1 per model ----
    prior = json.loads(open(a.prior, encoding="utf-8").read()) if os.path.exists(a.prior) else {}
    rows = [r for r in load(a.results)
            if r.get("mode") == "answer+search+interact" and r.get("forced_n", 0) == 0 and (r.get("axis_hits") or [])]
    by = collections.defaultdict(list)
    for r in rows:
        first = (r.get("axis_hits") or [None])[0]
        if first:
            by[r["model"]].append((first.get("question", ""), r.get("axes", [])))
    out = {}
    print(f"{'model':26s} {'n':>4s} {'old':>6s} {'lenient':>8s} {'central':>8s}")
    for m in sorted(by):
        items = by[m]; n = len(items)
        cen = sum(1 for q, ax in items if central(q) in set(ax)) / n
        old = prior.get(m, {}).get("ahit1_old")
        len_ = prior.get(m, {}).get("ahit1_new")
        out[m] = {"n": n, "ahit1_old": old, "ahit1_lenient": len_, "ahit1_central": round(cen, 4)}
        print(f"{m:26s} {n:>4d} {(f'{old:.3f}' if old is not None else '  -  '):>6s} "
              f"{(f'{len_:.3f}' if len_ is not None else '  -  '):>8s} {cen:>8.3f}")
    out["_central_validation"] = {"n": len(H), "agreement": round(po, 3), "kappa": round(kappa, 3),
                                  "human_hit_rate": round(p1, 3), "central_hit_rate": round(q1, 3)}
    open(a.out, "w").write(json.dumps(out, indent=2))
    print("\nwrote", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--annot", required=True)
    p.add_argument("--results", nargs="+", default=["data/results/eval_*.jsonl", "data/results/or_eval_20.jsonl"])
    p.add_argument("--prior", default="data/results/axishit_recompute.json")
    p.add_argument("--config", default="configs/openrouter.json")
    p.add_argument("--out", default="data/results/axishit_central.json")
    main(p.parse_args())
