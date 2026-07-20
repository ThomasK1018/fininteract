"""Per-axis AxisHit@1 recompute (tab:skills) with the corrected classifiers.

The original per-axis targeting table used the temporal-biased single-label classifier.
This recomputes per (model, primary-axis) AxisHit@1 in two forms on the stored first
clarification questions: touch (any predicted axis is a true axis, human-validated) and
central (single most-central axis is a true axis, discriminative but supplementary).

No agent calls. Deduped gpt-4o-mini classifier calls via OpenRouter.

Usage:
  python scripts/recompute_axishit_peraxis.py --config configs/openrouter.json \
      --out data/results/axishit_peraxis.json
"""
import argparse, glob, json, collections
import sys
sys.path.insert(0, "scripts")
import evaluate as E
from openai import OpenAI

MODELS = ["gpt-5", "gpt-4o", "gpt-5-mini", "qwen3-30b-a3b", "qwen3p5-35b-a3b"]
AXES = ["entity_scope", "metric_definition", "recognition_policy", "temporal_scope"]
# --models overrides MODELS at runtime (set in main)
AXSET = {"temporal_scope", "metric_definition", "entity_scope", "filing_vintage", "recognition_policy"}
CENTRAL_SYS = """What is the SINGLE ambiguity axis this clarifying question is PRIMARILY trying
to resolve? Choose the one axis the question most centrally asks the user to decide between,
not axes merely mentioned in passing. Options:
  temporal_scope, metric_definition, entity_scope, filing_vintage, recognition_policy, generic
Output EXACTLY ONE label. No explanation."""


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
    global MODELS
    if a.models:
        MODELS = a.models
    cfg = json.loads(open(a.config, encoding="utf-8").read())
    oai = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
    MODEL = "openai/gpt-4o-mini"
    tcache, ccache = {}, {}

    def touch_axes(q):
        if q not in tcache:
            info = E.classify_axis_hit(q, [], oai)  # true axes not needed for the label set
            tcache[q] = set(info["axis_pred"].split(",")) & AXSET
        return tcache[q]

    def central_axis(q):
        if q not in ccache:
            r = E.chat(oai, MODEL, messages=[{"role": "system", "content": CENTRAL_SYS},
                                             {"role": "user", "content": f"Clarifying question: {q}"}],
                       max_tokens=12)
            t = (r or "").lower().replace("-", "_")
            ccache[q] = next((ax for ax in AXSET if ax in t), None)
        return ccache[q]

    rows = [r for r in load(a.results)
            if r.get("model") in MODELS and r.get("mode") == "answer+search+interact"
            and r.get("forced_n", 0) == 0 and (r.get("axis_hits") or [])]

    # group by (model, primary axis)
    cells = collections.defaultdict(lambda: {"touch": [], "central": [], "n": 0})
    for r in rows:
        axes = r.get("axes", [])
        if not axes:
            continue
        pax = axes[0]
        first = r["axis_hits"][0]
        q = first.get("question", "")
        c = cells[(r["model"], pax)]
        c["n"] += 1
        c["touch"].append(bool(touch_axes(q) & set(axes)))
        c["central"].append(central_axis(q) in set(axes))

    def rate(v):
        return round(sum(v) / len(v), 3) if v else None

    out = {}
    for form in ("touch", "central"):
        print(f"\n=== {form.upper()} AxisHit@1 by primary axis ===")
        print(f"{'axis':20s} " + " ".join(f"{m[:9]:>10s}" for m in MODELS))
        for ax in AXES:
            line = f"{ax:20s} "
            for m in MODELS:
                c = cells.get((m, ax))
                val = rate(c[form]) if c else None
                out.setdefault(m, {}).setdefault(ax, {})[form] = val
                out[m][ax]["n"] = cells.get((m, ax), {}).get("n") if cells.get((m, ax)) else 0
                line += f"{(f'{val:.2f}' if val is not None else '  -  '):>10s} "
            print(line)
    open(a.out, "w").write(json.dumps(out, indent=2))
    print("\nwrote", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", default=["data/results/eval_*.jsonl"])
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--config", default="configs/openrouter.json")
    p.add_argument("--out", default="data/results/axishit_peraxis.json")
    main(p.parse_args())
