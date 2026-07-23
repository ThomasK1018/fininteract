"""Per-model error-type distribution (paper/fig_errordist.png + data/results/errordist.json).

Classifies every interact-mode instance into the error taxonomy of Table~\ref{tab:errors},
using the corrected touch axis classifier (evaluate.classify_axis_hit) so attribution is
consistent with the AxisHit@1 reported elsewhere:

  Correct            answer matches the intended interpretation
  E1 blindness       wrong, and the agent never asked (n_asks == 0)
  E2 wrong-category      wrong, asked, first question targets no true category
  E3 generic         wrong, asked, first question is generic (no axis)
  E5 misintegration  wrong, asked, first question touches a true axis (right ask, wrong answer)

E6 (evidence grounding) and E7 (over-interaction) require oracle/efficiency joins and are
not separated here. Cheap gpt-4o-mini classifier calls, deduped by question text.

Usage:
  python scripts/plot_error_distribution.py --config configs/openrouter.json
"""
import argparse, glob, json, collections, sys
sys.path.insert(0, "scripts")
import evaluate as E
from openai import OpenAI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["gpt-5", "gpt-4o", "gpt-5-mini", "qwen3-30b-a3b", "qwen3p5-35b-a3b"]
LABEL = {"gpt-5": "GPT-5", "gpt-4o": "GPT-4o", "gpt-5-mini": "GPT-5-mini",
         "qwen3-30b-a3b": "Qwen3-30B", "qwen3p5-35b-a3b": "Qwen3.5-35B"}
CATS = ["Correct", "E1 blindness", "E2 wrong-category", "E3 generic", "E5 misintegration"]
CCOL = {"Correct": "#2a9d8f", "E1 blindness": "#264653", "E2 wrong-category": "#e76f51",
        "E3 generic": "#f4a261", "E5 misintegration": "#8a5a83"}
AXSET = {"temporal_scope", "metric_definition", "entity_scope", "filing_vintage", "recognition_policy"}


def load(pats):
    seen = {}
    for pat in pats:
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
    E.GRADER_MODEL = "openai/gpt-4o-mini"
    cache = {}

    def classify(q):
        if q not in cache:
            cache[q] = E.classify_axis_hit(q, [], oai)["axis_pred"]
        return cache[q]

    rows = [r for r in load(a.results)
            if r.get("model") in MODELS and r.get("mode") == "answer+search+interact"
            and r.get("forced_n", 0) == 0]
    dist = {m: collections.Counter() for m in MODELS}
    for r in rows:
        m = r["model"]
        if r.get("correct"):
            dist[m]["Correct"] += 1
            continue
        if not r.get("n_asks"):
            dist[m]["E1 blindness"] += 1
            continue
        first = (r.get("axis_hits") or [None])[0]
        q = (first or {}).get("question", "") if first else ""
        pred = classify(q) if q else "generic"
        preds = set(pred.split(",")) & AXSET
        true = set(r.get("axes", []))
        if not preds:
            dist[m]["E3 generic"] += 1
        elif preds & true:
            dist[m]["E5 misintegration"] += 1
        else:
            dist[m]["E2 wrong-category"] += 1

    # normalize + save
    out = {}
    for m in MODELS:
        tot = sum(dist[m].values()) or 1
        out[m] = {c: round(dist[m][c] / tot, 4) for c in CATS}
    open(a.out_json, "w").write(json.dumps(out, indent=2))

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    x = range(len(MODELS))
    bottom = [0.0] * len(MODELS)
    for c in CATS:
        vals = [out[m][c] * 100 for m in MODELS]
        ax.bar(x, vals, bottom=bottom, color=CCOL[c], label=c, width=0.62)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABEL[m] for m in MODELS], fontsize=9.5)
    ax.set_ylabel("Percent of instances")
    ax.set_ylim(0, 100)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print("wrote", a.out, "and", a.out_json)
    for m in MODELS:
        print(f"{LABEL[m]:12s}", {c: f"{out[m][c]:.2f}" for c in CATS})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", default=["data/results/eval_*.jsonl"])
    p.add_argument("--config", default="configs/openrouter.json")
    p.add_argument("--out", default="paper/fig_errordist.png")
    p.add_argument("--out-json", default="data/results/errordist.json")
    main(p.parse_args())
