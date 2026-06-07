"""
Constructor-ablation study: how do different LLMs generate under the FinInteract pipeline?

Runs the SAME fixed passage sample through several constructor LLMs while holding the
verifier FIXED (OpenAI gpt-5-mini + gpt-5). This isolates the constructor as the only
variable, so differences in the resulting instances are attributable to the generator model.

For each constructor we measure:
  - acceptance rate            (passed 14-rule QC AND adversarial verifier)
  - QC-failure breakdown       (which rules each model trips)
  - axis distribution          (does the model favor certain ambiguity axes?)
  - mean H0 / difficulty
  - blind-solve (verifier vote) rate of accepted instances
  - mean intended-vs-default answer gap (R14 discrimination strength)

Reproducibility payoff: if weaker/different LLMs also yield valid instances after the same
QC+verifier, the benchmark is a product of the PIPELINE, not of GPT-5 specifically.

Models are addressed by id; namespaced ids (e.g. "anthropic/claude-3.7-sonnet",
"google/gemini-2.5-pro") route via OpenRouter (set OPENROUTER_API_KEY). Native OpenAI ids
(gpt-5, gpt-4o) use OPENAI_API_KEY. See get_model_client() in construct_instances.py.

Usage:
  export OPENAI_API_KEY=...          # verifier (fixed) + native OpenAI constructors
  export OPENROUTER_API_KEY=...      # for anthropic/google/etc constructors
  python scripts/constructor_ablation.py \\
      --passages data/sources/passages.jsonl --sample 40 \\
      --constructors gpt-5 gpt-4o anthropic/claude-3.7-sonnet google/gemini-2.5-pro \\
                     deepseek/deepseek-chat qwen/qwen-2.5-72b-instruct \\
      --out-dir data/ablation
"""

import argparse
import json
import random
import statistics
import re
from collections import Counter, defaultdict
from pathlib import Path

import construct_instances as ci


def _num(s):
    s = str(s).replace(",", "").replace("$", "").replace("%", "").strip()
    m = re.search(r"-?[\d.]+", s)
    if not m:
        return None
    v = float(m.group())
    sl = str(s).lower()
    if "亿" in str(s): v *= 1e8
    elif "b" in sl: v *= 1e9
    elif "m" in sl and "%" not in str(s): v *= 1e6
    return v


def sample_passages(path: Path, n: int, seed: int = 42) -> list[dict]:
    rows = [json.loads(l) for l in path.open() if l.strip()]
    # stratify by (language, primary candidate axis) for a representative shared sample
    strata = defaultdict(list)
    for r in rows:
        if not r.get("candidate_answer"):
            continue
        strata[(r.get("language"), (r.get("candidate_axes") or ["?"])[0])].append(r)
    rng = random.Random(seed)
    out = []
    per = max(1, n // max(len(strata), 1))
    for items in strata.values():
        rng.shuffle(items)
        out.extend(items[:per])
    rng.shuffle(out)
    return out[:n]


def run_constructor(model: str, passages: list[dict], verifier_client) -> dict:
    """Construct from the shared sample with `model`; verifier is the fixed `verifier_client`."""
    try:
        cclient = ci.get_model_client(model)
    except SystemExit as e:
        print(f"  [skip {model}] {e}")
        return {"model": model, "skipped": str(e)}

    accepted, rejected, failed = [], 0, 0
    for k, passage in enumerate(passages):
        inst = ci.construct_one(passage, k + 1, verifier_client, dry_run=False,
                                constructor_client=cclient, constructor_model=model)
        if inst is None:
            failed += 1
        elif inst["qc"]["accepted"]:
            accepted.append(inst)
        else:
            rejected += 1
    return {"model": model, "accepted": accepted,
            "n_rejected": rejected, "n_failed": failed, "n_passages": len(passages)}


def summarize(res: dict) -> dict:
    if res.get("skipped"):
        return {"model": res["model"], "skipped": True}
    acc = res["accepted"]
    n_in = res["n_passages"]
    axes = Counter((i.get("axes") or ["?"])[0] for i in acc)
    votes = [i["qc"].get("n_verifier_votes", i["qc"].get("n_verifier_correct", 0)) for i in acc]
    gaps = []
    for i in acc:
        a, d = _num(i.get("answer")), _num(i.get("default_answer"))
        if a and d and a != 0:
            gaps.append(abs(a - d) / abs(a))
    return {
        "model":            res["model"],
        "acceptance_rate":  len(acc) / max(n_in, 1),
        "n_accepted":       len(acc),
        "n_rejected":       res["n_rejected"],
        "n_failed":         res["n_failed"],
        "axis_dist":        dict(axes),
        "mean_h0":          round(statistics.mean(i["h0"] for i in acc), 2) if acc else None,
        "blind_solve_rate": round(statistics.mean(votes) / 10, 3) if votes else None,
        "mean_answer_gap":  round(statistics.mean(gaps), 3) if gaps else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", type=Path, default=Path("data/sources/passages.jsonl"))
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--constructors", nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/ablation"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    verifier_client = ci.get_client()   # FIXED verifier (OpenAI)
    passages = sample_passages(args.passages, args.sample, args.seed)
    print(f"Shared sample: {len(passages)} passages across "
          f"{len(set((p.get('language'),(p.get('candidate_axes') or ['?'])[0]) for p in passages))} strata")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for model in args.constructors:
        print(f"\n=== constructor: {model} ===")
        res = run_constructor(model, passages, verifier_client)
        if not res.get("skipped"):
            safe = model.replace("/", "_")
            with (args.out_dir / f"instances_{safe}.jsonl").open("w") as f:
                for i in res["accepted"]:
                    f.write(json.dumps(i, ensure_ascii=False) + "\n")
        s = summarize(res)
        summaries.append(s)
        print(f"  acceptance={s.get('acceptance_rate')}  axes={s.get('axis_dist')}  "
              f"blind={s.get('blind_solve_rate')}  gap={s.get('mean_answer_gap')}")

    with (args.out_dir / "ablation_summary.json").open("w") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    # comparison table
    print(f"\n{'='*78}\nCONSTRUCTOR ABLATION  (shared {len(passages)}-passage sample, fixed verifier)\n{'='*78}")
    hdr = f"{'model':<32}{'accept':>8}{'H0':>6}{'blind':>7}{'gap':>7}"
    print(hdr); print("-" * len(hdr))
    for s in summaries:
        if s.get("skipped"):
            print(f"{s['model']:<32}{'(skipped)':>8}")
            continue
        print(f"{s['model']:<32}{s['acceptance_rate']:>7.0%}{str(s['mean_h0']):>6}"
              f"{str(s['blind_solve_rate']):>7}{str(s['mean_answer_gap']):>7}")
    print(f"\nPer-model outputs + ablation_summary.json -> {args.out_dir}")


if __name__ == "__main__":
    main()
