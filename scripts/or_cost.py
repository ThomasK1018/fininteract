"""
Price a FinInteract eval run from its token-usage sidecar and project the full run.

Reads the <out>.usage.json written by evaluate.py (per-model prompt/completion tokens),
multiplies by live OpenRouter catalog prices, and extrapolates from the smoke-run
instance count to the full-dataset instance count.

Usage:
    python scripts/or_cost.py --usage data/results/smoke_matrix.usage.json \
        --smoke-per-mode 2 --full-per-mode 173 --n-modes 3 --n-agent-models 7
"""
import argparse
import json
import urllib.request
from pathlib import Path


def catalog(cfg):
    req = urllib.request.Request(cfg["base_url"].rstrip("/") + "/models",
                                 headers={"Authorization": f"Bearer {cfg['api_key']}"})
    data = json.loads(urllib.request.urlopen(req, timeout=30).read())
    return {m["id"]: m.get("pricing", {}) for m in data.get("data", [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usage", required=True)
    ap.add_argument("--config", default="configs/openrouter.json")
    ap.add_argument("--smoke-per-mode", type=int, default=2,
                    help="instances per mode in the measured run")
    ap.add_argument("--full-per-mode", type=int, default=173)
    ap.add_argument("--n-modes", type=int, default=3)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    prices = catalog(cfg)
    usage = json.loads(Path(args.usage).read_text(encoding="utf-8"))

    JUDGE = {"openai/gpt-5", "openai/gpt-4o-mini"}
    scale = args.full_per_mode / args.smoke_per_mode   # per-mode linear scale

    print(f"{'model':<32} {'calls':>6} {'in_tok':>9} {'out_tok':>9} {'$ smoke':>9} {'$ FULL':>10}")
    print("-" * 80)
    tot_smoke = tot_full = tot_judge_full = 0.0
    agent_rows = []
    for model, u in sorted(usage.items()):
        p = prices.get(model, {})
        pin, pout = float(p.get("prompt", 0) or 0), float(p.get("completion", 0) or 0)
        c = u["prompt"] * pin + u["completion"] * pout
        cf = c * scale
        tot_smoke += c
        tot_full += cf
        role = "JUDGE" if model in JUDGE else "AGENT"
        if model in JUDGE:
            tot_judge_full += cf
        else:
            agent_rows.append((model, cf))
        print(f"{model:<32} {u['calls']:>6} {u['prompt']:>9} {u['completion']:>9} "
              f"{c:>8.4f} {cf:>9.2f}  {role}")

    print("-" * 80)
    print(f"{'TOTAL (measured smoke run)':<32} {'':>6} {'':>9} {'':>9} {tot_smoke:>8.4f} {tot_full:>9.2f}")
    print()
    # The judge cost is shared across all agent models; per-agent full cost = its agent
    # cost + a per-agent share of judge. Report both raw and judge-inclusive views.
    n_agents = len(agent_rows)
    judge_share = tot_judge_full / n_agents if n_agents else 0
    print("Per-AGENT-model projected FULL cost (agent tokens + equal share of judge):")
    for m, cf in sorted(agent_rows, key=lambda x: -x[1]):
        print(f"  {m:<30} ${cf + judge_share:>7.2f}   (agent ${cf:.2f} + judge ${judge_share:.2f})")
    print()
    print(f"Judge (gpt-5 sim + gpt-4o-mini grader) full-run total: ${tot_judge_full:.2f}")
    print(f"PROJECTED FULL-RUN TOTAL (all agents, all modes): ${tot_full:.2f}")


if __name__ == "__main__":
    main()
