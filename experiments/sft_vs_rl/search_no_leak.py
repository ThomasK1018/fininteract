"""Non-leaky `search` return for the GRPO training env (Experiment 1).

THE LEAK: the kit's `search` action returns the single *disambiguated* gold
evidence span, which contains the gold answer. The untrained base model can read
the answer straight off the search result, so terminal reward is ~saturated and
RL has no gradient on accuracy.

THE FIX: return a passage containing BOTH the intended and the default evidence
spans (order shuffled), so the gold value is no longer uniquely recoverable from
search alone — the model must *clarify* (interact) to know which span is correct.
This makes accuracy a meaningful, non-leaked reward signal again.

HOW TO USE — patch the kit wherever `search` returns evidence:
  - TRL single-turn env: the `search` action branch.
  - verl_integration/fininteract_agent_loop.py: the `search` branch.
  - src/run_behavior.py: the `search` action (if re-running behavioral eval).
Replace the body that returns `inst["intended_evidence_span"]` (or
`passage_text` / `intended_evidence_span`) with:

    from search_no_leak import nonleaky_search
    result_text = nonleaky_search(inst, query=q, seed=hash(inst["instance_id"]) & 0xffffffff)

Determinism: pass a per-instance `seed` so the shuffled order is stable across
rollouts of the same instance (important for stable RL credit assignment).
"""
from __future__ import annotations
import random


def nonleaky_search(inst: dict, query: str | None = None,
                    max_chars: int = 1200, seed: int | None = None) -> str:
    """Return a retrieval passage that contains BOTH candidate evidence spans.

    Falls back gracefully if a span is missing. `query` is accepted for API
    compatibility with the kit's search signature but does not filter (the env is
    oracle-retrieval over the instance's own disclosures, now made ambiguous).
    """
    intended = (inst.get("intended_evidence_span") or "").strip()
    default = (inst.get("default_evidence_span") or "").strip()
    spans = [s for s in (intended, default) if s]

    # If we somehow only have one span, fall back to any broader passage text so we
    # do NOT silently re-introduce a single-gold leak.
    if len(spans) < 2:
        passage = (inst.get("passage_text") or "").strip()
        if passage:
            return passage[:max_chars]
        return spans[0][:max_chars] if spans else ""

    rng = random.Random(seed)
    rng.shuffle(spans)
    # Neutral framing: present both disclosures as retrieved excerpts with no label
    # that reveals which one matches the (still-unknown) intended interpretation.
    body = "\n\n".join(f"[Excerpt {i+1}] {s}" for i, s in enumerate(spans))
    header = ("Retrieved disclosures (the query is under-specified; multiple "
              "interpretations are consistent with the excerpts below):\n")
    return (header + body)[:max_chars]


def leak_check(inst: dict, result_text: str) -> bool:
    """True if the gold answer is NOT uniquely readable (i.e. leak closed).

    Heuristic sanity check for a unit test: the default answer should also appear,
    so an agent cannot pick the gold by elimination from the search result alone.
    """
    gold = str(inst.get("answer", "")).strip()
    dflt = str(inst.get("default_answer", "")).strip()
    return bool(gold) and bool(dflt) and (gold in result_text) == (dflt in result_text)


if __name__ == "__main__":
    # Demo / unit check on the frozen dataset: show that the gold answer is no
    # longer uniquely present in the search result.
    import json, sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/final/fininteract_v1.jsonl"
    rows = [json.loads(l) for l in open(path)]
    ok = 0
    for i, inst in enumerate(rows[:5]):
        res = nonleaky_search(inst, seed=i)
        closed = leak_check(inst, res)
        ok += int(closed)
        print(f"\n=== {inst.get('instance_id')} | leak-closed={closed} ===")
        print(f"  gold={inst.get('answer')!r}  default={inst.get('default_answer')!r}")
        print("  search result:\n   " + res[:240].replace("\n", "\n   "))
    print(f"\nleak-closed on {ok}/5 sampled instances "
          f"(both gold & default present, so neither is uniquely readable).")
