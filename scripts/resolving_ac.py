#!/usr/bin/env python3
"""Resolution-sufficient clarification (strict AC): does the question actually resolve it?

Reviewer concern: the touch form of AC@1 credits a compound question that merely
*mentions* a relevant category, so a high AC@1 does not show that the question would
have resolved the ambiguity, and E5 "misintegration" may hide inadequate clarification.

This script re-scores every stored clarification question with a stricter judge that
sees the question Q, the intended and default interpretations, and the clarifying
question, and decides whether a truthful yes/no answer to that question would let the
agent know WHICH of the two readings is meant (resolves = true). It then reports, per
model:
  RAC@1     : first question is resolving
  RAC@any   : at least one asked question is resolving
  and decomposes the wrong-answer-after-asking cases (the E5 bucket) into
  E5-resolving   (a resolving question was asked and answered, still wrong = integration)
  E5-nonresolving (only non-resolving questions were asked = inadequate clarification)

Usage:
  OPENAI_API_KEY=... python scripts/resolving_ac.py \
      --files data/results/eval_gpt5_gpt4o.jsonl data/results/eval_gpt5mini_interact.jsonl \
              data/results/eval_open_qwen3p5-35b-a3b.jsonl data/results/eval_open_qwen3-30b-a3b.jsonl \
      --instances data/final/fininteract_v1.jsonl --out data/results/resolving_ac.json
"""
import argparse, json, sys, time, collections, concurrent.futures as cf
from pathlib import Path
from openai import OpenAI

JUDGE_MODEL = "gpt-4o-mini"

SYSTEM = """You judge whether a clarifying question would RESOLVE an ambiguity.

You are given an ambiguous financial question, two candidate readings (INTENDED and DEFAULT,
each a structured record of entity, period, metric, and accounting basis), and one yes/no
clarifying question that an agent asked the user. Assume the user answers the clarifying
question truthfully according to the INTENDED reading.

Decide: after hearing that truthful answer, would the agent know which of the two readings
is meant? A question resolves the ambiguity only if the two readings give DIFFERENT truthful
answers to it (for example, if the readings differ in entity scope and the question asks
about entity scope, it resolves; a question about the period when both readings share the
same period does NOT resolve; a question that names several things but whose yes/no answer
would be the same under both readings does NOT resolve).

Reply with ONLY JSON: {"resolves": true} or {"resolves": false}."""

USER_TMPL = """Ambiguous question: {q}

INTENDED reading: {intended}
DEFAULT reading:  {default}
(The disambiguating context for the intended reading: {context})

Agent's clarifying question: {cq}

Would a truthful answer to the agent's question (given under the INTENDED reading) tell the
agent which of the two readings is meant?"""


def judge(client, inst, cq):
    prompt = USER_TMPL.format(q=inst["question"],
                              intended=json.dumps(inst.get("intended_interpretation", {}), ensure_ascii=False),
                              default=json.dumps(inst.get("default_interpretation", {}), ensure_ascii=False),
                              context=inst.get("context", ""), cq=cq)
    for attempt in range(5):
        try:
            r = client.chat.completions.create(
                model=JUDGE_MODEL, temperature=0,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                max_tokens=20)
            txt = (r.choices[0].message.content or "").strip()
            return bool(json.loads(txt[txt.find("{"):txt.rfind("}") + 1]).get("resolves"))
        except Exception as e:
            if "insufficient_quota" in str(e):
                raise
            time.sleep(3 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--instances", default="data/final/fininteract_v1.jsonl")
    ap.add_argument("--out", default="data/results/resolving_ac.json")
    ap.add_argument("--mode", default="answer+search+interact")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    insts = {}
    for line in open(a.instances, encoding="utf-8"):
        if line.strip():
            x = json.loads(line)
            insts[x.get("instance_id", x.get("id"))] = x
    client = OpenAI()
    cache_path = Path(a.out).with_suffix(".cache.json")
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    rows = []
    for f in a.files:
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("mode") != a.mode or r.get("forced_n") or r.get("user_sim") not in (None, "llm"):
                continue
            if r["instance_id"] in insts:
                rows.append(r)
    # unique (instance, question) pairs to judge
    todo = {}
    for r in rows:
        for h in r.get("axis_hits", []):
            key = r["instance_id"] + "||" + h.get("question", "")
            if key not in cache and h.get("question"):
                todo[key] = (insts[r["instance_id"]], h["question"])
    print(f"{len(rows)} trajectories, {len(todo)} questions to judge ({len(cache)} cached)")
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(judge, client, inst, cq): key for key, (inst, cq) in todo.items()}
        for i, fut in enumerate(cf.as_completed(futs)):
            cache[futs[fut]] = fut.result()
            if (i + 1) % 100 == 0:
                cache_path.write_text(json.dumps(cache), encoding="utf-8")
                print(f"  judged {i+1}/{len(todo)}", flush=True)
    cache_path.write_text(json.dumps(cache), encoding="utf-8")

    report = {}
    by_model = collections.defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    for m, rs in by_model.items():
        asked = [r for r in rs if r.get("axis_hits")]
        def res(r, first_only):
            hs = r["axis_hits"][:1] if first_only else r["axis_hits"]
            return any(cache.get(r["instance_id"] + "||" + h.get("question", "")) for h in hs)
        n = len(rs)
        rac1 = sum(res(r, True) for r in asked) / max(1, len(asked))
        racany = sum(res(r, False) for r in asked) / max(1, len(asked))
        touch1 = sum(bool(r["axis_hits"][0].get("is_hit")) for r in asked) / max(1, len(asked))
        # E5-style bucket: asked, at least one touch hit, wrong
        e5 = [r for r in asked if not r["correct"] and any(h.get("is_hit") for h in r["axis_hits"])]
        e5_res = sum(res(r, False) for r in e5)
        wrong_asked = [r for r in asked if not r["correct"]]
        wrong_res = sum(res(r, False) for r in wrong_asked)
        correct_asked = [r for r in asked if r["correct"]]
        corr_res = sum(res(r, False) for r in correct_asked)
        report[m] = dict(
            n=n, n_asked=len(asked), touch_AC1=round(touch1, 3),
            resolving_AC1=round(rac1, 3), resolving_ACany=round(racany, 3),
            E5_touch_wrong=len(e5), E5_resolving=e5_res, E5_nonresolving=len(e5) - e5_res,
            wrong_after_ask=len(wrong_asked), wrong_after_resolving_ask=wrong_res,
            correct_after_ask=len(correct_asked), correct_after_resolving_ask=corr_res,
            acc_given_resolving=round(corr_res / max(1, corr_res + wrong_res), 3),
            acc_given_nonresolving=round((len(correct_asked) - corr_res) /
                                         max(1, (len(correct_asked) - corr_res) + (len(wrong_asked) - wrong_res)), 3),
        )
    Path(a.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n| Model | n asked | touch AC@1 | resolving AC@1 | resolving AC@any | E5 (touch, wrong) | of which resolving | acc | resolving ask | acc | non-resolving ask |")
    print("|---|---|---|---|---|---|---|---|---|")
    for m, d in report.items():
        print(f"| {m} | {d['n_asked']} | {d['touch_AC1']:.2f} | {d['resolving_AC1']:.2f} | {d['resolving_ACany']:.2f} | "
              f"{d['E5_touch_wrong']} | {d['E5_resolving']} ({100*d['E5_resolving']/max(1,d['E5_touch_wrong']):.0f}%) | "
              f"{100*d['acc_given_resolving']:.1f}% | {100*d['acc_given_nonresolving']:.1f}% |")


if __name__ == "__main__":
    main()
