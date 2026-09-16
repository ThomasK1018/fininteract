#!/usr/bin/env python3
"""Exp 6 (reviewer response) -- SFT baseline demo variants, to isolate whether it is
AXIS-GUIDANCE specifically that helps, vs merely learning to ask a clarifying question.

Modes (teacher-prompt tweaks of gen_axis_guided_demos.py; same 7-message ask->search->answer
demo format so the only thing that varies is WHICH clarifying question is taught):
  guided      -- ask on the operative (gold) axis          [ = gen_axis_guided_demos.py ]
  generic     -- teacher gets NO axis hint; asks the obvious clarifying question
  random-axis -- teacher told a deliberately WRONG axis; asks on that
  always-ask  -- force a clarification but generic/non-targeted (ceiling on 'just ask')

Usage:
  python scripts/gen_demos_baselines.py --instances <frontier.jsonl> --axis entity_scope \
      --mode generic --out data/coevolve/baselines/entity_generic.jsonl
"""
import argparse, json, time, random, concurrent.futures as cf
from pathlib import Path
from openai import OpenAI

AXES = ["entity_scope", "metric_definition", "temporal_scope", "recognition_policy", "filing_vintage"]
SYSTEM = (
    "You are an expert financial analyst agent. The user's question may be AMBIGUOUS: the same\n"
    "question can have different correct answers depending on scope, accounting basis, period,\n"
    "entity, or filing version. Do NOT guess. Recognize the ambiguity and resolve it.\n\n"
    "Each turn, output exactly ONE action as a single JSON object:\n"
    '  {"action": "search", "query": "..."}        retrieve from the filing\n'
    '  {"action": "interact", "question": "..."}    ask the user ONE yes/no clarifying question\n'
    '  {"action": "answer", "value": "..."}         give your final answer\n\n'
    "Ask a targeted clarifying question about the specific dimension that is under-specified\n"
    "before answering. Output only the JSON, nothing else."
)
TEACHER_SYS = (
    "You are authoring ONE training demonstration of an analyst agent that asks a single yes/no "
    "clarifying question, then searches, then answers. Output ONLY valid JSON with keys: "
    "clarifying_question, user_reply, search_query, answer_value."
)


def _act(o): return json.dumps(o, ensure_ascii=False, separators=(",", ":"))


def teacher_call(client, model, prompt):
    msgs = [{"role": "system", "content": TEACHER_SYS}, {"role": "user", "content": prompt}]
    reasoning = model.startswith(("gpt-5", "o1", "o3", "o4"))
    for attempt in range(6):
        try:
            kw = dict(model=model, messages=msgs, response_format={"type": "json_object"})
            if reasoning: kw["max_completion_tokens"] = 2048
            else: kw["temperature"] = 0.7; kw["max_tokens"] = 512
            return json.loads(client.chat.completions.create(**kw).choices[0].message.content)
        except Exception:
            time.sleep(min(2 ** attempt, 30))
    return None


def build_demo(ins, axis, mode, client, model, rng):
    intended = ins.get("intended_interpretation") or ins.get("context")
    default = ins.get("default_interpretation") or ins.get("default_answer")
    span = ins.get("intended_evidence_span") or ins.get("context", "")
    ans = ins.get("answer", "")
    common = (f"Ambiguous question: {ins['question']}\n"
              f"Intended interpretation (the user's true meaning): {intended}\n"
              f"Default interpretation (the wrong reading to rule out): {default}\n"
              f"Evidence for the intended answer: {span}\n"
              f"Correct (intended) answer: {ans}\n\n")
    if mode == "guided":
        instr = (f"PRIVATE -- operative ambiguity axis: {axis}\n\nProduce JSON:\n"
                 f"  clarifying_question: ONE yes/no question that pins down the {axis} dimension, "
                 f"distinguishing intended from default. Do NOT mention the axis name.\n")
    elif mode == "random-axis":
        wrong = rng.choice([a for a in AXES if a != axis])
        instr = (f"PRIVATE -- ask about this dimension: {wrong}\n\nProduce JSON:\n"
                 f"  clarifying_question: ONE yes/no question about the {wrong} dimension "
                 f"(do NOT mention the dimension name).\n")
    elif mode == "generic":
        instr = ("Produce JSON:\n"
                 "  clarifying_question: ONE natural yes/no clarifying question you would ask about "
                 "this question, without being told which dimension is under-specified.\n")
    elif mode == "always-ask":
        instr = ("Produce JSON:\n"
                 "  clarifying_question: ONE generic yes/no question confirming the user's intent "
                 "before answering (a reflexive clarification, not necessarily targeted).\n")
    else:
        raise ValueError(mode)
    prompt = common + instr + (
        "  user_reply: 'Yes' or 'No' -- the reply a user who means the INTENDED reading gives.\n"
        "  search_query: a filing search query to retrieve the intended figure.\n"
        "  answer_value: one natural sentence stating the correct answer.")
    t = teacher_call(client, model, prompt)
    if not t or not all(k in t for k in ("clarifying_question", "user_reply", "search_query", "answer_value")):
        return None
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": ins["question"]},
        {"role": "assistant", "content": _act({"action": "interact", "question": t["clarifying_question"]})},
        {"role": "user", "content": str(t["user_reply"]).strip()},
        {"role": "assistant", "content": _act({"action": "search", "query": t["search_query"]})},
        {"role": "user", "content": f"[search result] {span}"},
        {"role": "assistant", "content": _act({"action": "answer", "value": t["answer_value"]})},
    ]
    return {"messages": msgs, "instance_id": ins["instance_id"], "reward": 1.7, "demo_mode": mode}


def main(a):
    client = OpenAI()
    rows = [json.loads(l) for l in open(a.instances) if l.strip()]
    rng = random.Random(a.seed)
    print(f"{len(rows)} instances | mode={a.mode} | axis={a.axis} | teacher={a.teacher_model}")
    demos = []
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(build_demo, r, a.axis, a.mode, client, a.teacher_model, rng): r for r in rows}
        for i, f in enumerate(cf.as_completed(futs), 1):
            try:
                d = f.result()
                if d: demos.append(d)
            except Exception as e:
                print(f"  [skip] {e}")
            if i % 20 == 0: print(f"  {i}/{len(rows)} ({len(demos)} kept)", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fo:
        for d in demos: fo.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"wrote {len(demos)}/{len(rows)} demos -> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--instances", required=True)
    p.add_argument("--axis", required=True)
    p.add_argument("--mode", required=True, choices=["guided", "generic", "random-axis", "always-ask"])
    p.add_argument("--teacher-model", default="gpt-5-mini")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    main(p.parse_args())
