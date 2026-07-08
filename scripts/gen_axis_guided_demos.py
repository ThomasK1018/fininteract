#!/usr/bin/env python3
"""
Axis-guided SFT demo generator for the co-evolution rounds.

Faithful reconstruction of the recipe used for round-0 / entity (whose OUTPUT
`data/coevolve/*/demos/sft.jsonl` is committed, but whose generator was not): for each
frontier instance, a teacher is PRIVATELY told the gold axis and the intended/default
interpretations, and produces a clean ReAct trajectory that (1) asks ONE on-axis yes/no
clarifying question, (2) searches, (3) answers. The private hint is NOT stored in the
saved messages — only the trajectory is. Output format is byte-matched to the committed
entity demos: {"messages":[...], "instance_id":..., "reward":1.7}.

Verify parity before trusting a new round: `head -1` this output vs the committed
`data/coevolve/entity/demos/sft.jsonl` — same message roles, same compact-JSON actions.

Usage:
  python scripts/gen_axis_guided_demos.py --instances data/coevolve/metric/train_W.jsonl \
    --axis metric_definition --teacher-model gpt-5-mini \
    --out data/coevolve/metric/demos/sft.jsonl
  # local teacher: add --teacher-base-url http://localhost:8000/v1
"""
import json, argparse, concurrent.futures as cf, time
from pathlib import Path
from openai import OpenAI

# byte-identical to the committed entity demos' system message
SYSTEM = (
    "You are an expert financial analyst agent. The user's question may be AMBIGUOUS: "
    "the same\nquestion can have different correct answers depending on scope, accounting "
    "basis, period,\nentity, or filing version. Do NOT guess. Recognize the ambiguity and "
    "resolve it.\n\nEach turn, output exactly ONE action as a single JSON object:\n"
    '  {"action": "search", "query": "..."}        retrieve from the filing\n'
    '  {"action": "interact", "question": "..."}    ask the user ONE yes/no clarifying question\n'
    '  {"action": "answer", "value": "..."}         give your final answer\n\n'
    "Ask a targeted clarifying question about the specific dimension that is under-specified\n"
    "before answering. Output only the JSON, nothing else."
)

TEACHER_SYS = (
    "You are authoring ONE training demonstration of an analyst agent that RESOLVES an "
    "ambiguous financial question by asking a single targeted yes/no clarifying question "
    "about a specific ambiguity axis, then searching, then answering. Output ONLY valid "
    "JSON with keys: clarifying_question, user_reply, search_query, answer_value."
)


def _act(obj):  # compact JSON, matching the committed demos exactly
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _is_reasoning(m):
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


def teacher_call(client, model, prompt):
    msgs = [{"role": "system", "content": TEACHER_SYS}, {"role": "user", "content": prompt}]
    for attempt in range(6):
        try:
            kw = dict(model=model, messages=msgs, response_format={"type": "json_object"})
            if _is_reasoning(model):
                kw["max_completion_tokens"] = 2048
            else:
                kw["temperature"] = 0.7; kw["max_tokens"] = 512
            r = client.chat.completions.create(**kw)
            return json.loads(r.choices[0].message.content)
        except Exception:
            time.sleep(min(2 ** attempt, 30))
    return None


def build_demo(ins, axis, client, model):
    intended = ins.get("intended_interpretation") or ins.get("context")
    default = ins.get("default_interpretation") or ins.get("default_answer")
    span = ins.get("intended_evidence_span") or ins.get("context", "")
    ans = ins.get("answer", "")
    prompt = (
        f"Ambiguous question: {ins['question']}\n"
        f"PRIVATE — operative ambiguity axis: {axis}\n"
        f"Intended interpretation (the user's true meaning): {intended}\n"
        f"Default interpretation (the wrong reading to rule out): {default}\n"
        f"Evidence for the intended answer: {span}\n"
        f"Correct (intended) answer: {ans}\n\n"
        f"Produce JSON:\n"
        f"  clarifying_question: ONE yes/no question that pins down the {axis} dimension, "
        f"distinguishing the intended from the default reading. Do NOT mention the axis name.\n"
        f"  user_reply: 'Yes' or 'No' — the reply a user who means the INTENDED reading gives.\n"
        f"  search_query: a filing search query to retrieve the intended figure.\n"
        f"  answer_value: one natural sentence stating the correct answer."
    )
    t = teacher_call(client, model, prompt)
    if not t or not all(k in t for k in ("clarifying_question", "user_reply", "search_query", "answer_value")):
        return None
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": ins["question"]},
        {"role": "assistant", "content": _act({"action": "interact", "question": t["clarifying_question"]})},
        {"role": "user", "content": str(t["user_reply"]).strip()},
        {"role": "assistant", "content": _act({"action": "search", "query": t["search_query"]})},
        {"role": "user", "content": f"[search result] {span}"},
        {"role": "assistant", "content": _act({"action": "answer", "value": t["answer_value"]})},
    ]
    return {"messages": messages, "instance_id": ins["instance_id"], "reward": 1.7}


def main(a):
    client = OpenAI(base_url=a.teacher_base_url) if a.teacher_base_url else OpenAI()
    rows = [json.loads(l) for l in open(a.instances) if l.strip()]
    print(f"{len(rows)} frontier instances | teacher={a.teacher_model} | axis={a.axis}")
    demos = []
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(build_demo, r, a.axis, client, a.teacher_model): r for r in rows}
        for i, f in enumerate(cf.as_completed(futs), 1):
            try:
                d = f.result()
                if d:
                    demos.append(d)
            except Exception as e:
                print(f"  [skip] {e}")
            if i % 20 == 0:
                print(f"  {i}/{len(rows)} ({len(demos)} kept)", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fo:
        for d in demos:
            fo.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"wrote {len(demos)}/{len(rows)} demos -> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--instances", required=True, help="frontier train_W.jsonl for the axis")
    p.add_argument("--axis", required=True)
    p.add_argument("--teacher-model", default="gpt-5-mini")
    p.add_argument("--teacher-base-url", default=None, help="local vLLM endpoint (optional)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", required=True)
    main(p.parse_args())
