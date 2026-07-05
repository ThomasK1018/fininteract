#!/usr/bin/env python3
"""
General-domain single-gold-illusion replication on AmbigQA.

Same *methodology* as scripts/evaluate.py (LLM answer-grader + simulated user
responder), with a general-domain grader. Three modes:
  answer-only : bare ambiguous Q -> answer          (measures the illusion)
  oracle      : the disambiguated Q -> answer        (ceiling: knowledge present?)
  interact    : agent may ask ONE clarifying question; an informative user
                responder steers toward the intended reading without revealing
                the answer -> answer                 (measures elicitation)

Per instance we grade the SAME output against:
  - the intended reading            -> intended accuracy
  - every reading                   -> any-valid accuracy (lenient single-gold)
  - every non-intended reading      -> default-capture (gave a defensible but
                                       not-the-specified answer)
The illusion = any-valid >> intended; default-capture is its direct evidence.
"""
import json, argparse, os, time, concurrent.futures as cf
from openai import OpenAI

client = OpenAI()

GRADER_SYS = (
    "You are a strict QA grader. Given a question, a reference answer (with "
    "acceptable aliases), and a model's answer, output 'Yes' if the model's "
    "answer refers to the SAME entity, date, quantity, or fact as the reference "
    "(allow paraphrase, aliases, date-format differences, and partial names that "
    "unambiguously match). Output 'No' otherwise. Output ONLY 'Yes' or 'No'."
)


def _is_reasoning(model):
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def _chat(sys, user, model, temp=0.0, max_tokens=256):
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    last = None
    for attempt in range(6):
        try:
            if _is_reasoning(model):
                # reasoning models: no temperature; reasoning consumes the budget,
                # so give ample room for the visible answer to survive.
                r = client.chat.completions.create(
                    model=model, max_completion_tokens=max(max_tokens, 2048),
                    messages=msgs)
            else:
                r = client.chat.completions.create(
                    model=model, temperature=temp, max_tokens=max_tokens, messages=msgs)
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            last = e
            time.sleep(min(2 ** attempt, 30))  # exp backoff for rate limits
    raise RuntimeError(f"chat failed after retries: {last}")  # fail loud, don't corrupt


def grade(question, aliases, predicted, grader_model):
    if not predicted:
        return False
    ref = " | ".join(aliases)
    out = _chat(GRADER_SYS,
                f"Question: {question}\nReference answer: {ref}\nModel answer: {predicted}",
                grader_model, max_tokens=4)
    return out.strip().lower().startswith("y")


def answer_only(q, model):
    return _chat("Answer the question concisely with just the answer.",
                 f"Question: {q}", model)


def oracle(disambig_q, model):
    return _chat("Answer the question concisely with just the answer.",
                 f"Question: {disambig_q}", model)


def interact(q, intended_disambig, model):
    """One optional clarification round with an informative responder."""
    sys = ("The user's question may be ambiguous. You may ask ONE clarifying "
           "question first (start your reply with 'CLARIFY:') or answer directly "
           "(start with 'ANSWER:'). Keep it to one line.")
    first = _chat(sys, f"Question: {q}", model)
    if first.upper().startswith("CLARIFY"):
        cq = first.split(":", 1)[-1].strip()
        resp_sys = ("You are the user. Your actual intent is exactly this specific "
                    f"reading of your question: \"{intended_disambig}\". Answer the "
                    "agent's clarifying question truthfully to steer them toward THAT "
                    "intent, but DO NOT reveal the final answer. One or two sentences.")
        resp = _chat(resp_sys, f"Agent asks: {cq}", model)
        final = _chat(sys + " You have now received the user's clarification; answer.",
                      f"Question: {q}\nUser clarification: {resp}\nNow give ANSWER:", model)
        asked = True
    else:
        final = first
        asked = False
    if final.upper().startswith("ANSWER"):
        final = final.split(":", 1)[-1].strip()
    return final, asked


def eval_instance(ins, model, grader_model, mode):
    q = ins["question"]
    if mode == "answer-only":
        pred = answer_only(q, model); asked = False
    elif mode == "oracle":
        pred = oracle(ins["context"], model); asked = False
    elif mode == "interact":
        pred, asked = interact(q, ins["context"], model)
    else:
        raise ValueError(mode)

    intended_ok = grade(q, ins["answer_aliases"], pred, grader_model)
    default_ok = any(grade(q, d["aliases"], pred, grader_model)
                     for d in ins["default_answers"])
    any_ok = intended_ok or default_ok
    return {"instance_id": ins["instance_id"], "mode": mode, "model": model,
            "pred": pred, "asked": asked,
            "intended_correct": intended_ok,
            "default_captured": default_ok,
            "any_valid": any_ok}


def main(a):
    rows = [json.loads(l) for l in open(a.instances)]
    if a.sample:
        rows = rows[:: max(1, len(rows) // a.sample)][:a.sample]
    print(f"{len(rows)} instances | model={a.model} | modes={a.modes}")
    results = []
    jobs = [(ins, m) for ins in rows for m in a.modes]
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(eval_instance, ins, a.model, a.grader, m): (ins, m)
                for ins, m in jobs}
        n_fail = 0
        for i, f in enumerate(cf.as_completed(futs), 1):
            try:
                results.append(f.result())
            except Exception as e:
                n_fail += 1
                print(f"  [skip] {e}", flush=True)
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)}  (failed {n_fail})", flush=True)
    if n_fail:
        print(f"WARNING: {n_fail}/{len(jobs)} jobs failed after retries")
    Path = __import__("pathlib").Path
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fo:
        for r in results:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")

    summ = {}
    for m in a.modes:
        sub = [r for r in results if r["mode"] == m]
        n = len(sub)
        summ[m] = {
            "n": n,
            "intended_acc": round(sum(r["intended_correct"] for r in sub) / n, 4),
            "any_valid_acc": round(sum(r["any_valid"] for r in sub) / n, 4),
            "default_capture": round(sum(r["default_captured"] for r in sub) / n, 4),
            "ask_rate": round(sum(r["asked"] for r in sub) / n, 4),
        }
    json.dump({"model": a.model, "summary": summ}, open(a.summary, "w"), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--instances", default="data/general_domain/ambig_instances.jsonl")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--grader", default="gpt-4o-mini")
    p.add_argument("--modes", nargs="+", default=["answer-only", "oracle", "interact"])
    p.add_argument("--sample", type=int, default=120)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="data/general_domain/illusion_gpt4omini.jsonl")
    p.add_argument("--summary", default="data/general_domain/illusion_gpt4omini.summary.json")
    a = p.parse_args()
    main(a)
