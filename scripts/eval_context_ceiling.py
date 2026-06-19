"""Context-oracle ceiling: the proper latent-capacity upper bound.

Each model is given the bare question, the *disambiguating context* C (which fixes
the intended interpretation), and a passage containing BOTH the intended and the
default evidence spans (so the data is present but the model must use C to pick the
right value). This isolates the question: given the ambiguity already resolved and
the data in hand, can the model produce the correct answer?

If accuracy here is high while interactive accuracy is low, the bottleneck is the
*interaction* (eliciting C), not knowledge or grounding -- the central claim.

Unlike the built-in template-oracle (which withholds data and forbids search, so
models just emit a search action and score ~0), this provides the data directly.
"""
import argparse, json, random, os
from pathlib import Path
from openai import OpenAI
import sys
sys.path.insert(0, "scripts")
from evaluate import chat, grade, make_client

SYSTEM = ("You are a precise financial analyst. Using the user's question, the "
          "clarifying context, and the disclosure excerpts provided, give ONLY the "
          "single correct value as your final answer (a number with units, a ticker, "
          "or a short phrase). Do not ask questions; the context fully resolves the query.")

def build_user(inst, rng):
    spans = [inst.get("intended_evidence_span", ""), inst.get("default_evidence_span", "")]
    spans = [s for s in spans if s]
    rng.shuffle(spans)  # avoid position bias toward the intended span
    excerpts = "\n".join(f"- {s}" for s in spans)
    return (f"Question: {inst['question']}\n\n"
            f"Clarifying context (the intended interpretation): {inst['context']}\n\n"
            f"Relevant disclosure excerpts:\n{excerpts}\n\n"
            f"Final answer:")

def main(a):
    inst = [json.loads(l) for l in open(a.instances)]
    if a.limit: inst = inst[:a.limit]
    client = make_client(a.model, {"base_url": a.base_url, "api_key": a.api_key} if a.base_url else {})
    grader = OpenAI()  # grader stays on the OpenAI API
    out = open(a.out, "w")
    n=correct=0
    for i, x in enumerate(inst):
        rng = random.Random(1000 + i)  # deterministic per-instance shuffle
        ans = chat(client, a.model, [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_user(x, rng)},
        ], max_tokens=256)
        ok = grade(x["question"], x["answer"], ans, grader)
        n += 1; correct += int(ok)
        rec = {"instance_id": x.get("instance_id"), "model": a.model,
               "mode": "context-oracle", "correct": bool(ok), "final_answer": ans,
               "correct_answer": x["answer"], "language": x.get("language", "en"),
               "axes": x.get("axes", []), "h0": x.get("h0", 0.0)}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
        mark = "✓" if ok else "✗"
        print(f"  [{i+1}/{len(inst)}] {x.get('instance_id')} {mark}  acc={100*correct/n:.1f}%")
    out.close()
    print(f"\n{a.model} context-oracle: {100*correct/n:.1f}%  (n={n})")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--instances", default="data/final/fininteract_v1.jsonl")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--base-url", default=os.environ.get("AGENT_BASE_URL"),
                   help="OpenAI-compatible endpoint for a local agent (vLLM). Grader stays on OpenAI.")
    p.add_argument("--api-key", default=os.environ.get("AGENT_API_KEY", "EMPTY"))
    main(p.parse_args())
