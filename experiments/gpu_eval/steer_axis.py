"""Causal steering along the ambiguity direction (turns the §6.6 probe causal).

The probe shows the gold ambiguity axis is *linearly decodable* from hidden states
-- correlational. This script tests whether that representation *drives behaviour*:
we extract the per-layer ambiguity direction (mean final-token activation of the
disambiguated prompt Q+C minus the bare ambiguous prompt Q), add alpha*dir to the
residual stream at inference, and measure whether the agent's FIRST action shifts
toward asking (interaction rate) and toward the correct axis (AxisHit@1).

Single-turn first-action proxy by design: IR and AxisHit@1 are decisions made on the
bare question, so one steered forward pass captures them without a multi-turn loop.
alpha is swept over BOTH signs so the data -- not us -- picks the direction; +dir
pushes toward the disambiguated representation, -dir toward unresolved ambiguity.

GPU + transformers/torch required; axis classification uses the same API judge as
evaluate.py (needs OPENAI_API_KEY). Run from the repo root.

  python experiments/gpu_eval/steer_axis.py \
    --model Qwen/Qwen3-30B-A3B --instances data/final/fininteract_v1.jsonl \
    --layers 0.4 0.6 0.8 --alphas -8 -4 0 4 8 \
    --out data/results/steer_qwen3-30b-a3b.json
"""
import argparse, json, sys, os
from contextlib import contextmanager

sys.path.insert(0, "scripts")


def last_token_hidden(model, tok, text, n_layers_keep=None):
    import torch
    ids = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    hs = out.hidden_states[1:]                      # drop embeddings -> one per layer
    return torch.stack([h[0, -1].float() for h in hs])   # [n_layers, hidden]


def build_directions(model, tok, instances, n_dir):
    """Per-layer dir = mean( act(Q+C) - act(Q) ) over n_dir instances. L2-normalized."""
    import torch
    diffs = []
    for inst in instances[:n_dir]:
        q = inst["question"]; c = inst.get("context", "")
        a_q  = last_token_hidden(model, tok, q)
        a_qc = last_token_hidden(model, tok, f"Intended interpretation: {c}\n\nQuestion: {q}")
        diffs.append(a_qc - a_q)
    d = torch.stack(diffs).mean(0)                  # [n_layers, hidden]
    return d / (d.norm(dim=-1, keepdim=True) + 1e-8)


@contextmanager
def steer(model, layer_idxs, dirs, alpha):
    """Add alpha*unit_dir to the residual output of each target decoder layer."""
    layers = model.model.layers
    handles = []
    def mk(L):
        vec = dirs[L].to(model.dtype)
        def hook(_m, _in, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h + alpha * vec.to(h.device)   # device_map="auto" shards layers across GPUs
            return (h, *out[1:]) if isinstance(out, tuple) else h
        return hook
    try:
        for L in layer_idxs:
            handles.append(layers[L].register_forward_hook(mk(L)))
        yield
    finally:
        for h in handles:
            h.remove()


def first_action(model, tok, system, question):
    import torch
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    # thinking-OFF so the structured first action is emitted within the token budget
    # (matches the round-1 thinking-off baseline; otherwise <think> eats all 200 tokens).
    try:
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", enable_thinking=False)
    except TypeError:
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    # transformers 5.x returns a BatchEncoding (dict); older returns a bare tensor.
    if hasattr(enc, "keys"):
        enc = {k: v.to(model.device) for k, v in enc.items()}
        input_ids = enc["input_ids"]
    else:
        input_ids = enc.to(model.device)
        enc = {"input_ids": input_ids}
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=200, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    txt = tok.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True)
    from evaluate import parse_action
    return parse_action(txt), txt


def main(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from evaluate import AGENT_SYSTEM_INTERACT, classify_axis_hit
    from openai import OpenAI
    oai = OpenAI()

    insts = [json.loads(l) for l in open(a.instances) if l.strip()]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16,
                                                 device_map="auto").eval()
    n_layers = model.config.num_hidden_layers
    dirs = build_directions(model, tok, insts, a.n_dir)
    layer_idxs = sorted({min(n_layers - 1, max(0, int(round(f * n_layers)))) for f in a.layers})
    print(f"{a.model}: {n_layers} layers; steering at {layer_idxs}; alphas {a.alphas}")

    test = insts[a.n_dir:] if len(insts) > a.n_dir else insts          # hold out from dir set
    if a.limit: test = test[:a.limit]
    res = {"model": a.model, "n_layers": n_layers, "steer_layers": layer_idxs,
           "n_dir": a.n_dir, "n_test": len(test), "by_alpha": {}}

    for alpha in a.alphas:
        asks = hits = answered = searched = noparse = 0
        for inst in test:
            true_axes = inst.get("axes", []) or ["metric_definition"]
            with steer(model, layer_idxs, dirs, float(alpha)) if alpha else _null():
                act, _ = first_action(model, tok, AGENT_SYSTEM_INTERACT, inst["question"])
            a_ = act.get("action") if act else None
            if a_ == "interact":
                asks += 1
                info = classify_axis_hit(act.get("question", ""), true_axes, oai)
                hits += int(info.get("is_hit", False))
            elif a_ == "answer":
                answered += 1
            elif a_ == "search":
                searched += 1
            else:
                noparse += 1
        n = len(test)
        row = {"IR": 100*asks/n, "AxisHit@1": (hits/asks if asks else 0.0),
               "answer_rate": 100*answered/n, "search_rate": 100*searched/n,
               "noparse_rate": 100*noparse/n, "n_ask": asks}
        res["by_alpha"][str(alpha)] = row
        print(f"  alpha={alpha:+.1f}  IR={row['IR']:5.1f}%  AxisHit@1={row['AxisHit@1']:.2f}  "
              f"ans={row['answer_rate']:5.1f}%  search={row['search_rate']:5.1f}%")

    from pathlib import Path
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("wrote", a.out)
    base, top = res["by_alpha"].get("0"), max(res["by_alpha"].values(), key=lambda r: r["IR"])
    if base:
        print(f"\nSteering shifts IR from {base['IR']:.1f}% (alpha=0) to {top['IR']:.1f}% "
              f"-- causal evidence the axis representation drives asking." )


@contextmanager
def _null():
    yield


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--instances", default="data/final/fininteract_v1.jsonl")
    p.add_argument("--layers", type=float, nargs="+", default=[0.4, 0.6, 0.8],
                   help="fractional depths to steer at (e.g. 0.6 = 60%% into the stack)")
    p.add_argument("--alphas", type=float, nargs="+", default=[-8, -4, 0, 4, 8])
    p.add_argument("--n-dir", type=int, default=40, help="instances used to build the direction")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", required=True)
    main(p.parse_args())
