"""
Fast generation-flow for LARGE / multi-GPU models.

Same output as flow_generate.py, but instead of `generate(output_hidden_states=True)`
(which accumulates per-step hidden states for every layer and is pathologically slow on a
device_map-sharded model), we:
  1. generate the response normally (fast, KV-cached),
  2. run ONE forward pass over [prompt + generated] with output_hidden_states,
  3. project the hidden state at each generating position onto the ambiguity direction.

The hidden state at sequence position p produces token p+1, so positions
[prompt_len-1 .. prompt_len-1+n_gen) are exactly the states that wrote each generated token —
equivalent to the per-step last-token states the original captures, at a fraction of the cost.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CLARIFY_RE = re.compile(r"\?|which|specify|clarif|please provide|哪|请|具体|年份|期间", re.I)


def ambiguity_directions(acts_path: Path):
    d = np.load(acts_path, allow_pickle=True)
    amb, dis = d["amb_last"], d["dis_last"]
    mu_amb, mu_dis = amb.mean(0), dis.mean(0)
    direction = mu_amb - mu_dis
    direction /= (np.linalg.norm(direction, axis=-1, keepdims=True) + 1e-8)
    midpoint = 0.5 * (mu_amb + mu_dis)
    return direction.astype(np.float32), midpoint.astype(np.float32)


def build_prompt(tok, question: str):
    return tok.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct")
    ap.add_argument("--acts", type=Path, required=True)
    ap.add_argument("--instances", type=Path, default=Path("data/fininteract_v1.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/interp/genflow.npz"))
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    direction, midpoint = ambiguity_directions(args.acts)
    dir_t = torch.tensor(direction); mid_t = torch.tensor(midpoint)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    load_kwargs = dict(device_map="auto")
    if args.dtype != "auto":
        load_kwargs["torch_dtype"] = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()

    insts = [json.loads(l) for l in args.instances.open() if l.strip()]
    if args.limit:
        insts = insts[:args.limit]

    dvc = model.device
    dir_d, mid_d = dir_t.to(dvc).float(), mid_t.to(dvc).float()

    flows, ids, axes, gen_texts, onsets, lengths = [], [], [], [], [], []
    for k, inst in enumerate(insts):
        text = build_prompt(tok, inst["question"])
        ids_in = tok(text, return_tensors="pt").to(dvc)
        plen = ids_in["input_ids"].shape[1]
        with torch.no_grad():
            seq = model.generate(**ids_in, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)[0]
        gen_ids = seq[plen:]
        n_gen = int(gen_ids.shape[0])
        gen_tokens = [tok.decode([t]) for t in gen_ids]
        gen_text = tok.decode(gen_ids, skip_special_tokens=True)

        if n_gen == 0:
            continue
        with torch.no_grad():
            out = model(seq.unsqueeze(0).to(dvc), output_hidden_states=True)
        hs = out.hidden_states                      # tuple L+1 of [1, seq, H]
        # positions that generate tokens 0..n_gen-1
        p0 = plen - 1
        pos = list(range(p0, p0 + n_gen))
        flow = []                                   # [n_gen, L]
        for p in pos:
            vecs = torch.stack([h[0, p, :].float() for h in hs])     # [L, H]
            proj = ((vecs - mid_d) * dir_d).sum(-1)                  # [L]
            flow.append(proj.cpu().numpy())
        flow = np.asarray(flow, dtype=np.float32).T                 # [L, n_gen]

        onset = next((i for i, t in enumerate(gen_tokens) if CLARIFY_RE.search(t)), -1)
        flows.append(flow); ids.append(inst["instance_id"])
        axes.append((inst.get("axes") or ["?"])[0])
        gen_texts.append(gen_text); onsets.append(onset); lengths.append(n_gen)
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(insts)}", flush=True)

    L = flows[0].shape[0]; T = max(f.shape[1] for f in flows)
    padded = np.full((len(flows), L, T), np.nan, dtype=np.float32)
    for i, f in enumerate(flows):
        padded[i, :, :f.shape[1]] = f

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, flow=padded, ids=np.array(ids), axes=np.array(axes),
                        onsets=np.array(onsets), lengths=np.array(lengths), model=args.model)
    with args.out.with_suffix(".gen.jsonl").open("w") as f:
        for iid, ax, gt, on in zip(ids, axes, gen_texts, onsets):
            f.write(json.dumps({"instance_id": iid, "axis": ax,
                                "generation": gt, "clarify_onset": int(on)}, ensure_ascii=False) + "\n")
    print(f"Saved generation-flow [{len(flows)} x {L} layers x {T} tokens] -> {args.out}")


if __name__ == "__main__":
    main()
