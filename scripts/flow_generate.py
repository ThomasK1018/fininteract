"""
Generation-flow capture (GPU): how the ambiguity signal evolves token-by-token.

For each instance we:
  1. load the per-layer "ambiguity direction" (diff-in-means of ambiguous vs disambiguated
     final-prompt-token activations) from a precomputed activations file (probe_activations.py),
  2. generate the model's response while capturing hidden states at each generated token,
  3. project each generated token's per-layer activation onto the ambiguity direction,
     yielding a [n_layers x n_generated_tokens] "flow" matrix per instance — the dynamic
     trace of how ambiguous the model's internal state is as it writes each word.

We also log the generated text + per-token strings, and flag the first clarification token
(?, "which", "specify", 请/哪/年份...) so the flow can be aligned to the moment the model
decides to ask. Projections are scalars, so storage stays tiny.

Requires a GPU. First run probe_activations.py to produce the activations file.

Usage:
  python src/flow_generate.py --model Qwen/Qwen3-4B-Instruct \
      --acts data/interp/acts_bare.npz --instances data/fininteract_v1.jsonl \
      --out data/interp/genflow.npz --max-new-tokens 64
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# tokens that signal the model is asking for clarification (EN + ZH)
CLARIFY_RE = re.compile(r"\?|which|specify|clarif|please provide|哪|请|具体|年份|期间", re.I)


def ambiguity_directions(acts_path: Path):
    """Per-layer unit diff-in-means direction + the centering midpoint, from acts file."""
    d = np.load(acts_path, allow_pickle=True)
    amb, dis = d["amb_last"], d["dis_last"]          # [N, L, H]
    mu_amb, mu_dis = amb.mean(0), dis.mean(0)        # [L, H]
    direction = mu_amb - mu_dis
    direction /= (np.linalg.norm(direction, axis=-1, keepdims=True) + 1e-8)
    midpoint = 0.5 * (mu_amb + mu_dis)               # center projections at the decision boundary
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
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    direction, midpoint = ambiguity_directions(args.acts)   # [L,H], [L,H]
    dir_t = torch.tensor(direction)
    mid_t = torch.tensor(midpoint)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    insts = [json.loads(l) for l in args.instances.open() if l.strip()]
    if args.limit:
        insts = insts[:args.limit]

    flows, ids, axes, gen_texts, onsets, lengths = [], [], [], [], [], []
    dvc = model.device
    dir_d, mid_d = dir_t.to(dvc).float(), mid_t.to(dvc).float()

    for k, inst in enumerate(insts):
        text = build_prompt(tok, inst["question"])
        ids_in = tok(text, return_tensors="pt").to(dvc)
        with torch.no_grad():
            out = model.generate(
                **ids_in, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tok.eos_token_id,
                output_hidden_states=True, return_dict_in_generate=True)
        gen_ids = out.sequences[0][ids_in["input_ids"].shape[1]:]
        gen_tokens = [tok.decode([t]) for t in gen_ids]
        gen_text = tok.decode(gen_ids, skip_special_tokens=True)

        # out.hidden_states: tuple over gen steps; each is tuple over layers of [1, seq, H]
        # take the LAST position (the newly generated token) per layer per step.
        flow = []   # [n_gen, n_layers]
        for step_hs in out.hidden_states:
            vecs = torch.stack([h[0, -1, :].float() for h in step_hs])   # [L, H]
            proj = ((vecs - mid_d) * dir_d).sum(-1)                       # [L]
            flow.append(proj.cpu().numpy())
        flow = np.asarray(flow, dtype=np.float32).T                      # [n_layers, n_gen]

        # clarification onset = first generated token index matching CLARIFY_RE
        onset = next((i for i, t in enumerate(gen_tokens) if CLARIFY_RE.search(t)), -1)

        flows.append(flow); ids.append(inst["instance_id"])
        axes.append((inst.get("axes") or ["?"])[0])
        gen_texts.append(gen_text); onsets.append(onset); lengths.append(flow.shape[1])
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(insts)}")

    # ragged flows -> pad to max length with nan for storage
    L = flows[0].shape[0]; T = max(f.shape[1] for f in flows)
    padded = np.full((len(flows), L, T), np.nan, dtype=np.float32)
    for i, f in enumerate(flows):
        padded[i, :, :f.shape[1]] = f

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, flow=padded, ids=np.array(ids), axes=np.array(axes),
                        onsets=np.array(onsets), lengths=np.array(lengths),
                        model=args.model)
    with args.out.with_suffix(".gen.jsonl").open("w") as f:
        for iid, ax, gt, on in zip(ids, axes, gen_texts, onsets):
            f.write(json.dumps({"instance_id": iid, "axis": ax,
                                "generation": gt, "clarify_onset": int(on)},
                               ensure_ascii=False) + "\n")
    print(f"Saved generation-flow [{len(flows)} x {L} layers x {T} tokens] -> {args.out}")


if __name__ == "__main__":
    main()
