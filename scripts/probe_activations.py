"""
Extract hidden-state activations from an open model for the FinInteract interpretability study.

For each instance we run TWO forward passes and capture per-layer residual-stream activations
at the final token (and mean-pooled):
  - AMBIGUOUS : the bare question (optionally under the Enumerate system prompt)
  - DISAMBIG  : the question followed by its disambiguating context (intended interpretation)

The within-item (ambiguous vs disambiguated) contrast is the basis for the ambiguity-detection
probe (analyze_probes.py) and is robust to axis class-imbalance. We also store the primary axis
label for the (entity-vs-metric) axis-decoding probe.

Requires a GPU (e.g. the project H100) + transformers/torch. Runs the model in eval mode with
output_hidden_states=True; no generation needed (single forward pass per prompt).

Usage:
  python scripts/probe_activations.py \
      --instances data/final/fininteract_v1.jsonl \
      --model Qwen/Qwen3-4B-Instruct \
      --prompt-mode bare \
      --out data/interp/acts_qwen_bare.npz
  # Enumerate-conditioned variant:
  python scripts/probe_activations.py ... --prompt-mode enumerate --out data/interp/acts_qwen_enum.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ENUMERATE_SYSTEM = (
    "You are a financial analyst. A question may be ambiguous: it can have different correct "
    "answers depending on scope, period, accounting basis, entity, or filing version. Before "
    "answering, enumerate every plausible interpretation of the question."
)


def build_prompt(tok, question: str, context: str | None, prompt_mode: str) -> str:
    """ambiguous = question alone; disambiguated = question + context."""
    user = question if context is None else f"{question}\n\nClarifying context: {context}"
    msgs = []
    if prompt_mode == "enumerate":
        msgs.append({"role": "system", "content": ENUMERATE_SYSTEM})
    msgs.append({"role": "user", "content": user})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def get_activations(model, tok, text: str):
    """Return (last_token, mean_pooled) per layer: arrays [n_layers, hidden]."""
    ids = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    out = model(**ids, output_hidden_states=True)
    hs = out.hidden_states            # tuple len n_layers+1, each [1, seq, hidden]
    mask = ids["attention_mask"][0].bool()
    last = np.stack([h[0, -1, :].float().cpu().numpy() for h in hs])           # [L, H]
    mean = np.stack([h[0][mask].float().mean(0).cpu().numpy() for h in hs])    # [L, H]
    return last, mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=Path, default=Path("data/final/fininteract_v1.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct")
    ap.add_argument("--prompt-mode", choices=["bare", "enumerate"], default="bare")
    ap.add_argument("--out", type=Path, default=Path("data/interp/acts.npz"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    insts = [json.loads(l) for l in args.instances.open() if l.strip()]
    if args.limit:
        insts = insts[:args.limit]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", output_hidden_states=True)
    model.eval()

    amb_last, amb_mean, dis_last, dis_mean = [], [], [], []
    axes, langs, ids = [], [], []
    for i, inst in enumerate(insts):
        q, c = inst["question"], inst.get("context", "")
        al, am = get_activations(model, tok, build_prompt(tok, q, None, args.prompt_mode))
        dl, dm = get_activations(model, tok, build_prompt(tok, q, c, args.prompt_mode))
        amb_last.append(al); amb_mean.append(am); dis_last.append(dl); dis_mean.append(dm)
        axes.append((inst.get("axes") or ["?"])[0])
        langs.append(inst.get("language", "en"))
        ids.append(inst.get("instance_id"))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(insts)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        amb_last=np.stack(amb_last), amb_mean=np.stack(amb_mean),
        dis_last=np.stack(dis_last), dis_mean=np.stack(dis_mean),
        axes=np.array(axes), langs=np.array(langs), ids=np.array(ids),
        model=args.model, prompt_mode=args.prompt_mode)
    print(f"Saved activations for {len(insts)} instances "
          f"({np.stack(amb_last).shape[1]} layers) -> {args.out}")


if __name__ == "__main__":
    main()
