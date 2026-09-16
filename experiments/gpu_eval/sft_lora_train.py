#!/usr/bin/env python3
"""Minimal category-guided SFT (LoRA, bf16) with plain transformers + peft, no TRL.

Trains on chat-format demos ({"messages": [...]}) with full-sequence loss, then merges
the adapter into the base weights and saves a servable bf16 checkpoint. Written so it
runs on one GPU with a recent transformers (5.x) where TRL/bitsandbytes pins are fragile.

Usage:
  python sft_lora_train.py --model ~/models/Qwen3-14B --data demos.jsonl \
      --output ~/models/Qwen3-14B-catsft [--epochs 3 --lr 2e-4 --r 16 --max-len 2048]
"""
import argparse, json, math, random, time
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model


class ChatDS(Dataset):
    def __init__(self, path, tok, max_len):
        self.items = []
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            msgs = json.loads(line)["messages"]
            try:
                ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False,
                                              enable_thinking=False)
            except TypeError:
                ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
            if hasattr(ids, "input_ids"):
                ids = ids["input_ids"]
            ids = list(ids)[:max_len]
            self.items.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ds = ChatDS(a.data, tok, a.max_len)
    print(f"[sft] {len(ds)} demos, mean tokens {sum(len(x) for x in ds.items)/len(ds):.0f}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    lora = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    loader = DataLoader(ds, batch_size=1, shuffle=True)
    steps = math.ceil(len(loader) * a.epochs / a.grad_accum)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, steps // 20), steps)
    model.train()
    t0 = time.time(); step = 0; acc_loss = 0.0; micro = 0
    for ep in range(a.epochs):
        for ids in loader:
            ids = ids.cuda()
            out = model(input_ids=ids, labels=ids)
            (out.loss / a.grad_accum).backward()
            acc_loss += out.loss.item(); micro += 1
            if micro % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 5 == 0 or step == steps:
                    print(f"[sft] epoch {ep+1} step {step}/{steps} loss {acc_loss/a.grad_accum/5 if step%5==0 else acc_loss/a.grad_accum:.4f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                    acc_loss = 0.0
    model = model.merge_and_unload()
    out_dir = Path(a.output); out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    print(f"[sft] merged checkpoint saved to {out_dir} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
