"""
Step 1 — SFT (behavior cloning) on rejection-sampled teacher trajectories.
QLoRA on a single GPU. Consumes data/sft.jsonl produced by gen_trajectories.py.

Run:
  accelerate launch src/sft_train.py --model Qwen/Qwen3-4B-Instruct \
      --data data/sft.jsonl --output outputs/sft
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct")
    ap.add_argument("--data", type=Path, default=Path("data/sft.jsonl"))
    ap.add_argument("--output", default="outputs/sft")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="auto")
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

    ds = load_dataset("json", data_files=str(args.data), split="train")  # 'messages' col

    cfg = SFTConfig(output_dir=args.output, per_device_train_batch_size=2,
                    gradient_accumulation_steps=8, num_train_epochs=3,
                    learning_rate=2e-4, bf16=True, logging_steps=5,
                    max_length=2048, packing=False,
                    gradient_checkpointing=True,
                    gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=lora)
    trainer.train()
    trainer.save_model(args.output)
    print(f"Saved SFT adapter to {args.output}")


if __name__ == "__main__":
    main()
