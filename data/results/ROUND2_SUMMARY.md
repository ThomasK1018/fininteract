# FinInteract GPU Round 2 — Executive Summary

**Scope:** `experiments/gpu_eval/TASK_gpu_round2.md`, all five experiment groups (A, B, C1, C2, D)
completed on the 8× A100-40GB box. Frozen eval `data/final/fininteract_v1.jsonl` (N=173);
agents served locally, OpenAI (GPT-5 / GPT-4o-mini) as simulator/grader/axis judge.
Full detail + tables in [`FINDINGS_gpu_round2.md`](FINDINGS_gpu_round2.md).

---

## Headline

Round 1 established the *correlational* picture: the gold ambiguity axis is **linearly
decodable** from both A3B models' hidden states, yet **+interaction accuracy is 0% under a
~91% context-oracle ceiling** — "represents but doesn't act." Round 2 nails down *why*, and
the five experiments converge on one mechanism:

> **The FinInteract "elicitation gap" is a *recall wall* sitting behind an elicitation policy
> that is steerable (A) and trainable (C2) but is not fixed by reasoning (B) or by scale (C1) —
> and even perfect elicitation would not close the gap, because the binding constraint is
> recalling the figure, not asking the question (D).**

Every lever that *could* move elicitation was tested. Only one (targeted axis-guided SFT)
moves it — and it still doesn't move accuracy, because accuracy is recall-bound.

---

## The five results

### A — Causal steering  *(centerpiece: correlation → causation)*
Adding `α·(ambiguity direction)` to the residual stream **causally drives asking**:
- **Qwen3-30B-A3B:** +α raises first-action IR **61.7 → 75.9%** and AxisHit **0.21 → 0.59**; −α suppresses to 0.
- **Qwen3.5-35B-A3B:** needs a ~4× smaller α (more steer-sensitive); +α flips its search-first habit into asking, **IR 6 → 94%**.

The decodable axis is **not read-only** — it controls the elicitation policy. → `steer_curves.png`

### B — Reasoning ON vs OFF (same weights)
Thinking **fixes commitment but not accuracy**: 30B commit rate **30 → 66%** (Chinese ask-loops
halve), IR up — but **accuracy stays 0.0%** for both models. Reasoning buys asking/committing,
not the answer.

### C1 — Scaling sweep, Qwen3 0.6B → 32B
The context-oracle **ceiling rises monotonically 60.1 → 93.1%** with scale, while **+interact
stays pinned at ~0% at every size**. Scale buys *reading*, not *self-elicitation*. → `scaling_curve.png`

### C2 — GRPO ladder scale-up (axis-guided SFT at 8B/14B/32B)
The axis-guided teacher recipe is **scale-invariant**. SFT lifts on-axis asking from an erratic
base into a tight band at every size:

| size | AxisHit base → SFT | Interaction → | Accuracy → |
|------|--------------------|---------------|------------|
| 4B (ref) | 25 → 68.6% | 100% | 88% |
| 8B  | 30.6 → 64.7% | 100% | 82% |
| 14B | 16.3 → 72.5% | 100% | 82% |
| 32B | 25.0 → 72.5% | 100% | 71% |

Base models show **no scale trend** on the axis (14B base 16% < 8B base 31%) — the gain is the
recipe, not the parameters. The 8B full ladder confirms **KTO/GRPO refine within noise**
(AxisHit identical 64.7% across SFT/KTO/GRPO). → `c2_grpo_ladder.png`

### D — Elicitation vs. recall decomposition  *(the reframe)*
The new `interp-oracle` mode hands the agent the **resolved interpretation** but **no evidence
spans**:

| model | ceiling (read) | interp-oracle (recall) | +interact | elicitation cost | **recall cost** |
|-------|---------------|------------------------|-----------|------------------|-----------------|
| 30B | 90.2% | 0.0% | 0.0% | +0.0 | **+90.2** |
| 35B | 91.9% | 1.7% | 0.0% | +1.7 | **+90.2** |

The +interact 0% is **~100% recall cost**. Told exactly what to fetch but given no evidence,
both models still score 0. The headline shifts from "can't ask" to **"can't recall even when
told what to fetch."**

---

## Candidate §6 sentence

> Across two A3B generations the gold ambiguity axis is not only linearly decodable but
> **causally drives elicitation** (steering moves first-action IR 62→76% at 30B, 6→94% at 35B),
> and a 372-demo axis-guided SFT raises on-axis asking to a 65–73% band at every scale from 4B
> to 32B — yet +interaction accuracy stays 0% under a ~91% ceiling, because the gap decomposes
> into **~90 points of recall cost and ~0 of elicitation cost**, and neither chain-of-thought
> nor model scale touches it. The "elicitation gap" is mechanistically a **recall wall** behind
> a steerable, trainable-but-not-by-scale elicitation policy.

---

## Deliverables (this branch, `data/results/`)
- **A:** `steer_{qwen3-30b-a3b,qwen3p5-35b-a3b}.json`, `steer_curves.png`
- **B:** `eval_think_{on,off}_{qwen3-30b-a3b,qwen3p5-35b-a3b}.jsonl` (+summaries)
- **C1:** `eval_{open,ceiling}_qwen3-{0.6b,1.7b,4b,8b,14b,32b}.jsonl`, `scaling_curve.png`
- **C2:** `c2_grpo_ladder.{json,png}`, `c2_eval_{base,sft,kto,grpo}_qwen3-*.log`, `experiments/gpu_eval/c2_sft_train_gc.py`
- **D:** `eval_interp_oracle_{qwen3-30b-a3b,qwen3p5-35b-a3b}.jsonl` (+summaries)
- **Report:** `FINDINGS_gpu_round2.md`, this summary

## Reproducibility / infra notes
- vLLM cannot serve these models on this box (driver 535 / CUDA 12.2 vs cu130; `qwen3_5_moe`
  absent from vLLM's registry) → all agents run on a small custom HF OpenAI server
  (`experiments/gpu_eval/hf_openai_server.py`).
- `steer_axis.py` needed transformers-5 (BatchEncoding) + device-map-sharding + thinking-off
  fixes, plus a `search_rate` metric (the 35B searches before asking).
- C2 used a pinned training venv (torch 2.7.1+cu126, transformers 4.51.3, trl 0.19.1); the 32B
  QLoRA OOM was fixed with gradient checkpointing. Adapters omitted (large/regenerable).
- A/D are thinking-off; B toggles it. Steering over-saturation arms (30B ±8, 35B ≥+2) are
  off-scale and marked in the findings.
