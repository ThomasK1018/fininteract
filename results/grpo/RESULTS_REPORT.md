# FinInteract — GRPO/RLVR Ladder Results

**Date:** 2026-06-02 · **Base model:** `Qwen/Qwen3-4B-Instruct-2507` (4-bit QLoRA)
**Infra:** 8× A100-40GB. All sim/grader/axis judges + teacher run **locally** (vLLM, OpenAI-compatible) → **$0 OpenAI cost**.

## Setup notes / deviations from the kit
- README's default model id `Qwen/Qwen3-4B-Instruct` is a dead repo → used `Qwen/Qwen3-4B-Instruct-2507`.
- Pinned stack for driver 535 / CUDA 12.2: `torch 2.5.1+cu121`, `trl 0.19.1`, `transformers 4.51.3`.
- **Judges hosted locally:** `Qwen2.5-32B-Instruct-AWQ` on GPU7 (:8000) for sim/grader/axis.
- **Teacher hosted locally:** `Qwen2.5-72B-Instruct-AWQ` (TP=2) on GPU5,6 (:8001).
- **Key fix — axis-guided teacher:** an uninformed teacher (even gpt-5/72B) asks the *obvious* ambiguity
  (usually time period), never the *subtle* gold axis (entity scope, recognition policy, …) → 0% AxisHit.
  Solution: privately give the teacher the gold axis when generating SFT demos (hint NOT stored in the
  trajectory). Lifted strict on-axis SFT yield from ~0% → ~79%.
- GRPO script fixes: it ignored `--sft-adapter` (now starts from the KTO adapter); `num_generations` 6→4
  (must divide effective batch 16).

## Data (generated locally, $0)
- SFT: 372 on-axis+correct demos (axis-guided 72B teacher, strict filter, 117×4 episodes).
- KTO: 936 records, 783 desirable / 153 undesirable (guided + unguided merged).

## Ladder (held-out test.jsonl, n=51)

| Stage | Accuracy | Interaction | AxisHit@1 | Mean reward |
|-------|---------:|------------:|----------:|------------:|
| gpt-4o-mini (API baseline) | 5.9% | 100% | 3.9% | 0.085 |
| Qwen3-4B base (same-model) | 84.3% | 54.9% | 25.0% | 0.956 |
| **SFT** | 88.2% | 100% | 68.6% | 1.278 |
| **KTO** | 84.3% | 100% | **70.6%** | 1.267 |
| **GRPO** | **88.2%** | 100% | 66.7% | **1.299** |

(GRPO: 30 steps from the KTO adapter, G=4, ~95 min single-GPU; reward hovered ~1.3 with no clear climb —
within-group reward_std rose 0.14→~0.5 so there was gradient signal, but limited headroom.)

## Reading the results
- Accuracy is high even on the base model because the env's `search` returns the *disambiguated*
  evidence span, which contains the gold answer (a leak) — so accuracy has a low ceiling-to-baseline gap.
- The honest signal of learned behavior is **AxisHit@1** and **Interaction**, which can't be gamed by the
  leak: base **25%/55%** → SFT **68.6%/100%**. Training taught the model to *always interact and ask the
  right axis*.
- **SFT did the heavy lifting.** KTO and GRPO are roughly on par with SFT (differences are within
  run-to-run noise at n=51 with stochastic LLM judges). GRPO edged the best mean reward (1.299) by
  recovering accuracy to 88.2% while holding interaction at 100%; AxisHit stayed in the 67–71% band.
- This matches the expected story for an RLVR task whose SFT seed is already strong and whose terminal
  reward is partly leaked: the policy-improvement steps (KTO/GRPO) refine rather than transform.

## Artifacts
- Adapters: `outputs/{sft,kto,grpo}/` · Eval logs: `outputs/eval_{base_qwen,sft,kto,grpo}.log`
- Data: `data/full_guided/sft.jsonl` (372), `data/kto_merged.jsonl` (936)
- Parallel + axis-guided gen driver: `src/gen_trajectories_par.py` (`--axis-guided`, `--teacher-base-url`)
- Local servers: 32B-AWQ judges `:8000` (GPU7), 72B-AWQ teacher `:8001` (GPU5,6)

---

# Path B: True multi-turn GRPO via verl (proof-of-life attempt)

**Goal:** optimize the policy over the *entire* multi-turn episode with loss masked on the
simulator/search tokens (vs. the TRL single-turn approximation above).

**Integration written (complete, in `verl_integration/`):**
- `fininteract_agent_loop.py` — custom verl `AgentLoopBase` running the FinInteract ReAct
  episode: generate turn → parse JSON action → call local judges (sim/grader/axis) → repeat.
  Assistant tokens get `response_mask=1` (trained); env/simulator tokens get `0` (masked).
  The axis-aware reward is returned via `AgentLoopOutput.reward_score` (verl → `rm_scores`).
- `make_parquet.py`, `fininteract_agent.yaml` (agent registration), `run_grpo_sglang.sh`.
- `flash_attn/` — pure-torch shim for `bert_padding` (no flash-attn wheel for torch 2.9.1, no nvcc).

**Stack:** isolated `~/verl_sglang_venv` — verl 0.8.0, sglang 0.5.8, torch 2.9.1+cu128.
(vLLM backend was tried first but its async engine-core deadlocks inside Ray on this box.)

**Result — reached the optimizer step.** The multi-turn rollout RUNS end-to-end: SGLang
generates episodes through the custom FinInteract agent loop, the reward fires, log-probs
compute, and execution enters the Adam update. ~15 successive infra blockers were resolved
(datasets/pyarrow, cachetools, verl↔sglang kernel-name check, libcudart loader path, SGLang
mem-fraction, torchao↔peft conflict, GPU memory-balance check, flash-attn shim, GPU cleanup, ...).

**Remaining blocker (infra, not integration):** the Adam step OOMs on the 40 GB cards. verl's
actor FSDP **replicates** the full-parameter 4B optimizer per GPU (`data_parallel_size: 1`,
FULL_SHARD not engaging in hybrid+agent-loop mode), so ~32 GB of Adam state can't coexist with
the colocated SGLang rollout on one 40 GB A100. Paths to a completed step: get FSDP to shard
the optimizer, 8-bit/paged Adam, ≥4 GPUs for the actor, or LoRA (blocked here by the
sglang↔torchao version conflict). The FinInteract integration itself is complete and correct.

**PROOF OF LIFE — COMPLETED.** The 4B optimizer-replication OOM is purely a memory limit, so
we ran the same pipeline with **Qwen3-0.6B** (optimizer fits) and it trained end-to-end:

| step | pg_loss | grad_norm | reward mean (max/min) | num_turns (mean) |
|------|---------|-----------|------------------------|------------------|
| 1 | 0.0664 | 5.65 | 0.225 (1.0 / -0.3) | 5.5 (range 4–7) |
| 2 | 0.0309 | 6.30 | — | — |

This validates the full path-B loop: SGLang multi-turn rollout through the FinInteract agent
loop (4–7 turns/episode), axis-aware reward → GRPO advantages → actor optimizer step → weight
sync. Run: `verl_integration/run_grpo_sglang.sh` (model path = `Qwen/Qwen3-0.6B`).
For 4B, shard the optimizer (FSDP FULL_SHARD / ≥4 GPUs / 8-bit Adam) to fit the 40 GB cards.

---

# 4B verl multi-turn GRPO — full stable run (8-GPU box)

Ran the path-B pipeline on **Qwen3-4B-Instruct-2507** across **6 GPUs** (judge 32B-AWQ on a 7th):
FSDP shards the optimizer (~14 GB/card), SGLang rollout colocated (`free_cache_engine=False` to
avoid sleep/wake), 117-instance dataset kept within one epoch (the step-3 hang was an
**epoch-boundary dataloader deadlock** caused by an earlier 24-row proof-of-life parquet).

**15 steps, no hangs.** Learning signal (true multi-turn, n=8 GRPO group):

| step | reward mean | reward max | num_turns mean | grad_norm |
|------|-------------|------------|----------------|-----------|
| 1 | 0.55 | 1.55 | 5.7 | 4.0 |
| 7 | 1.08 | 1.70 | 6.1 | 6.5 |
| 13 | 1.11 | 1.70 | 8.6 | 6.9 |
| 15 | 1.08 | 1.70 | 7.3 | 5.4 |

Reward rose (0.55→~1.05) and **interaction increased (turns 5.7→8.6)** — GRPO is teaching the
4B to ask more clarifying questions (the axis-aware reward working). Advantages span ±2.5.
Run: `verl_integration/run_grpo_sglang.sh` (batch 6, n=8, 15 steps, free_cache_engine=False).
