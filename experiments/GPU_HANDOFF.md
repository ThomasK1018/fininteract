# GPU Handoff — Reviewer-Response Experiments (8×A100)

For a colleague running the GPU-bound reviewer experiments (Exp 5, 6, 9) that cannot run on
the laptop. The API-only experiments (Exp 1, 2, 3b, 4b, 8, 9-lexical) are already done — see
`experiments/REVIEWER_RESPONSE_PLAN.md` for the full plan and results. This doc is only the
GPU work.

## Environment
- **Hardware:** 8×A100-SXM4-40GB (NVLink), per `experiments/gpu_eval/RUNBOOK.md`.
- **Stack (pinned, from `results/grpo/RESULTS_REPORT.md`):** `torch 2.5.1+cu121`,
  `transformers 4.51.3`, `trl 0.19.1`, plus `peft`, `bitsandbytes`, `vllm`, `accelerate`,
  `scikit-learn`, `numpy`. Trained backbone: **Qwen3-4B-Instruct-2507**. Local judges served
  via vLLM: Qwen2.5-32B-Instruct-AWQ (sim/grader), teacher Qwen2.5-72B-Instruct-AWQ.
- **Serving:** `experiments/gpu_eval/serve.sh` + `hf_openai_server.py` (loads a LoRA adapter,
  exposes an OpenAI-compatible endpoint for `scripts/evaluate.py`).
- **Judges via API (optional):** if you route sim/grader through OpenAI/OpenRouter instead of
  local vLLM, note `configs/openrouter.json` is **gitignored** — bring your own key. For a
  fully local run you do not need it.
- **Data (all committed):** `data/final/fininteract_v1.jsonl` (benchmark),
  `data/sources/passages.jsonl` (passages), `data/coevolve/mr/_pools/*.jsonl` (per-axis
  frontier pools), `data/coevolve/{entity,metric}/` (demos, probes).

---

## Exp 5 — Matched sample sizes + multiple seeds (salience / co-evolution)
**Goal:** downsample entity/metric training to the same 9/26-item sizes as recognition;
recognition learning curves at 30/60/90; ≥3–5 seeds; report mean±sd.

Machinery: `scripts/mr_coevolve_v2.py` (slice → `gen_axis_guided_demos.py` → SFT → AxisHit
eval → validation-gated promotion), analysis `scripts/analyze_learnability.py`, plots
`scripts/plot_multiround_v2.py`. Knobs: `--slice-size` (default 30), `--train-window`
(1/2/3 → 30/60/90), `--val-size`, `--seed` (default 1234).

Seed sweep example (repeat per axis and per size; loop seeds):
```
for SEED in 1234 7 21 99 2025; do
  python scripts/mr_coevolve_v2.py --axis entity --cuda 0 --port 8000 \
      --k 3 --slice-size 30 --train-window 1 --seed $SEED
done
```
**Build still needed:** an explicit equal-N subsampler — current code slices by pool order,
not a target-N sampler, so cutting *exactly* 9/26 across axes needs a small pre-cut of the
`_pools/*.jsonl` (pools differ: entity 119, metric 93, recognition 41, temporal 5). The SFT
trainer `experiments/gpu_eval/c2_sft_train_gc.py` has **no seed arg** — add one for
weight-init/data-order control. Recognition (n≈9 human probe, 41 generated) stays noisy;
either mint more via `scripts/pull_recognition_policy.py` or frame recognition
scale-invariance as a hypothesis.

---

## Exp 6 — SFT/RL baselines + non-leaky eval
**Goal:** compare axis-guided SFT vs generic-clarification SFT, random-axis SFT, always-ask
SFT, axis-aware prompt (no train), and SFT-without-KTO/GRPO; company/filing-disjoint test,
ordinary retrieval, mean±sd over seeds.

Present: SFT trainer `experiments/gpu_eval/c2_sft_train_gc.py` (TRL + QLoRA), axis-guided
demo generator `scripts/gen_axis_guided_demos.py`, non-leaky retrieval
`experiments/sft_vs_rl/search_no_leak.py`, guided-vs-unguided spec
`experiments/sft_vs_rl/README.md`.

SFT baseline demo variants to build (each is a teacher-prompt tweak of
`gen_axis_guided_demos.py`):
- **generic-clarification SFT** — teacher gets no axis hint (asks the obvious question).
- **random-axis SFT** — teacher told a deliberately wrong axis.
- **always-ask SFT** — force a clarification regardless of axis.
Then SFT each with `c2_sft_train_gc.py` and evaluate with `scripts/evaluate.py` using
`search_no_leak.py` (ordinary retrieval, not the gold span) on a company/filing-disjoint
split (the `company`/`source` fields are on every row).

> **BLOCKER — KTO/GRPO trainers are NOT in this repo.** They lived in an external kit
> (`fininteract_grpo_kit/`, `verl_integration/fininteract_agent_loop.py`) referenced in
> `experiments/sft_vs_rl/README.md` and `results/grpo/RESULTS_REPORT.md`. The **SFT arm is
> fully reproducible here**; the SFT-without-KTO/GRPO ablation and any KTO/GRPO-benefit claim
> require re-obtaining that kit (TRL `KTOTrainer`; verl multi-turn `GRPOTrainer`) or
> reimplementing it. Until then, report SFT results only and soften RL claims.

---

## Exp 9 — Probe controls + causal steering (activations)
**Goal:** company/source/template-disjoint probe splits; shuffled-label & bag-of-words &
length-matched controls; cross-language transfer; and causal steering.

The **lexical/CPU controls are already done** on the laptop (`scripts/probe_bow_control.py`:
BoW/length/company-disjoint axis decoding). What needs GPU is activation extraction + the
neural probe + steering:

1. **Extract activations** (per model in the paper's set):
```
python scripts/probe_activations.py --instances data/final/fininteract_v1.jsonl \
    --model Qwen/Qwen3-4B-Instruct --prompt-mode bare --out data/interp/acts_qwen3-4b.npz
```
2. **Neural probe + behavioral mismatch:** `scripts/analyze_probes.py` (detection probe,
   axis decode, `behavioral_mismatch`). **Add** shuffled-label null for the *linear* probe and
   company/source/template-disjoint + cross-language (EN/ZH) splits — the `.npz` stores
   `axes`/`langs`/`ids`; company is joinable from `fininteract_v1.jsonl`. Compare the neural
   probe against the already-measured BoW floor (0.73–0.80 AUROC) — the neural probe must beat it.
3. **Causal steering (the key upgrade):** `experiments/gpu_eval/steer_axis.py` is already
   written — run it at scale to turn "represented but not acted on" from correlational to causal:
```
python experiments/gpu_eval/steer_axis.py --model Qwen/Qwen3-30B-A3B \
    --instances data/final/fininteract_v1.jsonl \
    --layers 0.4 0.6 0.8 --alphas -8 -4 0 4 8 --out data/results/steer_qwen3-30b-a3b.json
```
Show that adding/subtracting the decoded ambiguity direction changes IR / AxisHit@1.

---

## Suggested order
1. Exp 9 steering (code done — fastest causal win) + activation extraction for the disjoint-split controls.
2. Exp 5 seed sweeps (add the equal-N subsampler + SFT seed arg first).
3. Exp 6 SFT baselines; resolve the KTO/GRPO external-code blocker before any RL claim.

Report mean±sd across seeds for everything. Questions → see `REVIEWER_RESPONSE_PLAN.md`.
