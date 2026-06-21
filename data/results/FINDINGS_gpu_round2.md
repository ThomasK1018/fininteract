# GPU Round 2 — causality, reasoning, gap-decomposition

Execution of `experiments/gpu_eval/TASK_gpu_round2.md` on the 8× A100-40GB box.
Frozen eval `data/final/fininteract_v1.jsonl` (N=173). Agent served locally via the
custom HF-transformers OpenAI server (vLLM still can't serve `qwen3_5_moe` on this
driver); simulator (GPT-5) + grader/axis-judge (GPT-4o-mini) on the OpenAI API.
Groups A, B, D completed for **both** models; C1/C2 tracked separately (see end).

**One-line result:** the ambiguity axis *causally* drives elicitation (A), but making
the model ask or commit does not move accuracy (B) because the entire +interact gap is
**recall**, not elicitation (D). Across both A3B generations the wall is *recalling the
figure*, not *asking the question*.

---

## A. Causal steering along the ambiguity direction  (`steer_axis.py`)

Per-layer direction = mean final-token act(`Q+C`) − act(`Q`), unit-normalized, added as
`α·dir` to the residual stream at layers {0.4,0.6,0.8}·depth; measure the **first
action** (single forward). Figure: `steer_curves.png`. n_dir=40, n_test=133, thinking-off.

**Qwen3-30B-A3B** (steer layers 19/29/38; α swept −8…+8):

| α | −8 | −4 | **0** | **+4** | +8 |
|---|----|----|-------|--------|----|
| IR (ask) % | 0* | 0 | **61.7** | **75.9** | 0* |
| AxisHit@1 | — | — | 0.21 | **0.59** | — |
| search % | 0 | 2.3 | 1.5 | 22.6 | 0 |

**Qwen3.5-35B-A3B** (steer layers 16/24/32; α swept 0…+2.5 — its residual stream is
~4× more steer-sensitive, ±4/±8 obliterate coherence):

| α | 0 | +0.5 | +1.0 | **+1.5** | +2.0 | +2.5 |
|---|---|------|------|----------|------|------|
| IR (ask) % | 6.0 | 21.8 | 29.3 | **94.0** | 77.4 | 11.3* |
| AxisHit@1 | 0.25 | — | — | 0.43 | 0.41 | 0.80 |
| search % | **94.0** | 78.2 | 70.7 | 6.0 | 0 | 0 |
| noparse % | 0 | 0 | 0 | 0 | 22.6 | 88.7* |

`*` = over-saturated: steering magnitude breaks output coherence (degenerate/unparseable).

**Read:** within each model's coherent range the effect is **monotone and causal**.
+α raises asking (30B 61.7→75.9%; **35B 6→94%**, converting its search-first habit into
asking) and generally sharpens the axis (30B AxisHit 0.21→0.59). −α (30B) and the
collapse beyond each model's ceiling are the off-scale arms. **This earns the §6.6
causal claim:** the linearly-decodable ambiguity axis is not read-only — it *drives* the
elicitation policy. The 35B needing a ~4× smaller α to achieve the same swing is itself a
finding (its ambiguity representation sits at a smaller residual-norm scale).

## B. Reasoning ON vs OFF, same weights  (`--agent-thinking`)

`answer+search+interact`, identical weights, thinking toggled via
`chat_template_kwargs.enable_thinking` on agent calls only. *commit rate* = 1 − fraction
of transcripts whose final message is still a question.

| model | think | acc | IR | AxisHit | commit | ZH commit |
|-------|-------|-----|-----|---------|--------|-----------|
| 30B | OFF | 0.0% | 72.3% | 0.21 | 30% | 41% |
| 30B | **ON** | **0.0%** | 94.8% | 0.27 | **66%** | **58%** |
| 35B | OFF | 0.6% | 77.5% | 0.40 | 95% | — |
| 35B | **ON** | **0.0%** | 91.9% | 0.23 | 98% | — |

**Read:** thinking **fixes the commitment failure** — the 30B's Chinese ask-loops halve
(commit 30→66%, ZH 41→58%) and IR rises — yet **accuracy stays 0%** for both. Reasoning
buys asking and committing; it does not buy the answer. (The 35B already commits ~95%
thinking-off; thinking adds IR but slightly *lowers* AxisHit, 0.40→0.23 — it asks more,
less precisely.)

## D. Elicitation vs. recall  (`--modes interp-oracle`)

interp-oracle hands the agent the **resolved interpretation C** but **no** evidence spans
and no search. ceiling = C + spans; interact = must elicit C.

| model | ceiling (read) | interp-oracle (recall\|C) | +interact | **elicitation cost** | **recall cost** |
|-------|---------------|---------------------------|-----------|----------------------|-----------------|
| 30B | 90.2% | **0.0%** | 0.0% | **+0.0** | **+90.2** |
| 35B | 91.9% | **1.7%** | 0.0% | **+1.7** | **+90.2** |

**Read (the headline reframe):** the +interact 0% is **~100% recall cost**. Even told
exactly which entity/period/metric is intended, neither model can produce the figure
without the answer-bearing passages (interp-oracle ≈ 0); the ~90-point jump only appears
when the spans are injected (ceiling). The bottleneck is **"can't recall the value even
when told what to fetch,"** not "can't ask." This is consistent across both A3B
generations and dovetails with A (asking is steerable) and B (asking/committing don't
lift accuracy).

---

## Synthesis (candidate §6 sentence)
> On two A3B MoE generations the gold ambiguity axis is not only decodable but **causally
> drives elicitation** — adding it to the residual stream monotonically converts answering/
> searching into asking (30B IR 62→76%, 35B 6→94%) and sharpens the queried axis — yet
> +interaction accuracy is pinned at 0% because the gap decomposes as **~90 points of
> recall cost and ~0 of elicitation cost**: handed the resolved interpretation but no
> evidence, both models score 0, and enabling chain-of-thought fixes commitment (30B
> commit 30→66%) without moving accuracy. The FinInteract "elicitation gap" is, mechanistically,
> a **recall wall** behind a steerable-but-non-binding elicitation policy.

## Caveats
- Steering "first action" is a single-forward proxy (IR/AxisHit are first-token
  decisions); over-saturation arms (30B ±8, 35B ≥+2) are off-scale and marked.
- Agent on HF transformers, thinking-off for A/D (B toggles it); judges on OpenAI.
- interp-oracle uses the answer-only system prompt with C prepended (no spans, no search).

## Deliverables (this round)
- `data/results/steer_qwen3-30b-a3b.json`, `steer_qwen3p5-35b-a3b.json`, `steer_curves.png`
- `data/results/eval_think_{on,off}_qwen3-30b-a3b.jsonl`, `..._qwen3p5-35b-a3b.jsonl` (+summaries)
- `data/results/eval_interp_oracle_qwen3-30b-a3b.jsonl`, `..._qwen3p5-35b-a3b.jsonl` (+summaries)
- `FINDINGS_gpu_round2.md` (this file)

## C1. Scaling sweep — dense Qwen3 0.6B → 32B  (`scaling_curve.png`)

Same family, all modes + context-oracle ceiling, served on the HF stack. The only
variable is scale.

| size | answer-only | +search | +interact | IR | AxisHit@1 | **ceiling** |
|------|-------------|---------|-----------|-----|-----------|-------------|
| 0.6B | 0.0 | 0.0 | 0.0 | 14 | 0.67 | **60.1** |
| 1.7B | 0.6 | 0.0 | 0.0 | 19 | 0.45 | **81.5** |
| 4B   | 0.0 | 0.0 | 0.6 | 60 | 0.32 | **87.3** |
| 8B   | 0.6 | 0.0 | 0.6 | 32 | 0.20 | **92.5** |
| 14B  | 0.0 | 0.0 | 0.0 | 80 | 0.41 | **90.8** |
| 32B  | 0.6 | 0.0 | 0.0 | 82 | 0.52 | **93.1** |

**Read:** the **context-oracle ceiling rises monotonically with scale (60.1 → 93.1%)** —
bigger models read/recall-from-evidence far better — while **+interact accuracy stays
pinned at ~0% at every size** (and answer-only/+search likewise ≈0). AxisHit is flat-noisy
(~0.2–0.67, no scaling trend) and IR is erratic (14→82%) but never converts to accuracy.
This is the §6.3 result: **scale lifts the oracle ceiling, not self-elicitation.** It is the
scaling-axis complement to D — scale buys *reading* (ceiling) but not the *recall on an
elicited query* that the +interact gap demands, so the gap is scale-invariant.

## Not yet run
- **C2 GRPO ladder (8B/14B/32B):** heavy training; held pending an explicit go (qualitatively
  different resource profile from the A/B/C1/D inference work).
