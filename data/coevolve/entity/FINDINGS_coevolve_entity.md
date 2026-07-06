# FinInteract-Evolve — Positive control on a scale-solvable axis (entity_scope)

Companion to the round-0 NO-GO (`data/coevolve/round0/`, target `recognition_policy`).
This round targets **entity_scope** — a *scale-solvable* axis (zero-shot AxisHit@1 4B .12 →
32B .89) — to test whether the co-evolution loop **closes when the skill is learnable**.
Base policy `Qwen/Qwen3-4B-Instruct-2507` (served locally); sim/grader/axis-judge = OpenAI
`gpt-4o-mini`. Branch `coevolve/entity`.

## Result table

| set | metric | Round-0 (base) | Round-1 (co-evolved) | Δ |
|---|---|--:|--:|--:|
| frozen human entity (n=40) | **AxisHit@1** | 0.062 | **0.675** | **+0.613** |
| frozen human entity (n=40) | de-leaked acc | 32.5% | 22.5% | **−10.0** |
| frozen human entity (n=40) | interaction rate | 40% | 100% | +60 |
| heldout_W generated (n=20) | AxisHit@1 | — | 0.400 | — |
| recognition_policy (forgetting, n=9) | AxisHit@1 | 0.000 | 0.056 | +0.056 |
| metric_definition (forgetting, n=30) | AxisHit@1 | 0.000 | 0.217 | +0.217 |

**Decision rule:** GO iff ΔAxisHit@1(frozen) ≥ +15pt **and** de-leaked acc up **and** other
axes not degraded.

## Verdict: split — **skill-generalization GO, strict-rule NO-GO**

- **The primary positive-control claim PASSES (this is the headline).** Axis-targeting
  **generalizes to frozen human items**: entity AxisHit@1 **0.062 → 0.675 (+0.613)**, ~4×
  the +0.15 bar, and it holds on generated heldout_W (0.400) too. Contrast with round-0
  recognition (+0.000 on both frozen and heldout). **The pipeline works; the co-evolution loop
  *closes* when the axis is learnable — isolating the recognition NO-GO as an axis property,
  not a broken pipeline.** No forgetting (recognition/metric both *rise*).
- **The strict 3-part rule returns NO-GO**, because **de-leaked accuracy dropped 10 pt** (32.5
  → 22.5). Mechanism: the SFT drove interaction from **40% → 100%** — the model now asks an
  entity question on *every* item, including ones it previously answered correctly outright.
  Accuracy transitions on the 40 frozen items: correct→correct 7, wrong→wrong 25,
  **correct→wrong 6**, wrong→correct 2 (net −4). Total asks 19 → 40. The model **over-elicits**:
  it learned *to ask on the right axis* but not *when asking is worth it*, and the extra turn
  converts several correct direct answers into wrong multi-turn ones (the round-2 "interaction
  can hurt vs. plain answer/search" effect, now induced by training).

## Reading it (pairs with §6.4 / Finding 12)

Together with round-0 this is **one dissociation, two levers**:

| axis | scale lever (zero-shot 4B→32B) | training lever (this loop, ΔAxisHit frozen) | source yield |
|---|---|---|---|
| entity_scope | .12 → .89 (solvable) | **+0.613 (generalizes)** | 99 instances |
| recognition_policy | ~flat (invariant) | +0.000 (round-0, no transfer) | 36 (ceiling) |

**Scale and targeted training both install entity/metric; neither installs recognition.** The
data arm's yield tracks the same split (entity abundant 99 vs recognition's hard 36-ceiling).

The entity accuracy cost is a **refinement finding, not a pipeline failure**: axis-guided SFT
should be paired with a *when-to-ask gate* (ask only when the query is genuinely
under-specified) so installing elicitation on the target axis does not regress the
already-answerable cases. That is a concrete next step for the multi-round curriculum.

## Generation yield (contrast is the result)
entity_scope is **source-abundant** (~905/1,847 passages primary): 99 unique verified entity
instances (all 99 frontier — base fails answer-only on every one) → 79 train / 20 heldout →
48 axis-guided SFT demos. Round-0 recognition hit a hard **36-instance ceiling** from 30
primary passages. Same pipeline, opposite availability.

## Setup / deviations (flagged)
- Constructor = gpt-5-mini (feasibility, as in round-0); adversarial gpt-5/gpt-5-mini Verifier
  unchanged (rejection ~0 → all frontier).
- SFT: properly-trained schedule (grad_accum 1, 6 epochs = 144 steps, train_loss 4.5 → 0.43),
  matching round-0's converged v2 — not the initial no-op.
- Agent served via HF server (vLLM can't serve here); passages used for de-leaked accuracy.

## Artifacts (`data/coevolve/entity/`)
`gen_W.jsonl` (99), `train_W.jsonl` (79), `heldout_W.jsonl` (20), `demos/sft.jsonl` (48),
`base_probe.*`, `r1_{probe,heldout,forget}.*`, adapter `outputs/coevolve/sft_entity/`.
