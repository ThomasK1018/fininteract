# FinInteract-Evolve — Discriminating test on metric_definition (abundant, weakly scale-learnable)

Third co-evolution round, the one the §7.6 learnability framework made a **falsifiable
prediction** about. Base `Qwen/Qwen3-4B-Instruct-2507`; sim/grader/axis-judge OpenAI
`gpt-4o-mini`. Branch `coevolve/metric`. Same pipeline as entity/recognition; axis =
metric_definition.

## Prediction (stated before running) — and whether it held

> **Predicted: PARTIAL GO** — ΔAxisHit@1 on frozen human metric ~+0.2 to +0.4, clearly above
> recognition's +0.00 but well short of entity's +0.62 (because metric is abundant but only
> *weakly scale-learnable*: zero-shot AxisHit 4B .05 → 32B .26, Δ+0.21 vs entity's +0.77).

**The prediction did NOT hold.** metric is a **full GO** (ΔAxisHit +0.550 ≈ entity's +0.613),
**and** the *only* axis where co-evolution *raised* de-leaked accuracy (+45 pt).

## Result table

| set | metric | Round-0 (base) | Round-1 (co-evolved) | Δ |
|---|---|--:|--:|--:|
| frozen human metric (n=40) | **AxisHit@1** | 0.042 | **0.592** | **+0.550** |
| frozen human metric (n=40) | **de-leaked acc** | 15.0% | **60.0%** | **+45.0** |
| frozen human metric (n=40) | interaction rate | 50% | 100% | +50 |
| heldout_W generated (n=16) | AxisHit@1 | — | 0.875 | — |
| entity_scope (forgetting, n=30) | AxisHit@1 | 0.038 | 0.567 | +0.528 |
| recognition_policy (forgetting, n=9) | AxisHit@1 | 0.000 | 0.000 | +0.000 |

**Strict verdict: GO** (ΔAxisHit +0.550 ≥ +0.15; de-leaked acc up +45; no forgetting).
Accuracy transitions on the 40 frozen items: wrong→correct **18**, correct→correct 6,
wrong→wrong 16, **correct→wrong 0** — the round *fixed* 18 items and regressed none.

## The three-way result (what it actually shows)

| axis | source abundance | scale-learnability | **ΔAxisHit (frozen human)** | Δ de-leaked acc |
|---|---|---|--:|--:|
| recognition_policy | scarce (30 primary) | invariant (~flat) | **+0.000** | +0.0 |
| entity_scope | abundant (905) | high (Δ+0.77) | **+0.613** | **−10.0** |
| **metric_definition** | **abundant (498)** | **low (Δ+0.21)** | **+0.550** | **+45.0** |

**Two conclusions, both against the pre-registered framing:**

1. **Scale-learnability does NOT predict the co-evolution outcome.** metric is *weakly*
   scale-learnable (Δ+0.21) yet *strongly* training-learnable (+0.55) — as high as entity. The
   binding constraint is **source abundance / data availability**, not the scale slope: both
   abundant axes (entity, metric) install and generalize; the scarce axis (recognition, 36-
   instance ceiling) does not. Recognition's NO-GO is an **availability** failure, not (or not
   only) a "hard-to-learn" one. This *revises* the §7.6 story: the discriminator separated
   abundance from scale-learnability, and **abundance won**.

2. **Whether elicitation converts to accuracy is a per-axis property** — a clean new
   dissociation:
   - **metric: asking RESOLVES the ambiguity → +45 acc** (base can't answer without the
     metric definition; the clarifying question supplies it; 18 wrong→correct, 0 regressions).
     Demo quality reflects this: 157/240 teacher episodes were correct-and-on-axis (65%).
   - **entity: asking OVER-elicits → −10 acc** (base can often answer the entity directly;
     forcing a question converts 6 correct direct answers to wrong multi-turn ones).
   Both drove IR→100%; the accuracy sign flips on whether the base could already answer.

## Generation yield
metric abundant: 76 unique verified metric instances (all 76 frontier — base fails answer-only
on every one) → 60 train / 16 heldout → **157** axis-guided SFT demos (highest yield of the
three rounds; recognition 26, entity 48). SFT: grad_accum 1 / 6 epochs, train_loss 4.5→0.17.

## Bottom line for §7.5/§7.6
The co-evolution loop **closes on both abundant axes** (entity +0.61, metric +0.55) and fails
only on the **scarce** one (recognition +0.00). The pre-registered "abundance × scale-
learnability" two-factor model is **partly falsified**: scale-learnability did not gate the
outcome; **abundance did**. The genuinely new finding is the **accuracy dissociation** —
installing on-axis elicitation *helps* metric (+45) but *hurts* entity (−10), so a multi-round
curriculum needs a **when-to-ask gate** on axes the model can already answer directly.

## Artifacts (`data/coevolve/metric/`)
`gen_W.jsonl` (76), `train_W.jsonl` (60), `heldout_W.jsonl` (16), `demos/sft.jsonl` (157),
`base_probe.*`, `r1_{probe,heldout,forget}.*`, adapter `outputs/coevolve/sft_metric/`.
