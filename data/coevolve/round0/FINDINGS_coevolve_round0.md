# FinInteract-Evolve — Round-0 gated proof-of-concept: FINDINGS

Executes `experiments/gpu_eval/TASK_coevolve_round0.md` / `experiments/coevolve/DESIGN.md`.
One gated co-evolution round on the weakest axis **W = `recognition_policy`** (base
AxisHit@1 = 0, only 9 human items). Base policy: `Qwen/Qwen3-4B-Instruct-2507`, served
locally; simulator/grader/axis-judge = OpenAI `gpt-4o-mini`. Branch `coevolve/round0`.

## Result table

| set | metric | Round-0 (base) | Round-1 (co-evolved) | Δ |
|---|---|--:|--:|--:|
| frozen human recognition (n=9) | **AxisHit@1** | **0.000** | **0.000** | **+0.000** |
| frozen human recognition (n=9) | de-leaked acc | 11.1% | 11.1% | +0.0 |
| frozen human recognition (n=9) | interaction rate | 67% | 100% | +33 |
| heldout_W generated (n=8) | AxisHit@1 | — | **0.000** | — |
| entity_scope (forgetting, n=30) | AxisHit@1 | 0.091 | **0.793** | **+0.702** |
| metric_definition (forgetting, n=30) | AxisHit@1 | 0.045 | 0.345 | +0.299 |

**Decision rule:** GO iff ΔAxisHit@1 on the frozen human items ≥ +15 pt, de-leaked acc up,
other axes not degraded.

## >>> VERDICT: **NO-GO** <<<
ΔAxisHit@1 on the frozen human `recognition_policy` items = **+0.000** (bar +0.15).

## What actually happened (the mechanism — this is the finding)

The co-evolution round **did** work as a training procedure — the SFT converged (train_loss
4.49 → 0.19) and *transformed* behaviour: interaction rate 67 → **100%**, and axis-hitting on
`entity_scope` jumped **0.091 → 0.793** and `metric_definition` 0.045 → 0.345. So it installed
**general self-elicitation** and *improved* the other axes (no catastrophic forgetting — the
opposite).

But it **did not install the target axis.** On the frozen human items *and on held-out
generated instances from its own training distribution*, `recognition_policy` AxisHit stays
**0.000**. Inspecting the questions explains why:

- **Teacher (axis-guided) demos** ask genuine recognition/basis questions:
  *"Do you want operating income as presented … include items classified …"*,
  *"Should I include Meta's pre-tax restructuring charges (i.e. GAAP operating income) …"*
- **Co-evolved model** reverts to the **obvious** ambiguity — temporal/period:
  *"Do you mean revenue for the fiscal year ended January 31, 2023 rather than the prior
  fiscal year …"*, *"for the year ended December 31, 2023 …"*

**The model learned to ask, but defaulted to the salient axis (period/entity), not the subtle
recognition-basis distinction the teacher demonstrated — even on its own training
distribution.** This is the paper's central "models ask the obvious, not the gold axis"
finding reproduced at the *training* level: one round of axis-guided SFT on the achievable
recognition data is not enough to overcome the model's prior toward salient ambiguity.

## Generation yield (itself a result)

`recognition_policy` is **source-scarce**: only **30 / 1,847** passages have it as the primary
candidate axis (vs 905 for entity_scope) — precisely why v1 has only 9 human items.

| stage | count |
|---|--:|
| recognition-primary passages | 30 |
| gpt-5 Constructor accepted (from 30 primary, ~20 min/passage) | 5 |
| gpt-5-mini Constructor accepted (240 forced-primary secondary passages) | 31 |
| **unique recognition instances** | **36** |
| frontier (base fails answer-only) | 36 / 36 |
| train_W / heldout_W | 28 / 8 |
| axis-guided SFT demos (strict filter) | 26 |

The verifier rejection rate was ~0% (every accepted instance is genuinely hard — base fails
all 36 answer-only), so the *quality* gate is intact; the bottleneck is **availability**, not
verifiability.

## Honest deviations (round-0 engineering, flagged)
- **Constructor = gpt-5-mini**, not gpt-5. The spec'd gpt-5 Constructor ran at ~20 min/passage
  (5 instances in 4 h) — infeasible for a training set. gpt-5-mini is ~10× faster; the
  **rigorous gpt-5/gpt-5-mini adversarial Verifier (the quality gate) is unchanged**. This is a
  feasibility choice, not a quality shortcut (rejection rate stayed ~0 = all hard).
- **SFT schedule** for the tiny set: `grad_accum 8→1`, `epochs 3→20` (~260 steps). The default
  3-epoch/grad-accum-8 schedule gave only ~3 optimizer steps on 26 demos → a no-op adapter
  (byte-identical greedy outputs to base). The reported numbers use the properly-trained v2.

## Verdict in the DESIGN's terms & recommendation
This is the DESIGN's anticipated **NO-GO**, and a *strong* one: the gain did not even appear on
the self-generated `heldout_W` (AxisHit 0.000), so the round installs axis-*targeting* on the
easy axes but not on the target. Combined with the **availability bottleneck** (36 instances
max from v1's corpus), the honest read is:

> A single targeted round of grounded co-evolution installs general self-elicitation (and
> improves the salient axes) but **does not install the subtle target axis** and **does not
> generalize to held-out human items** — motivating (a) **multi-round** curriculum co-evolution
> and (b) **fresh-filing ingestion** (FY2024-25 EDGAR) to break the source-scarcity of rare
> axes, before a training round on `recognition_policy` can be expected to generalize.

Publishable as the §7.5 round-0 result: the engine runs end-to-end, the verdict is NO-GO, and
the *why* (salient-axis reversion + source scarcity) is the contribution.

## Artifacts (`data/coevolve/round0/`)
`gen_W.jsonl` (36), `train_W.jsonl` (28), `heldout_W.jsonl` (8), `demos/sft.jsonl` (26),
`base_probe.*`, `r1v2_{probe,heldout,forget}.*`, adapter `outputs/coevolve/sft_recognition_v2/`.
