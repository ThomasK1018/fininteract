# The when-to-ask gate — fixing the entity over-elicitation regression

§7.5 found axis-guided SFT drives interaction to 100%, and the accuracy sign flips on whether
the base could already answer: asking **helps metric** (15→60, +45) but **hurts entity**
(32.5→22.5, −10 — it asks even on items it could answer, converting correct answers to wrong
multi-turn ones). This task installs a **when-to-ask gate** and tests whether it recovers the
entity regression while keeping the metric gain. Frozen human probes (n=40 each); de-leaked
accuracy (`--passage-file`); branch `coevolve/when-to-ask`.

## Success criterion (stated before running)
Gated policy: entity de-leaked acc **≥ base (kill the −10)** AND metric acc **≈ +45 over base**,
ask-rate dropping from 100% toward each axis's real need.

## Design correction (a finding in itself): the recall wall breaks the literal recipe
The task's gate signal is "does the base answer it **answer-only**." But FinInteract has a
**recall wall** — base answer-only = **0/39** correct (nothing is answerable without retrieval).
So that check yields *zero* no-ask demos and the gate collapses to always-ask. The correct
"can already answer" signal is **base answers correctly WITH search but WITHOUT asking
(n_asks=0)**. Under that signal the gated demo mix is asymmetric exactly as expected:
**entity 19% no-ask** (11 no-ask + 48 ask — base resolves many entity items directly) vs
**metric 4% no-ask** (7 + 157 — metric genuinely needs the definition).

## Results (frozen human probe, n=40)

| axis | policy | de-leaked acc | ask-rate | AxisHit@1 |
|---|---|--:|--:|--:|
| entity | base (natural gate) | 32.5 | 40% | 0.06 |
| entity | always-ask (co-evolved) | 22.5 | 100% | 0.68 |
| entity | **gated-A (train-time)** | **7.5** | 15% | 0.67 |
| entity | **gated-B (inference, base-uncertainty)** | **30.0** | 40% | — |
| metric | base | 15.0 | 50% | 0.04 |
| metric | always-ask | 60.0 | 100% | 0.59 |
| metric | **gated-A (train-time)** | **2.5** | 5% | 0.00 |
| metric | **gated-B (inference)** | **35.0** | 50% | — |

## Verdict

**Variant A (train-time gate): FAILS.** Mixing no-ask demos into the SFT — even 7 of 164 for
metric — **catastrophically destabilizes the ask policy**. The gated model collapses to
answering *directly* (75% of first actions are `answer`, skipping even retrieval — a behaviour
*no* demo teaches), which is fatal on the recall-wall task: entity 32.5→**7.5**, metric
60→**2.5**. Verified not an artifact: the always-ask adapter reproduces IR 100%/acc 50% under
the identical serving setup, while the gated adapter serves at IR 5%. SFT over ReAct behaviours
is bimodal (always-ask vs always-answer) and the no-ask demos tip it into the wrong basin.
**This is the failure mode the task flagged ("under-ask on metric costs the +45") — realized in
the extreme.**

**Variant B (inference gate — ask only where the base is naturally uncertain): recovers entity,
partially keeps metric.** Routing by the base model's own ask decision (base asked → use the
co-evolved asker's answer; base didn't ask → use the base's direct answer):
- **entity: 22.5 → 30.0** — the −10 over-elicitation regression is **recovered** to ≈ base
  (32.5), at ask-rate 40% (vs always-ask 100%). **✓ criterion met for entity.**
- **metric: 35.0** — keeps **+20** of the +45 (vs base 15), but not the full gain, because the
  base's natural ask-rate on metric is only 50% where metric needs ~100%. **~ partial.**

## Interpretation & upgrade for §7.5

The when-to-ask gate moves from "future work" to **demonstrated (inference-time)**: a
zero-training gate that asks only on the base's uncertain items **recovers the entity
over-elicitation regression** (30.0 ≈ base, vs always-ask's 22.5) while still lifting metric
(+20). The gate is real; its *calibration* is the open lever:
- The naive **train-time** gate is the wrong tool — it destabilizes the learned policy (bimodal
  collapse), a caution for demo-mixing recipes.
- The **inference** gate works but the base's uncertainty signal *under-asks on metric* (50% vs
  the ~100% metric needs). A per-axis or confidence-calibrated gate (ask when the base's
  no-ask answer would be *wrong*, not merely when the base is *unsure*) should recover entity
  AND keep the full +45 — the concrete next step.

**Net:** over-elicitation is fixable at inference (entity recovered); keeping the full metric
gain needs a better-calibrated gate, not a train-time one.

## Artifacts (`data/coevolve/`)
`when_to_ask_{entity,metric}_{nonprobe,baseeval}.jsonl`, `{entity,metric}/demos/sft_gated.jsonl`,
`{entity,metric}/r1_gated_probe.jsonl`, adapters `outputs/coevolve/sft_{entity,metric}_gated/`,
`build_gated_demos_v2.py`.
