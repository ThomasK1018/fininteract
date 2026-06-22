# Round-2 Extensions — deepening A and D, and closing the C2 loop

Follow-ups run after the main round-2 task, building on the recall-wall result. Same eval
set and judges. Detail tables: this file; figures `steer_localization.png`, raw
`d_recall_breakdown.txt`, `steer_loc_*.json`.

---

## E1 — Steering layer-localization (extends A): *where* does the axis drive behavior?

Experiment A steered a mid-late band {0.4,0.6,0.8}·depth together. Here we steer **one band
at a time** — early (~0.2–0.3), mid (~0.5–0.6), late (~0.8–0.9) — at each model's working α,
to localize the causal locus. (n_dir=40, limit=90, first-action IR/AxisHit.)

| model (α) | band | layers | ΔIR (steer−base) | AxisHit@1 | note |
|-----------|------|--------|------------------|-----------|------|
| 30B-A3B (4) | early | 10/14 | **−20.0** | 0.33 | steering early *suppresses* asking |
| 30B-A3B (4) | **mid** | 24/29 | **+8.9** | **0.82** | max asking + axis precision |
| 30B-A3B (4) | late | 38/43 | +2.2 | 0.13 | little effect |
| 35B-A3B (1.5) | early | 8/12 | breaks (100% noparse) | — | early steering destroys coherence |
| 35B-A3B (1.5) | **mid** | 20/24 | **+67.8** | 0.56 | flips search→ask |
| 35B-A3B (1.5) | late | 32/36 | +1.1 | 0.33 | no effect (stays searching) |

**Finding — the causal lever is mid-network for *both* models, and it is *not* where the
probe peaks.** Round 1 found the axis most linearly decodable **early** in Qwen3-30B-A3B
(L6/47, ~13% depth) and **late** in Qwen3.5-35B-A3B (L37/39, ~95% depth). But the layer where
*adding* the axis direction actually changes the first action is **mid (~50–60% depth) for
both** (30B mid ΔIR +8.9 / AxisHit 0.82; 35B mid ΔIR +67.8). Early steering suppresses or
breaks; late steering does nothing.

> **Decodability locus ≠ causal locus.** The axis can be *read out* early/late, but it is
> *converted into the elicitation policy* in the middle of the network. This sharpens the
> represents-vs-acts distinction into a depth statement: the "acting" happens mid-stack,
> regardless of where the "representing" is strongest. → `steer_localization.png`

## E2 — Recall-wall universality (extends D)

Decomposition of the interp-oracle gap (ceiling − interp-oracle = recall cost) split by
language and by primary axis:

```
                       ceiling   interp-oracle   recall-cost
30B-A3B  EN (n=53)       69.8%       0.0%           69.8
30B-A3B  ZH (n=120)      99.2%       0.0%           99.2
35B-A3B  EN (n=53)       73.6%       1.9%           71.7
35B-A3B  ZH (n=120)     100.0%       1.7%           98.3
  per axis (both models): interp-oracle ≈ 0% for entity_scope / metric_definition /
  recognition_policy / temporal_scope; ceiling ranges 57–97%.
```

**Finding — the recall wall is total, not localized.** interp-oracle accuracy is ~0% in
**every** language and **every** axis; only the ceiling (which injects evidence spans) varies.
So "can't recall the figure once the interpretation is known" is not a quirk of one axis or
language — it is a uniform absence of parametric recall of specific filing figures. (Note the
ceiling is much higher in ZH, 99–100%, than EN, 70–74% — the models *read* Chinese evidence
nearly perfectly, yet still recall nothing without it, so the recall wall is even starker in
ZH, ~98–99 pts.) → `d_recall_breakdown.txt`

## E3 — C2 on the dissected star MoE (Qwen3-30B-A3B)  *(eval running)*

C1/C2 used dense Qwen3 as scale proxies; here the axis-guided SFT recipe is applied to the
**exact MoE model A/B/D dissected** (Qwen3-30B-A3B). SFT trained (QLoRA 4-bit + gradient
checkpointing, 69 steps); base-vs-SFT eval on the held-out test set is running.

> *Result pending — to be filled with base→SFT AxisHit@1 / interaction / accuracy. Expectation
> from the dense ladder (4B/8B/14B/32B all → 65–72.5% AxisHit): the recipe should lift the same
> model whose representation we localized in E1, closing the loop representation→behaviour→fix.*
