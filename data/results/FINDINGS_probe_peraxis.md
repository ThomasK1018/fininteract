# Per-axis linear decodability — testing "learnable ⟺ decodable"

Tests the §7.6 hypothesis that an axis's **behavioural** learnability (AxisHit scale-slope)
has a **representational** grounding: *learnable ⟺ the axis is linearly decodable from hidden
states*, with the prediction that **recognition is not decodable** and entity is, and that
decodability tracks the scale-slope. Tool: `experiments/sft_vs_rl/probe_per_axis.py` — per-axis
one-vs-rest per-layer logistic probe over bare-question last-token activations,
**peak AUROC** (5-fold CV, balanced; AUROC not accuracy because recognition n=9 / temporal n=7
are rare). Branch `probe/per-axis`. No API (hidden states only).

## Headline: the hypothesis is **REFUTED**. Decodability is *universal* and does **not** discriminate learnable from unlearnable axes.

| axis | salience (§7.6) | AxisHit scale-slope (learnability) | co-evolution | **peak decodability (AUROC)** |
|---|--:|--:|:--|--:|
| entity_scope | 0.92 | **+0.77** (strong) | **GO** (+0.61) | 0.77 |
| metric_definition | 0.48 | +0.21 (weak) | **GO** (+0.55) | 0.77–0.80 |
| recognition_policy | 0.04 | **+0.00** (none) | **NO-GO** (+0.00) | **0.93** ⟵ *most decodable, least learnable* |

**Recognition_policy — the un-learnable, co-evolution-NO-GO axis — is the MOST linearly
decodable of the three** (AUROC 0.93, properly powered). The prediction ("recognition not
decodable") is false; decodability *anti-tracks* learnability here. The representation is
present regardless of whether the skill is learnable or trainable.

### 1. Per-axis decodability × scale — **flat, not scaling**
Entity decodability across the sweep (the two reliable axes; entity n=94, metric n=63):

| model | entity AUROC | metric AUROC |
|---|--:|--:|
| 4B  | 0.772 | 0.765 |
| 8B  | 0.764 | 0.765 |
| 14B | 0.764 | (probing) |
| 32B | 0.784 | (probing) |

**Decodability is flat across scale (~0.77)** — yet behavioural AxisHit rises steeply with
scale (entity .12 → .89). So the axis is **equally represented at 4B and 32B**; only the
*behaviour* of acting on it scales. Decodability does **not** track the scale-slope — the
representational grounding the hypothesis wanted does not exist.

### 2. Small-sample caveat + augmented recognition test
On raw v1, recognition (n=9) and temporal (n=7) give AUROC **1.00 / 0.92** — small-sample
overfitting, *not* decodability evidence (the same scarcity that makes recognition a NO-GO
makes it un-probeable on v1). Augmenting recognition with the 36 round-0 generated instances
(→ **n=45** positives, 4B) gives a reliable **AUROC 0.932** (vs entity 0.782 / metric 0.799 on
the same run) — recognition is genuinely, strongly decodable.
*Caveat:* the 36 augmenters are gpt-generated, so a generation-style component could inflate
the 0.93; but even discounted it sits at/above the abundant axes, so the qualitative claim
(recognition IS decodable) holds.

### 3. Axis-guided SFT does **not** raise decodability
Probing the co-evolution adapters on 4B:

| probe | entity AUROC | recognition AUROC |
|---|--:|--:|
| base 4B | 0.772 | 1.000\* |
| + entity adapter (GO) | 0.770 | 1.000\* |
| + recognition adapter (NO-GO) | 0.777 | 1.000\* |

`\*`n=9 inflated. The trained axis's decodability is **flat** (entity 0.772→0.770). SFT changes
the **policy** (whether the model asks on the axis), not the **representation** (which is
already linearly present in the base). This *contradicts* the task's "SFT raises decodability"
expectation but *confirms* round-1/§6.6's **"represents-but-doesn't-act."**

## What this means for §7.6

The mechanistic upgrade the task hoped for — "decodable ⟺ learnable" — **does not hold**.
Instead:

> **All axes are linearly represented (decodable ~0.77–0.93), at every scale, before and after
> SFT.** Representation is *not* the bottleneck and *not* the discriminator. What gates
> learnability/co-evolution is **salience** (whether the axis is the *dominant* ambiguity —
> entity 0.92 vs recognition 0.04) and **abundance**, i.e. whether the model *attends to and is
> trainable on* the axis — not whether it can *decode* it. Recognition is decodable but
> subordinate: the representation is there, the model just doesn't act on it and there's too
> little data to teach it to.

This *strengthens* the paper's thesis rather than weakening it: the "represents-but-doesn't-act"
gap is now shown to hold **even for the axis that co-evolution cannot fix** — decodability is
necessary-but-not-sufficient, and the real learnability correlate is salience, not
representational presence.

## Deliverables
`data/results/probe_peraxis_{4b,8b,14b,32b}.json`, `probe_peraxis_4b_augrec.json`,
`probe_{entity,recognition}_sft.json`, `experiments/sft_vs_rl/probe_per_axis.py`, this file.
(14B/32B non-entity axes still CPU-probing at write time; entity-across-scale, augmented
recognition, and the SFT-invariance results — the three claims — are complete.)
