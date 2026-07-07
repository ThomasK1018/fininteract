# What makes an axis "learnable" vs. not — a 5-axis characterization

Turns the entity(GO)/recognition(NO-GO) co-evolution pair into measurable,
predictive definitions, tested across all five axes.
Source: `scripts/analyze_learnability.py` over committed corpus + sweep data.

## Per-axis table (corpus = 1,847 passages)

| axis | primary | any-cand | salience ratio | human n | AxisHit 4B→32B | Δ | verdict |
|---|--:|--:|--:|--:|--:|--:|---|
| entity_scope | 905 | 980 | **0.92** | 94 | .12→.89 | **+0.77** | learnable+abundant → **GO** (observed) |
| metric_definition | 498 | 1035 | 0.48 | 108 | .05→.26 | +0.21 | abundant, *weakly* learnable → partial (untested) |
| temporal_scope | 150 | 998 | 0.15 | 7 | 1.00→1.00 | ~0 | already solved zero-shot → co-evo moot |
| filing_vintage | 264 | 794 | 0.33 | 0 | — | — | 0 human instances → untestable |
| recognition_policy | **30** | 796 | **0.04** | 9 | .00→.00 | **+0.00** | scarce + not-learnable → **NO-GO** (observed) |

*salience ratio = primary / any-candidate = "when this axis is present, how often is it the **dominant** ambiguity."*

## Measurable definitions that emerge

- **Learnable** ⟺ Targeting (AxisHit@1) **rises with scale**. It's a gradient, not
  binary: entity (Δ+0.77, strong) > metric (+0.21, weak) > recognition (+0.00, none);
  temporal is already saturated (skill present at every scale). Scale-slope is a
  cheap proxy for "the skill is representable and emerges."
- **Source-abundant** ⟺ high **primary-passage count** (entity 905 vs recognition 30).
  This is the data arm's raw supply of frontier instances.

## The key insight: both conditions share one root cause — **salience**

recognition_policy is present as a candidate axis in 796 passages but is *primary*
in only 30 (salience 0.04): it is almost always the **subordinate** ambiguity, sitting
behind a more salient entity/period axis in the same passage. That single fact
explains **both** failures:

1. **Scarcity** — rarely the dominant axis ⇒ few frontier instances to generate.
2. **Non-learnability** — when a salient competing axis co-occurs, the model's prior
   reaches for it; training can't overcome the pull, so it reverts to period/entity
   (exactly what the recognition round showed).

So **an axis is co-evolvable to the degree it is *salient* in the source.** Salient
axes (entity, salience 0.92) are both abundant and learnable; subordinate axes
(recognition, 0.04) are neither — the same property gates both conditions.

**Instructive exception (temporal):** salience 0.15 (rarely primary) yet AxisHit 1.0
(fully learnable). So corpus-salience and intrinsic-recognizability are *correlated but
distinct*: temporal is rare-as-primary but trivially recognizable when it is the
question ("which fiscal year?"). Learnability is best read from the **scale-slope**;
abundance from the **primary count**; salience-ratio explains why they usually move
together.

## Falsifiable prediction (the discriminating test)

**metric_definition** is the decisive case the framework has not yet run: **abundant**
(498 primary) but only **weakly learnable** (Δ+0.21, 32B AxisHit .26 vs entity .89). If
learnability is the binding constraint, a metric co-evolution round should be a
**partial GO** — clearly better than recognition (+0.00) but well short of entity
(+0.62). Running it would test whether abundance or learnability dominates.

## Deeper (mechanistic) definition of learnability — GPU follow-up
Scale-slope is a behavioural proxy. The representational test: a **per-axis linear
probe** on hidden states (reuse `experiments/sft_vs_rl/probe_sft_vs_base.py`) —
*learnable ⟺ the axis is linearly decodable*. Prediction: entity/metric decodable,
recognition-basis not (or only in the largest models), giving learnability a
mechanistic grounding rather than a behavioural one.
