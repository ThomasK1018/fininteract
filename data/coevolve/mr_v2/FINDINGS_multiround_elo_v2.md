# Multi-round co-evolution v2 — non-degenerate AxisHit Elo + overfit guard

Fixes the two v1 negatives (degenerate resolution-Elo; apparent multi-round overfit). Script
`scripts/mr_coevolve_v2.py`; Step-0 de-risk `scripts/mrv2_step0_derisk.py`; figure
`data/results/multiround_v2_axishit.png` + `coevolve_elo_v2_*.png`. Branch `coevolve/multiround-v2`.

**Headline: the AxisHit game is NON-degenerate and the abundance-gating holds on a working
instrument — entity LEARNS targeting and TRANSFERS to humans (val 0.05→0.78, human 0.05→0.92;
solver Elo 1000→1696), recognition STALLS (flat ≈0.06, human 0). And v1's scary
"multi-round overfits → human AxisHit=0" DOES NOT REPLICATE: it was an eval artifact.**

## Two infrastructure bugs found & fixed this session (both would have faked a null result)
1. **Server never loaded the adapter.** The committed `hf_openai_server.py` had no
   `HF_ADAPTER`/PEFT code (my v1 LoRA-load edit was uncommitted and `git reset --hard` dropped
   it). Every checkpoint served *base* → all AxisHit identical ≈0.05 — which masquerades exactly
   as the brief's "instrument unsuitable, stop" condition. Patched (committed now so it can't be
   lost again); verified `loaded LoRA adapter` in every server log and behaviour change 0.05→0.79.
2. **Miner battles were eval-rows, not instances.** `battle_g` was written from evaluate.py
   *output* rows (no `question` field) → the tournament re-eval errored `KeyError:'question'`
   on every cell → 0/20 everywhere (a *different* all-zero than v1). Fixed to compute the win
   matrix by **lookup** from each solver's per-instance val AxisHit (`val_S{g}_forminer`) — free
   and correct — instead of re-serving. (Also fixed the Elo output path colliding across
   `--train-window`/`--val-size` variants.)

## Step 0 — de-risk (no training): the AxisHit game SEPARATES existing checkpoints ✓
Scored on the frozen entity val (40) + human probe, with adapters actually loaded:

| checkpoint | VAL AxisHit@1 | TEST human AxisHit@1 |
|---|--:|--:|
| base | 0.05 | 0.00 |
| single-round `sft_entity` | 0.55 | 0.675 |
| multi-round `mr/entity/S1` | 0.65 | **0.875** |
| multi-round `mr/entity/S3` | 0.675 | **0.80** |

Non-degenerate, monotone. **This alone refutes v1's overfit story: v1's own final adapter
(`mr/entity/S3`) scores human AxisHit 0.80 when the adapter is actually loaded** — v1's reported
"→0" was the adapter-load bug in the eval, not overfitting.

## Main runs (K=3, slice 30, val 40 frozen, overfit guard on)
`val` = frozen selection set; `test` = frozen human probe (report-only, never selects).

| run | val r0/r1/r2 | test(human) r0/r1/r2 | best val | solver Elo | verdict |
|---|:--:|:--:|:--:|:--:|:--|
| **entity w1** (single-round) | 0.68/**0.78**/0.73 | 0.63/**0.93**/0.93 | 0.775 (S2) | 1000→1506→**1551** | SOLVER-RUNAWAY |
| **entity w3** (accumulate) | 0.63/**0.80**/0.73 | 0.68/**0.88**/0.88 | 0.800 (S2) | 1000→1629→**1696** | SOLVER-RUNAWAY |
| **recognition** val40 | 0.05/0.05/0.05 | 0/0/0 | 0.05 | 1000 (flat) | **STALL** |
| **recognition** val15 | 0.067/0.067/0.067 | 0/0/0 | 0.067 | 1000 (flat) | **STALL** |

Entity win matrix (w1), `solver_wins` of S_s on S_g's hardest-20 (miner):
```
        s0  s1  s2  s3
   g0    0  14  15  15      base loses everywhere; trained win 14-15/20
   g1    1   7  11  11      S1's hard set dents S1 (7) but S2/S3 recover (11)
   g2    2  11  11  11
   g3    2  11  11  11
```

### Answering the brief's questions
1. **Non-degenerate?** YES. Entity solver Elo spreads 1000→~1550–1696 (v1 was all-zero). The
   instrument works once win = AxisHit@1 and the matrix is computed correctly.
2. **entity HEALTHY / recognition STALL?** Confirmed in direction. Entity's solver *climbs
   steeply and transfers* (human 0.93); recognition is *dead flat* at base level even with a fair
   26-instance training set (val15) — it genuinely cannot learn to target. The tool labels
   entity **SOLVER-RUNAWAY** rather than "HEALTHY arms race": the solver learns targeting faster
   than the corpus-bounded miner can raise difficulty (generator Elo rises 1388→1546 then
   plateaus). So: a real, one-sided race on entity; a flat stall on recognition — the
   abundance/salience gating, now on a non-degenerate instrument.
3. **Does guarded window-3 accumulation beat single-round?** **No clear win — they're
   comparable.** w3 is marginally better on val/Elo (best val 0.80 vs 0.775; solver Elo 1696 vs
   1551) but marginally worse on the human probe (0.88 vs 0.93). Within noise. **One targeted
   round is roughly the sweet spot; accumulation neither helps nor hurts once the guard is on.**
4. **Does the overfit guard keep human transfer?** YES — and it *bit*: in BOTH entity runs the
   round-2 candidate regressed on val (0.78→0.73 / 0.80→0.73) and was **rolled back** to S2,
   preserving human AxisHit at 0.88–0.93 instead of drifting. This is the mechanism v1 lacked.

## The v1 correction (important for §7.6)
v1 concluded multi-round co-evolution **overfits** (cumulative SFT washes out the gain; human
AxisHit→0). v2 shows that was an **adapter-load/eval artifact**: with adapters verifiably loaded,
both single-round and guarded multi-round entity solvers transfer to the human probe at
**0.88–0.93**, and v1's own S3 checkpoint scores 0.80. The corrected §7.6 claim: **axis-guided
SFT on an abundant/salient axis (entity) is learnable, decodable-and-actable, and transfers to
held-out human ambiguity; a scarce axis (recognition) stalls regardless of the training regime.
Multi-round adds little over one well-targeted round — the win is the targeting objective + a
validation guard, not the number of rounds.**

## Guardrails / honesty
- `val_fixed` and the human probe were held out; the probe never entered selection.
- recognition val40 is degenerate (pool 41 → train 1); the val15 run (train 26) is the fair test
  and it *still* stalls, so the STALL is a learnability result, not a data-starvation artifact.
- Elo "SOLVER-RUNAWAY" for entity is the honest tool verdict (solver outpaces a corpus-bounded
  miner); it is *not* a clean two-sided "HEALTHY" race — grounded generation can't manufacture
  adversarially harder targeting items, same structural limit noted in v1.

## Follow-up runs (metric axis + seed robustness + full v1 retraction)

### Peak is at 2 rounds, not 3 — the guard rejects round 3 (correction)
In every entity run the val AxisHit peaks at **S2 (round 1)** and round 2's S3 does **not** improve:
seed1234 w1 0.675/**0.775**/0.725, w3 0.625/**0.800**/0.725 (S3 rolled back both); seed7 w3
0.775/**0.900**/0.875 (S3 ≤ S2). So the claim is **"two guarded rounds peak; the third overfits
and the guard rejects/does-not-exceed it,"** NOT monotone "more rounds better." Robust across seeds.

### Metric (salience 0.48) — accumulation is REQUIRED (unlike entity)
| metric run | val r0/r1/r2 | human r0/r1/r2 | solver Elo | verdict |
|---|:--:|:--:|:--:|:--|
| w1 (single 30-slice) | 0.0/0.0/0.0 | 0/0/0 | flat 1000 | **STALL** |
| w3 (accumulate) | 0.0/**0.975**/0.0 | 0.025/**0.75**/0.75 | 1000→1000→**3066** | SOLVER-RUNAWAY |

A single 30-instance slice teaches metric targeting **not at all** (w1 dead flat); **two
accumulated slices (~53) reach val 0.975 / human 0.75**; the third overfits (S3 0.0, guard rolls
back to S2). So for metric, **guarded accumulation clearly beats single-round (0.75 vs 0.0)** —
the opposite of entity, where w1≈w3. Reproduced independently by v1's own metric adapters
(S1 0.0 / **S2 0.75** / S3 0.05 below). The salience→data-need gradient: **entity (0.92) learns
from one slice; metric (0.48) needs to accumulate two; recognition (0.04) doesn't reliably learn.**

### Full v1 retraction table (human AxisHit@1, adapters re-evaluated with the fixed server)
`scripts/mrv2_v1_retraction_eval.py`; probes held out, n(entity)=n(metric)=40, n(recognition)=9.

| axis | v1 *reported* | S1 corrected | S2 corrected | S3 corrected |
|---|:--:|:--:|:--:|:--:|
| entity | 0 / 0 / 0 | **0.875** | **0.900** | **0.775** |
| metric | (no v1 probe) | 0.0 | **0.75** | 0.05 |
| recognition | 0 / 0 / 0 | **0.444** | 0.0 | 0.0 |

**entity's v1 "0/0/0" was entirely the adapter-less-server artifact** — the same checkpoints
transfer to humans at 0.78–0.90. **Honest nuance:** recognition S1 corrected to **0.444 (4/9)** —
so recognition is *not* a perfectly clean stall; on the tiny 9-instance probe it is noisy and
inconsistent (S2/S3 fall back to 0, and the v2 val15 run reads 0.067). Report recognition as
**weakly/unreliably learnable on a scarce, noisy axis**, not "cannot learn at all."

## Deliverables
`scripts/mr_coevolve_v2.py` (fixed miner + Elo path), `scripts/mrv2_step0_derisk.py`,
`scripts/plot_multiround_v2.py`, `experiments/gpu_eval/hf_openai_server.py` (adapter load),
`data/results/multiround_v2_axishit.png`, `coevolve_elo_v2_{entity,entity_w3,recognition,
recognition_v15}.{json,png}`, per-run `data/coevolve/mr_v2/<tag>/{state.json,elo_matches.json,
val_fixed.jsonl,val_S*,test_probe_S*,forminer}`, Step-0 `mr_v2/entity/step0_*`. Adapters in
`outputs/coevolve/mr_v2/` (regenerable from demos).
