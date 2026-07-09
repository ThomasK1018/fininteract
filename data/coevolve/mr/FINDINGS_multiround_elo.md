# Multi-round co-evolution + Elo arms race — findings

K=3 curriculum over four axes spanning the salience gradient (entity 0.92, metric 0.48,
temporal 0.15, recognition 0.04). Tool: `scripts/mr_coevolve.py`. Branch `coevolve/multiround`.

**Headline (honest, and it is not the predicted clean Elo curve): the Elo arms-race
instrument is _structurally degenerate_ on a recall-walled benchmark — 0 solver wins in every
tournament cell, for every axis. The valid abundance signal is the FRONTIER-YIELD trajectory,
which cleanly tracks salience. And multi-round co-evolution OVERFITS its own generated data:
zero transfer to the frozen human probe.** Two of the three results are cautionary.

## Method + two adaptations (both flagged, both matter)
1. **Recall-wall frontier.** The task defines frontier as "S_i fails *answer-only*", but
   FinInteract has a recall wall (answer-only ≈ 0 for every solver), so that filter never
   moves. Used the analogue: frontier = "S_i fails *answer+search+interact* resolution".
2. **Pool-partition generation.** Generation is solver-independent (`construct` doesn't see the
   solver), so per-round construction only wastes budget. Built a reusable instance **pool**
   per axis and partitioned it into fresh per-round slices (seeded). Pool *size* is the
   abundance ceiling: entity 119, metric 93, recognition 41, temporal 5.
   - *(A first 11h run was discarded: slicing passages in file order front-loaded the
     low-yield `docfinqa` source → battle sets of 1 and metric-yield 0. Diagnosed and rebuilt
     on curated pools; see `scripts/mr_coevolve.py`.)*
3. Solver training = axis-guided gated demos on the **cumulative** train set (train_0..train_i),
   SFT from base each round (accumulate-data-from-base; robust vs adapter-chaining).

## Result 1 — Frontier yield IS the abundance ceiling (the one clean positive)
Fresh frontier the solver still fails, per round (headline figure
`data/results/multiround_frontier_yield.png`, left panel):

| axis | salience | pool | r0 | r1 | r2 | trajectory |
|---|--:|--:|--:|--:|--:|:--|
| entity | 0.92 | 119 | 20 | 27 | 20 | **sustained** — corpus keeps supplying frontier |
| metric | 0.48 | 93 | 40 | 40 | 13 | sustained 2 rds, declines at pool exhaustion |
| temporal | 0.15 | 5 | 2 | 0 | 0 | **collapse @ r1** (generation barely yields) |
| recognition | 0.04 | 41 | 35 | 1 | 0 | **collapse @ r1** (pool exhausts after r0) |

The corpus's ability to source fresh frontier tracks salience/abundance. **The core
prediction holds at this level: abundant axes (entity, metric) sustain the loop; scarce axes
(recognition, temporal) collapse the moment the corpus runs dry.** The generator-side Elo
(despite being degenerate overall, below) *persists* for exactly as many rounds as frontier
lasts — collapses at r3 for entity/metric, r2 for recognition, r1 for temporal — an
independent read of the same ceiling.

## Result 2 — the Elo tournament is DEGENERATE (methodological finding, verified real)
Every `(generator g × solver s)` cell has **0 solver wins**: entity 0/52, metric 0/72,
recognition 0/32, temporal 0/4. So `coevolve_elo.py` returns "SOLVER RUNAWAY" for all four
axes — **these verdicts are artifacts of an all-zero win matrix, not signal. Do not use them.**

- **Not a bug** — verified: in `tourney_g2_s3` the best solver S3 asks (n_asks=1) and searches
  up to 7× but returns *"I am unable to find the specific operating income figure in the
  search results"*. It's the **recall/retrieval wall**, not a loading error (base resolves
  20/40 of a *fresh* slice, so wins are possible).
- **Root cause is circularity.** `battle_g` = held-out **frontier** = instances *defined by*
  the solver failing to resolve them → no solver resolves them → all-zero matrix. The task
  warned that an *answer-only* frontier "makes the generator win by construction"; the identical
  circularity applies to the *interactive-resolution* frontier. **A frontier-defined battle set
  cannot measure solver improvement — the Elo instrument can't get off the ground here.**

## Result 3 — no transfer: multi-round co-evolution OVERFITS generated data
The task asked: does multi-round keep generalizing, or overfit to generated data? **Answer:
overfit.** On the frozen human probe (held out, never trained), across all 3 rounds:

| axis (probe n) | resolve acc S1→S3 | AxisHit@1 S1→S3 | IR |
|---|:--:|:--:|:--:|
| entity (40) | 0.35 → 0.30 → 0.35 (**flat**) | 0.0 → 0.0 → 0.0 | 0.40 |
| recognition (9) | 0.11 → 0.11 → 0.11 (**flat**) | 0.0 → 0.0 → 0.0 | 0.67 |

The teacher demos are clean, on-axis (e.g. *"Do you mean the company's total consolidated
revenue including subsidiaries…?"*), yet the trained entity solver asks **off-axis** on the
human probe — it asks *temporal*-scope questions ("哪个时间段的净利润?") on *entity*-scope items.
The loop improves the model on its own generated distribution (well enough that the generated
tail stays hard, above) **without transferring axis-asking to human-authored ambiguity.**
Contrast the single-round entity result (AxisHit 25→68.6): cumulative multi-round SFT on the
reused mixed pool washed the gain out.

## Diversity (mode-collapse check)
Distinct entities per round: entity 28/29/29 and metric 24/26/12 stay **diverse** (no
mode-collapse — grounded generation resists it while the corpus lasts); recognition 29/1/0 and
temporal 4/0/0 collapse — but that is *scarcity forcing collapse*, not the generator degenerating.

## Verdict vs the stated predictions
- **entity HEALTHY / recognition STALL:** confirmed **only** on the frontier-yield axis
  (entity sustains, recognition collapses). **Not** confirmed on Elo — the Elo is degenerate
  for *all* axes, so it cannot show "healthy". The prediction's mechanism ("generator keeps
  producing frontier the improved solver still fails") is *half* true: the generator keeps
  producing frontier (for abundant axes), but the solver never improves enough to resolve *any*
  of it, so there is no two-sided race — it is corpus-abundance-limited on one side and
  recall-walled on the other.
- **Elo stalls where abundance runs out:** yes, via *generator-Elo persistence* (collapse round
  = frontier-exhaustion round) — but the level is dominated by the degenerate win matrix, so
  this is a weak, secondary read of the robust frontier-yield result.

## What would make the instrument work (for a follow-up)
1. **Decouple battle from frontier**: score solvers on a *fixed held-out set that includes
   resolvable instances* (not the failure-defined frontier) so improvement is measurable.
2. **Solver-conditioned generation**: grounded (corpus-bounded) construction cannot produce
   *adversarially harder* frontier as the solver improves — the "arms race" reduces to corpus
   abundance. A real race needs a generator that adapts difficulty, which passage-grounded
   generation structurally can't.
3. Break the recall wall (better retrieval) before resolution-based wins can be nonzero.

## Deliverables
`data/results/multiround_frontier_yield.png` (headline), `coevolve_elo_{entity,metric,
recognition,temporal}.{json,png}` (degenerate — kept for transparency),
`data/coevolve/mr/<axis>/{state.json,elo_matches.json,gen_r*,frontier_r*,train_r*,battle_r*,
tourney_*}`, adapters `outputs/coevolve/mr/<axis>/S{1,2,3}`, `scripts/mr_coevolve.py`,
`scripts/plot_multiround.py`. Held-out human probes never trained on.
