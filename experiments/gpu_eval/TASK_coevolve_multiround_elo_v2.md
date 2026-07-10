# TASK: Multi-round co-evolution v2 -- non-degenerate AxisHit Elo + overfit guard

**Read `data/coevolve/mr/FINDINGS_multiround_elo.md` first.** v1 gave two negatives we now
fix: (a) the Elo tournament was DEGENERATE (0 solver wins everywhere) because win = interactive
RESOLUTION (recall-walled) on a battle set = the FRONTIER (circular); (b) multi-round training
OVERFIT (cumulative SFT washed out the single-round gain; human AxisHit -> 0). Script:
`scripts/mr_coevolve_v2.py` (built + syntax-checked; Elo tool `coevolve_elo.py` unchanged --
it is metric-agnostic).

## The two fixes (already implemented in the script)
1. **WIN = AxisHit@1, not resolution.** Leak-proof and NOT recall-walled, so outcomes vary and
   the Elo matrix is non-degenerate (verified in sim: an AxisHit tournament with varying
   targeting yields a clean HEALTHY-arms-race Elo curve).
2. **Battle = FIXED frozen val set + per-round MINER battles** (generator role: the instances
   the current solver targets worst on the fixed val set). Not the failure-defined frontier.
3. **Overfit guard = validation-based checkpoint selection.** Promote S_{i+1} only if its VAL
   AxisHit does not regress the best-so-far; else roll back. Capped-cumulative `--train-window`
   (default 1 = repeated single-round strength; raise to test whether accumulation helps once
   the guard is on). The frozen **human probe is TEST-only**, never used for selection.

## Step 0 -- CHEAP de-risk FIRST (no training): does the AxisHit game separate checkpoints?
Before retraining, confirm the instrument is non-degenerate on checkpoints you ALREADY have
(base + single-round `sft_entity`/`sft_metric` + multi-round `mr/*/S{1,2,3}`). Score each on the
frozen val + human probe by AxisHit@1 and build a small Elo tournament:
```bash
# serve each adapter, eval AxisHit on data/coevolve/mr_v2/<axis>/val_fixed.jsonl, then:
#   solver_wins = # instances with axis_hits[0].is_hit ; feed matrix to coevolve_elo.py
```
Expected: base ~0.06, single-round ~0.6+, multi-round-S3 ~0.0 (overfit) -> **non-degenerate**,
and single-round clearly out-Elos multi-round. If this separates, the instrument works; proceed.
(If every checkpoint scores ~0 AxisHit on val too, stop -- the val set itself is unsuitable.)

## Step 1 -- run v2 (the training you start)
```bash
export ...   # box env as usual (OPENAI_API_KEY etc.)
for AX in entity recognition; do   # add metric if budget allows
  python scripts/mr_coevolve_v2.py --axis $AX --cuda 0 --port 8010 --k 3 \
    --slice-size 30 --val-size 40 --battle-size 20 --train-window 1
done
# then, to test whether ACCUMULATION helps once the overfit guard is on:
python scripts/mr_coevolve_v2.py --axis entity --cuda 0 --port 8010 --k 3 --train-window 3
```
The script writes per-round `val_axishit`, `promoted`, `test_human_axishit` to `state.json`,
runs the AxisHit tournament, and produces `data/results/coevolve_elo_v2_<axis>.{json,png}`.

## Predictions (state before running)
- **AxisHit Elo is non-degenerate** (solver Elos spread; not all-zero as in v1).
- **entity: HEALTHY** arms race (solver targeting Elo climbs; miner keeps finding hard-to-target
  items while the val pool lasts). **recognition: STALL** (val pool tiny; solver can't improve
  targeting; both flat) -- the abundance-gating, now on a working instrument.
- **Overfit guard bites:** with `--train-window 1`, per-round TEST human AxisHit stays near the
  single-round level (~0.6 entity) instead of collapsing to 0 as in v1. Whether
  `--train-window 3` *beats* single-round on val/test is the open question the guard lets you ask.

## Deliverables -- `data/coevolve/mr_v2/FINDINGS_multiround_elo_v2.md`
Per axis: Elo json+png (with verdict), the per-round `val`/`test` AxisHit table (does the guard
keep human transfer?), and the Step-0 checkpoint-separation Elo. State whether the AxisHit game
is non-degenerate, whether entity=HEALTHY/recognition=STALL held, and whether guarded
accumulation (window 3) beats single-round (window 1). Commit to `coevolve/multiround-v2`, push.

## Guardrails
- `val_fixed.jsonl` and the human probe are held out -- never trained on. Never edit `data/final/*`.
- Tournament WIN = AxisHit@1 (leak-proof); resolution accuracy is recall-walled here and must NOT
  be the win condition. Sim/grader/axis-judge on OpenAI; policy local.
- Report honestly: if guarded accumulation still doesn't beat single-round, that is the finding
  ("for this setup, one targeted round is the sweet spot; multi-round adds no human transfer").
