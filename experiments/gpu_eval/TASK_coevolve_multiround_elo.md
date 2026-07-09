# TASK: Multi-round co-evolution + Elo arms-race tracking

**Read `data/results/FINDINGS_learnability.md` + the three round-0 FINDINGS first.** So far
each axis is a SINGLE round. This runs a **K-round curriculum** and instruments it with an
**Elo arms race** between the generator (Constructor) and the solver (trained agent), using
the committed, tested tool `scripts/coevolve_elo.py`. The goal is a robustness result: show
the loop's health is gated by **abundance**, quantified by a standard measure.

## Prediction (state before running)
- **entity (abundant):** HEALTHY arms race — solver Elo AND generator Elo both climb over
  rounds (the generator keeps producing frontier the improved solver still fails).
- **recognition (scarce):** STALL — the generator hits its ~36-instance ceiling, cannot
  out-pace the solver; both Elos flatten. Elo stalling *exactly* where abundance runs out
  triangulates Finding 13 from an independent instrument.

Run **entity** and **recognition**; add **metric** if budget allows.

## The curriculum (per axis, K=4)
`S_0` = base Qwen3-4B. For round i = 0..K-1:
1. **Generate against the current solver.** Constructor produces candidates; frontier_i =
   verified instances that **S_i fails answer-only** (the frontier moves as the solver
   improves). Split frontier_i into `train_i` / `battle_i` (80/20). Keep `battle_i` for the
   tournament (never trained on).
   ```bash
   python scripts/construct_fast.py --source data/sources/passages.jsonl \
     --out data/coevolve/mr/<axis>/gen_r$i.jsonl --target 300
   # filter to axis, then keep S_i-answer-only-wrong -> frontier_i ; split train_i/battle_i
   ```
2. **Train the next solver.** Gated demos (reuse `gen_axis_guided_demos.py`) on `train_i`;
   SFT S_i -> S_{i+1} (`outputs/coevolve/mr/<axis>/S{i+1}`). Continue from S_i (cumulative)
   OR from base each round — **log which**; cumulative is the intended arms race.
3. Log per round: frontier yield (the abundance ceiling shows up here for recognition), and
   **held-out human AxisHit** on the frozen probe (does multi-round keep generalizing or
   overfit to generated data?).

## Diversity rider (cheap, do it inline)
For each round's generated set, record instance **diversity** — mean pairwise embedding
distance (any local encoder) + count of distinct entities/sub-topics. Collapsing diversity
across rounds = the mode-collapse failure that grounded co-evolution is supposed to avoid;
report it either way.

## The Elo tournament (the headline)
Play every generator checkpoint against every solver checkpoint on the held-out battle sets,
**with interaction** (the solver wins iff it RESOLVES — answer+search+interact graded correct;
do NOT use answer-only, or the frontier filter makes the generator win by construction):
```bash
for i in 0..K; do for j in 0..K; do
  python scripts/evaluate.py --instances data/coevolve/mr/<axis>/battle_r$i.jsonl \
    --models S$j --modes answer+search+interact --passage-file data/sources/passages.jsonl \
    --agent-base-url http://localhost:8000/v1 \
    --out data/coevolve/mr/<axis>/tourney_g${i}_s${j}.jsonl --summary /dev/null
done; done
```
Assemble the win matrix and fit Elo:
```bash
python3 - <<'PY'
import json, glob, re
matches=[]
for f in glob.glob("data/coevolve/mr/<axis>/tourney_g*_s*.jsonl"):
    i,j=map(int,re.search(r"_g(\d+)_s(\d+)",f).groups())
    rows=[json.loads(l) for l in open(f)]
    w=sum(1 for r in rows if r.get("correct")); n=len(rows)   # solver 'wins' = resolved
    matches.append({"gen_round":i,"solver_round":j,"solver_wins":w,"n":n})
json.dump({"n_rounds":4,"axis":"<axis>","matches":matches},
          open("data/coevolve/mr/<axis>/elo_matches.json","w"))
PY
python scripts/coevolve_elo.py --matches data/coevolve/mr/<axis>/elo_matches.json \
  --out data/results/coevolve_elo_<axis>.json --fig data/results/coevolve_elo_<axis>.png
```
`coevolve_elo.py` is already tested (`--synthetic` reproduces a healthy race and a stall);
it outputs solver/generator Elo trajectories, the per-round gap, and a HEALTHY / STALL /
SOLVER-RUNAWAY verdict.

## Deliverables — `data/coevolve/mr/FINDINGS_multiround_elo.md`
Per axis: (1) `coevolve_elo_<axis>.{json,png}` with the verdict; (2) per-round frontier yield;
(3) per-round **held-out human AxisHit** (generalization trajectory); (4) diversity trajectory.
State whether the prediction held (entity HEALTHY, recognition STALL) and whether Elo-stall
coincides with the abundance ceiling. Commit to branch `coevolve/multiround`, push.

## Budget note
Biggest co-evolution experiment: K generations + K SFTs + a $(K{+}1)^2$ tournament (K=4 ->
25 eval cells x ~40 battle instances x interaction per axis). Log any cell you cap or skip;
do not silently truncate the tournament (a missing cell biases the Elo fit).

## Guardrails
- `battle_i` and the frozen human probe are held-out — never trained on. Never edit `data/final/*`.
- Tournament games are INTERACTIVE resolution, `--passage-file` required. Sim/grader on OpenAI.
- Report STALL honestly if recognition (or even entity) fails to sustain the race.
