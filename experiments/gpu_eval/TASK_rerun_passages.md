# TASK: Re-run open-model behavioural evals WITH retrieval passages (bug fix)

**Why:** the round-1/round-2 open-model runs used `eval_model.sh` *without*
`--passage-file`, so `search` returned `context[:500]` (the disambiguator C, **no
answer values**) instead of the real retrieval passage. This silently zeroed every
open model's `+search`/`+interact` accuracy and default-capture (the smoking gun:
default-capture is exactly 0.0% for all open models vs. gpt-4o's 37%, which used
`passages.jsonl`). `eval_model.sh` is now fixed to pass
`data/sources/passages.jsonl` (covers all 173 instances). This task regenerates the
**accuracy** numbers that were invalid.

**What is NOT affected and must NOT be re-run** (already valid in the repo):
- Causal steering (A) — first-action behaviour, independent of search.
- Oracle ceiling (C1) and GRPO ladder AxisHit/interaction (C2) — span-injected /
  leak-proof, independent of this bug.
- The probe / representation results.
Only the open-model `+search` / `+interact` **accuracy + default-capture** are wrong.

## Preconditions
```bash
export OPENAI_API_KEY=sk-...          # simulator + grader
git pull                              # get the fixed eval_model.sh
```
The fixed `eval_model.sh` already passes `--passage-file data/sources/passages.jsonl`;
just run it as before. (If you call `evaluate.py` directly, add
`--passage-file data/sources/passages.jsonl`; it now prints a loud warning if a
search-mode run has no passages.)

## 1. A3B pair (fixes the two `tab:main` rows + §6.6)
```bash
for M in "qwen3-30b-a3b Qwen/Qwen3-30B-A3B" "qwen3p5-35b-a3b Qwen/Qwen3.5-35B-A3B"; do
  set -- $M; LABEL=$1; HF=$2
  ./experiments/gpu_eval/serve.sh "$LABEL" "$HF" 4 >/tmp/vllm_$LABEL.log 2>&1 &  # or hf_openai_server.py
  SRV=$!; until curl -sf http://localhost:8000/v1/models >/dev/null; do sleep 10; done
  ./experiments/gpu_eval/eval_model.sh "$LABEL"      # now passes --passage-file
  kill $SRV; wait $SRV 2>/dev/null; sleep 5
done
python experiments/gpu_eval/rollup_open.py qwen3-30b-a3b qwen3p5-35b-a3b
```

## 2. Scaling sweep (fixes the C1 accuracy line; ceiling already valid)
```bash
./experiments/gpu_eval/scaling_sweep.sh        # uses the fixed eval_model.sh
python experiments/gpu_eval/rollup_open.py     # all eval_open_*
```

## 3. Reasoning ON/OFF (fixes the accuracy column of group B)
Re-run the `--agent-thinking {on,off}` commands from `TASK_gpu_round2.md` §B — they
now include `--passage-file`. IR / AxisHit / commit-rate were already valid; only
accuracy changes.

## 4. Elicitation-vs-recall decomposition — REDO honestly (group D was invalid)
The earlier "recall wall" conclusion is **withdrawn**: it equated `+interact` (which
was 0% only because of the bug) with `interp-oracle`. Redo with three *now-valid*
conditions and report the decomposition only if the gap is real:
```bash
# +interact (now with passages) and interp-oracle (value-less BY DESIGN) and ceiling
./experiments/gpu_eval/eval_model.sh qwen3-30b-a3b         # gives valid +interact
python scripts/evaluate.py --instances data/final/fininteract_v1.jsonl \
  --models qwen3-30b-a3b --modes interp-oracle \
  --agent-base-url http://localhost:8000/v1 \
  --out data/results/eval_interp_oracle_qwen3-30b-a3b.jsonl \
  --summary data/results/eval_interp_oracle_qwen3-30b-a3b.summary.json
```
Decomposition (only valid once +interact has real retrieval):
`elicitation cost = interp-oracle − +interact`; `recall cost = ceiling − interp-oracle`.
State the conclusion the numbers actually support; do not assume recall dominates.

## Deliverables
Overwrite the confounded files in `data/results/` (eval_open_* + summaries for the
A3B pair and the sweep; eval_think_* ; eval_interp_oracle_*), and write
`FINDINGS_rerun_passages.md` with the corrected `tab:main` rows (from
`rollup_open.py`), the corrected scaling accuracy line, the reasoning ON/OFF accuracy,
and the *honest* elicitation-vs-recall decomposition. Commit + push.

## Guardrails
- Never edit `data/final/*.jsonl`.
- Simulator/grader stay on the OpenAI API.
- If `evaluate.py` prints the no-passage warning, STOP — the passage file is missing.
- Report whatever the corrected numbers show, including if `+interact` is still low.
