# FINDINGS — Re-run WITH retrieval passages (bug-fix correction)

Executes `experiments/gpu_eval/TASK_rerun_passages.md`. The round-1/2 open-model evals
ran `evaluate.py` **without `--passage-file`**, so the `search` action returned
`context[:500]` (the disambiguator C, **no answer values**) instead of the real filing
passage. This silently zeroed every open model's `+search`/`+interact` **accuracy** and
default-capture. Re-ran with `data/sources/passages.jsonl` (passage_id-keyed,
**173/173 coverage**, answer present in 168/173). Branch `interp/rerun-passages`.

**Validation of the bug (4B, n=12):** `+search` acc 0→16.7%, default-capture 0→58% — the
old uniform 0% was the missing-passage artifact.

**What was NOT affected (unchanged, still valid):** causal steering (A), the probe /
representation results, the context-oracle **ceiling**, and C2's AxisHit / interaction
(span-injected / leak-proof / search-independent). Only the open-model `+search`/`+interact`
**accuracy** and the group-D decomposition were wrong.

---

## Group 1 — A3B pair, corrected `tab:main`

| model | Ans-only | +Search | +Interact | IR | AxisHit@1 | Ceiling |
|-------|---------:|--------:|----------:|----|----------:|--------:|
| qwen3-30b-a3b | 1.2 | **28.3** | **11.6** | 72 | .22 | 90.2 |
| qwen3p5-35b-a3b | 0.6 | **35.8** | **28.9** | 36 | .39 | 91.9 |

(was `+Search`/`+Interact` = 0.0 / 0.0 and 2.3 / 0.0). LaTeX:
```
qwen3-30b-a3b   & 1.2 & 28.3 & 11.6 & 72 & .22 & -- \\
qwen3p5-35b-a3b & 0.6 & 35.8 & 28.9 & 36 & .39 & -- \\
```

## Group 2 — scaling sweep, corrected accuracy line (ceiling unchanged)

| size | Ans-only | +Search | +Interact | Ceiling |
|------|---------:|--------:|----------:|--------:|
| 0.6B | 0.0 | 25.4 | 15.6 | 60.1 |
| 1.7B | 1.2 | 6.9 | 9.8 | 81.5 |
| 4B | 0.0 | 8.1 | 12.1 | 87.3 |
| 8B | 0.6 | 14.5 | 24.9 | 92.5 |
| 14B | 0.0 | 18.5 | 22.5 | 90.8 |
| 32B | 1.2 | 27.2 | 8.4* | 93.1 |

`*32B +interact at n=155 (the eval hung on interact episode 157 and was killed; answer-only
and +search are complete at n=173).` `+Search`/`+Interact` were ~0 across all sizes before
the fix. Now both are nonzero; `+search` rises roughly with scale (4B 8% → 32B 27%)
alongside the ceiling, but stays far below it.

## Group 3 — reasoning ON/OFF (with passages)

| model | think | acc | IR | AxisHit | commit |
|-------|-------|----:|----|--------:|-------:|
| 30B | OFF | 12.1 | 72 | .23 | 73 |
| 30B | ON | 11.0 | 94 | .26 | 86 |
| 35B | OFF | 28.9 | 36 | .38 | 98 |
| 35B | ON | 35.3 | 72 | .33 | 97 |

Thinking is ~neutral on accuracy for the 30B (12.1→11.0, within n=51… n=173 noise) and
**helps the 35B (28.9→35.3)**; it raises IR for both. (Earlier partial-n estimates of a
large 30B drop / 35B 43% were small-sample artifacts — these full-173 numbers supersede them.)

## Group 4 — elicitation vs. recall, redone HONESTLY

The earlier "recall wall" (the +interact 0% ⇒ ~90 pt recall cost) is **WITHDRAWN** — it
equated a *bugged* 0% +interact with interp-oracle. With all conditions now valid:

| model | ceiling (read) | interp-oracle (no evidence) | +search | +interact |
|-------|---------------:|----------------------------:|--------:|----------:|
| 30B | 90.2 | 0.0 | 28.3 | 11.6 |
| 35B | 91.9 | 2.3 | 35.8 | 28.9 |

- **recall-from-memory cost** (ceiling − interp-oracle) ≈ **90 / 90** — *this part holds*:
  given only the resolved interpretation but no evidence, the models score ~0; they
  genuinely cannot produce specific filing figures from parametric memory and need evidence.
- **retrieval recovers** (+search − interp-oracle) = **+28.3 / +33.5** — the models' own
  search recovers a substantial chunk (the *withdrawn* part: it was never 0).
- **interaction HURTS vs. plain search** (+interact − +search) = **−16.8 / −6.9** — asking
  clarifying questions makes accuracy *worse* than just searching.

**Honest verdict.** The bottleneck is **evidence retrieval/grounding, not parametric
recall, and certainly not a "recall wall" at 0%.** Models recover 28–36% by retrieving;
they cannot answer without evidence (interp-oracle ≈ 0); and interaction is *net-negative*
relative to plain search. The paper's "self-elicitation doesn't convert" thesis **survives
and is in fact stronger** (interaction actively hurts), but the round-2 framing of a total
recall wall was an artifact of the missing-passage bug and is retracted.

## Deliverables (overwritten / added, branch `interp/rerun-passages`)
`eval_open_*` (+summaries) for the A3B pair and the 0.6B–32B sweep; `eval_think_{on,off}_*`;
`eval_interp_oracle_*`; this file. Steering/probe/ceiling/C2 unchanged.
