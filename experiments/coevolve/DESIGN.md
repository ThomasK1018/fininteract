# FinInteract-Evolve — a self-refreshing, contamination-resistant benchmark loop

**One-line thesis.** FinInteract already contains *both arms* of a data↔model
co-evolution loop — a grounded **data-generation engine** (the three-role
Constructor→Verifier pipeline, §4.2) and a **model-improvement engine** (the
axis-guided SFT/GRPO ladder, §7.4). Closing the loop turns the benchmark from a
frozen snapshot into a *living* benchmark that regenerates hard instances on the
axes where the current best model still fails, resisting saturation and
contamination.

This is the **"living-benchmark engine"** scope (chosen 2026-07-02): FinInteract
stays a Resource paper (benchmark + taxonomy + findings); co-evolution is the
mechanism that makes it *not saturate*, demonstrated with **one gated
proof-of-concept round** — not a full method paper.

## Why this is safe here (and not in ungrounded self-play)

Multi-Agent Evolve (arXiv 2510.23595, Proposer/Solver/Judge) and similar
self-play loops risk **reward hacking / quality collapse** because the Proposer
and Judge are ungrounded — nothing outside the model pins down truth. FinInteract
is different: **every instance is anchored to a real filing passage** under
"Easy to verify, Ambiguous to resolve," and the adversarial Verifier is a
hard filter (reject if ≥2 models answer without context). The ground truth lives
in EDGAR/cninfo, not in the model. That grounding is exactly what lets the data
arm evolve without drifting into nonsense — and is a genuine differentiator to
state over MAE.

## The loop (roles mapped to existing kit)

| MAE role | FinInteract component | Co-evolution job |
|---|---|---|
| Proposer | Constructor (`scripts/construct_instances.py`, GPT-5/Opus over filings) | generate instances on the **weakest axis** |
| Judge    | Adversarial Verifier + GPT-4o grader | keep only **verifiable-with-context AND base-model-fails-without** ("frontier") |
| Solver   | Search/interact agent + **SFT/GRPO companion** (`c2_sft_train_gc.py`, §7.4 ladder) | train on the fresh frontier set |

```
measure per-axis skill  ->  Constructor generates on weakest axis W
        ^                              |
        |                    Verifier keeps frontier_W (verifiable & model-fails)
   re-measure on                       |
   HELD-OUT human W  <---  SFT/GRPO on train_W  <--- split frontier_W
```

The **per-axis skill analysis** (§6.4) selects the target: `recognition_policy`
has AxisHit@1 = 0 across OpenAI models and only **9 human instances** in v1 — too
few to train on, which is *precisely* what the data arm is for.

## Gated protocol — ONE round decides whether to go further

Target axis **W = `recognition_policy`** (clearest blind spot; robustness rerun
on `metric_definition`, the partial-skill axis, if round-0 is positive).

0. **Baseline.** Measure base Qwen3-4B on the **9 frozen human `recognition_policy`
   items**: AxisHit@1 (leak-proof) + **de-leaked accuracy** (uses
   `passages.jsonl` retrieval, *not* the oracle span — now possible after the
   passage-file fix; the old ladder could not report honest accuracy).
1. **Data arm.** Constructor generates *K*≈120 new `recognition_policy` instances
   from FY2024–25 EDGAR filings (contamination-safe). Verifier keeps only the
   **frontier**: verifiable *with* context AND base model fails *without*. Split
   `frontier_W` → `train_W` / `heldout_W`.
2. **Model arm.** Axis-guided SFT (+GRPO) on `train_W` only, reusing the §7.4
   recipe (privately reveal gold axis to the SFT teacher; hint NOT stored in the
   trajectory).
3. **Re-measure.** On **`heldout_W` + the 9 frozen human items** (never trained
   on), plus AxisHit on `entity_scope`/`metric_definition` to check for
   **catastrophic forgetting**.

### Decision rule
- **GO** — ΔAxisHit@1 on the **frozen human** items ≥ **+15 pt**, de-leaked
  accuracy up, other axes not degraded → the loop *generalizes*. Then (and only
  then) invest in the multi-round curriculum and write it up as a method
  (likely a follow-up / extended companion section, NOT folded into the Resource
  paper's core).
- **NO-GO** — gain appears only on self-generated `heldout_W` but not on the
  frozen human items, or is within noise → **still a clean Resource finding**:
  "a single targeted round installs axis-targeting on generated data but does not
  generalize to held-out human items → motivates multi-round co-evolution as
  future work." Publishable either way.

## What lands in the paper regardless of GO/NO-GO
- §3.2 Design Principles: add a **contamination-resistance / non-saturation**
  principle citing the engine.
- New §7.5 "Co-Evolving the Benchmark: A Self-Refreshing Loop": the loop figure,
  the round-0 result table (Round-0 vs Round-1 AxisHit + de-leaked accuracy on
  frozen human items), and the honest GO/NO-GO verdict.
- Related Work: position vs Multi-Agent Evolve (2510.23595, ungrounded self-play)
  and ACE / "Think Twice, Act Once" (ySWrJer7mW, LLM↔RL decision-making) — we are
  **grounded** data↔model co-evolution for a benchmark.
- Contributions: add "a self-refreshing construction↔training loop that keeps the
  benchmark hard as models improve."

## Guardrails
- Never edit `data/final/*.jsonl`. Generated instances go to
  `data/coevolve/round0/` and are a *separate* release artifact, not v1.x.
- The 9 frozen human `recognition_policy` items are a **held-out probe** — never
  in any training split.
- De-leaked accuracy must use `--passage-file data/sources/passages.jsonl`
  (retrieval), never the oracle span. If `evaluate.py` prints the no-passage
  warning, STOP.
- Report whatever the numbers show, including a NO-GO.
