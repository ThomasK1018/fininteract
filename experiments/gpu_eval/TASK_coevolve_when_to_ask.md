# TASK: The when-to-ask gate (fix the entity over-elicitation regression)

**Read `data/coevolve/entity/FINDINGS_coevolve_entity.md` + `.../metric/...` first.** §7.5
found axis-guided SFT drives interaction to 100% and the accuracy sign flips on whether
the base could already answer: asking **helps metric** (base can't answer without the
definition, $15\to60$, $+45$) but **hurts entity** ($32.5\to22.5$, $-10$: it asks even on
items it could answer directly and converts 6 correct answers into wrong multi-turn ones).
This task installs a **when-to-ask gate** — ask only when the model cannot already answer —
and tests that it **recovers the entity regression while keeping the metric gain**.

## Success criterion (state before running)
Gated policy on frozen human probes: entity de-leaked accuracy **≥ base (no $-10$)** AND
metric accuracy **stays ≈ +45 over base**, with ask-rate dropping from 100% toward each
axis's base answer-only *failure* rate (entity asks little; metric asks a lot).

## Two variants — run both, they answer different questions

### A. Train-time gate (the principled fix; recommended)
Teach *when* to ask by mixing demo types. For each frontier/train instance, check whether
the **base** answers it answer-only:
- base **correct**  -> demo is *answer directly* (no interact turn).
- base **wrong**    -> demo is *ask on-axis then answer* (the existing recipe).
```bash
# 1. label train_W by base answer-only correctness (reuse genW_baseonly outputs if present)
python scripts/evaluate.py --instances data/coevolve/entity/train_W.jsonl \
  --models qwen3-4b --modes answer-only --agent-base-url http://localhost:8000/v1 \
  --out data/coevolve/entity/gate_baseonly.jsonl --summary /dev/null
# 2. build gated demos: no-ask demos for base-correct, ask-on-axis for base-wrong
#    extend scripts/gen_axis_guided_demos.py with a --gate flag that, when the instance
#    id is in the base-correct set, emits a 3-msg [system,user,assistant(answer)] demo
#    (no interact/search turns) instead of the 7-msg ask trajectory.
python scripts/gen_axis_guided_demos.py --instances data/coevolve/entity/train_W.jsonl \
  --axis entity_scope --gate data/coevolve/entity/gate_baseonly.jsonl \
  --teacher-model gpt-5-mini --out data/coevolve/entity/demos/sft_gated.jsonl
accelerate launch experiments/gpu_eval/c2_sft_train_gc.py \
  --model Qwen/Qwen3-4B-Instruct --data data/coevolve/entity/demos/sft_gated.jsonl \
  --output outputs/coevolve/sft_entity_gated
```

### B. Inference-time self-gate (cheap baseline; no retrain)
Two-stage policy on the **already-trained** co-evolved model: first attempt answer-only;
if the model is not confident (self-report "UNSURE", or answer-only would be graded wrong
by a held-out check / low self-consistency across 3 samples), *then* run the interact
policy. Report as a no-training reference point.

## Evaluation (both variants, both axes)
Run on the **frozen human probes** (`probe_human_entity.jsonl`, `probe_human_metric.jsonl`)
and report, per axis, three policies:

| policy | de-leaked acc | ask-rate | AxisHit@1 (on asked) | gate precision* |
|---|--:|--:|--:|--:|
| base (no interact) | … | 0 | -- | -- |
| always-ask (current co-evolved) | … | 100 | … | -- |
| **gated (A)** | … | … | … | … |
| gated (B, inference) | … | … | … | … |

`*gate precision = P(base actually wrong | model chose to ask)` — a well-calibrated gate asks
mostly on the items it needs to.

## Deliverable — `data/coevolve/FINDINGS_when_to_ask.md`
The table above for entity AND metric, plus the explicit verdict on the success criterion:
did the gate **recover entity** (accuracy back to ≥ base) while **keeping metric** (+45)?
Also report the metric round as a control (a good gate should barely change metric, since the
base fails almost everything there). Commit to branch `coevolve/when-to-ask`, push.

## Guardrails
- Frozen human probes are held-out — never in any training split. Never edit `data/final/*`.
- De-leaked accuracy MUST use `--passage-file`. Sim/grader on OpenAI, policy local.
- Report the honest verdict, including if the gate under-asks on metric (which would cost
  the +45 — the failure mode to watch).
