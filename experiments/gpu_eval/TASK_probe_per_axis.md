# TASK: Per-axis linear-probe decodability — the MECHANISTIC definition of "learnable"

**Read `data/results/FINDINGS_learnability.md` first.** The paper (§7.6, Finding 13)
defines an axis's *learnability* behaviourally, as the **scale-slope of AxisHit**
(entity Δ+0.77 > metric +0.21 > recognition +0.00). This task replaces that proxy with a
**representational** definition:

> **Learnable ⟺ the axis is linearly decodable from the model's hidden states.**

Prediction: entity (and to a lesser degree metric) is linearly decodable; **recognition
policy is not** — or only in the largest models. If decodability tracks the scale-slope,
we have a mechanistic grounding for the learnability condition, not just a behavioural one.

## Tool
`experiments/sft_vs_rl/probe_sft_vs_base.py` already extracts per-layer last-token
activations and trains a per-layer logistic-regression axis probe (5-fold CV). It takes
`--base-model`, optional `--adapter`/`--sft-model`, `--instances`, `--out`.

## What to run

### 1. Per-axis ONE-VS-REST decodability across scale
For each model in the sweep (Qwen3 **4B, 8B, 14B, 32B**) and each axis
$A\in\{$entity, metric, recognition, temporal$\}$, decode a binary label
"operative axis is $A$ vs not" from the bare-question hidden states over
`data/final/fininteract_v1.jsonl`. (If the current probe is multiclass, add a
`--per-axis` one-vs-rest mode, or post-process the multiclass confusion into per-axis
recall — either gives per-axis decodability.)
```bash
for M in Qwen/Qwen3-4B-Instruct Qwen/Qwen3-8B Qwen/Qwen3-14B Qwen/Qwen3-32B; do
  tag=$(echo $M | sed 's#.*/##')
  python experiments/sft_vs_rl/probe_sft_vs_base.py --base-model "$M" \
    --instances data/final/fininteract_v1.jsonl \
    --out data/results/probe_peraxis_${tag}.json
done
```
Report **peak per-axis decodability (accuracy or AUROC over a balanced split) per model
size**, and the **decodability-vs-scale slope per axis**.

### 2. Does axis-guided SFT RAISE decodability of the trained axis?
Using the co-evolution adapters already produced:
```bash
# entity adapter (GO): expect entity decodability to RISE
python experiments/sft_vs_rl/probe_sft_vs_base.py --base-model Qwen/Qwen3-4B-Instruct \
  --adapter outputs/coevolve/sft_entity \
  --instances data/final/fininteract_v1.jsonl --out data/results/probe_entity_sft.json
# recognition adapter (NO-GO): expect recognition decodability to STAY flat
python experiments/sft_vs_rl/probe_sft_vs_base.py --base-model Qwen/Qwen3-4B-Instruct \
  --adapter outputs/coevolve/sft_recognition_v2 \
  --instances data/final/fininteract_v1.jsonl --out data/results/probe_recognition_sft.json
```

## The comparison that matters
Build one table: **per-axis behavioural learnability (AxisHit scale-slope, from
`scripts/analyze_learnability.py`) vs. representational decodability (this probe)**.
The claim lands if the two **correlate** — decodable axes are the learnable ones — and
if **recognition is the axis that is neither decodable nor learnable**, while entity is
both. That upgrades §7.6's learnability from a behavioural proxy to a mechanistic one.

## Deliverables — `data/results/FINDINGS_probe_peraxis.md`
- Per-axis peak decodability × {4B,8B,14B,32B} table + per-axis decodability-scale slope.
- SFT-raises-decodability check (entity ↑ expected, recognition flat expected).
- The behavioural-vs-representational correlation (does decodability predict the GO/NO-GO?).
Commit to branch `probe/per-axis`, push. No API needed (hidden states only; local HF).

## Guardrails
- Balance one-vs-rest classes (rare axes: recognition n=9, temporal n=7 — report CIs /
  note underpower; entity n=94, metric n=108 are solid).
- Never edit `data/final/*`. This is read-only w.r.t. the benchmark.
