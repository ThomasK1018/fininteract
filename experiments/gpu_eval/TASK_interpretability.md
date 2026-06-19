# TASK: Larger-Model Interpretability — Qwen3-30B-A3B vs. the A3B sibling

**For the cloud agent:** this is a self-contained work order. Execute it against
this repository on an 8× A100-40GB (NVLink) box. Read it top to bottom, resolve
the one open question in Step 0, then run Steps 1–4 and commit the outputs.

---

## Objective

FinInteract's central claim is that the bottleneck for ambiguous financial
queries is **self-elicitation** (asking the right clarifying question), *not*
answering once the ambiguity is resolved. The OpenAI ladder shows this as a flat
+interact accuracy under a near-ceiling context-oracle score, and a flat AxisHit.

This task tests whether that pattern is **scale-/generation-invariant across two
comparably-sized MoE models**, and locates the difference *inside the network*:

- **Behaviourally** — do the two models differ in +interact accuracy, AxisHit, and
  the oracle ceiling, or is the elicitation gap the same for both?
- **Mechanistically** — is the gold ambiguity axis *linearly decodable* from each
  model's hidden states, and at which layers? If a model can't elicit but the
  axis is decodable mid-network, the failure is in the *policy*, not the
  *representation*.

**Hypothesis:** both models reach a high oracle ceiling and decode the axis
mid-network, yet both stay low on +interact/AxisHit — i.e. the representation
exists, the elicitation behaviour does not, regardless of model generation.

---

## Step 0 — Resolve the model ids (DO THIS FIRST)

Set the two checkpoints. **Model A is confirmed; Model B must be verified** — the
name "Qwen3.5-35B-A3B" has no known public release. Confirm the exact Hugging Face
id (or substitute the intended sibling) before running anything.

```bash
# Model A (confirmed): Qwen3 MoE, ~30B total / ~3B active
export MODEL_A_ID="Qwen/Qwen3-30B-A3B"        ;  export MODEL_A_LABEL="qwen3-30b-a3b"
# Model B (VERIFY the id — replace if "Qwen3.5-35B-A3B" is a typo/internal name)
export MODEL_B_ID="<CONFIRM-HF-ID>"           ;  export MODEL_B_LABEL="qwen3p5-35b-a3b"
```
If `MODEL_B_ID` cannot be resolved on the Hub, **stop and report** rather than
guessing — the whole comparison hinges on it. Both A3B models fit the box
(fp16 ≈ 60–70 GB, sharded across GPUs; use `TP=4` for vLLM).

## Preconditions
```bash
pip install "vllm>=0.6.3" openai transformers torch scikit-learn matplotlib
export OPENAI_API_KEY=sk-...   # user-simulator (GPT-5) + grader (GPT-4o-mini) only
# run everything from the repo root; data/final/fininteract_v1.jsonl must be present
```
All experiments use the **frozen** `data/final/fininteract_v1.jsonl` (N=173) so
results line up with the paper. **Do not modify the data files.** The agent under
test runs locally; the simulator and grader stay on the OpenAI API (same judge as
the published rows).

---

## Step 1 — Behavioural evaluation (both models)

For each model: serve with vLLM, run all three modes + the context-oracle ceiling,
tear down, then roll up.

```bash
for pair in "$MODEL_A_LABEL $MODEL_A_ID" "$MODEL_B_LABEL $MODEL_B_ID"; do
  set -- $pair; LABEL=$1; MODEL=$2
  ./experiments/gpu_eval/serve.sh "$LABEL" "$MODEL" 4 >/tmp/vllm_$LABEL.log 2>&1 &
  SRV=$!
  until curl -sf http://localhost:8000/v1/models >/dev/null; do sleep 10; done
  ./experiments/gpu_eval/eval_model.sh "$LABEL"
  kill $SRV; wait $SRV 2>/dev/null; sleep 5
done
python experiments/gpu_eval/rollup_open.py "$MODEL_A_LABEL" "$MODEL_B_LABEL"
```
**Produces:** `data/results/eval_open_<label>.jsonl`,
`data/results/eval_ceiling_<label>.jsonl`, and two paste-ready `tab:main` LaTeX
rows (Ans-only / +Search / +Interact / IR / AxisHit@1 / ECE) printed by the rollup.

## Step 2 — Mechanistic probe (both models, layer-wise axis decodability)

Run the existing probe once per model with `--base-model` only (no adapter); each
run trains a per-layer linear probe to decode the gold axis from hidden states.
Comparing the two curves is the interpretability result.

```bash
for pair in "$MODEL_A_LABEL $MODEL_A_ID" "$MODEL_B_LABEL $MODEL_B_ID"; do
  set -- $pair; LABEL=$1; MODEL=$2
  python experiments/sft_vs_rl/probe_sft_vs_base.py \
    --instances data/final/fininteract_v1.jsonl \
    --base-model "$MODEL" --rep last \
    --out "data/results/probe_${LABEL}.json" \
    --fig "data/results/probe_${LABEL}.png"
done
```
**Read it as:** peak per-layer probe accuracy ≫ chance ⇒ the axis is linearly
present (representation is fine); compare the two models' peak accuracy and the
layer at which it peaks. A model that scores low on AxisHit (Step 1) but high on
the probe (Step 2) has the knowledge and lacks the *behaviour* — the paper's
thesis, now shown inside two different networks.
> Note: `probe_sft_vs_base.py` was written for base-vs-SFT. Used with
> `--base-model` alone it yields a single model's curve, which is exactly what we
> diff across the two models. No code change is required; if you want a single
> overlaid figure, post-process the two JSONs.

## Step 3 — Optional: reasoning/thinking control

If either A3B model exposes a thinking toggle, the same-weights thinking-on vs
-off comparison isolates whether test-time reasoning changes *recognition* vs
*answering*. Skip unless time permits; the matched-pair design in
`reasoning_sweep.sh` is the fallback.

## Step 4 — Report

Write `data/results/FINDINGS_interpretability.md` containing:
1. The two `tab:main` rows (from `rollup_open.py`) with EN/ZH split
   (`python scripts/analyze_breakdowns.py` globs the new result files for CIs).
2. The two probe peak accuracies + peak layers, and the overlaid/again-side-by-side
   figures.
3. A 3–5 sentence verdict: does the elicitation gap replicate across both models,
   and is the gold axis decodable despite low AxisHit? State it as a candidate
   *Finding* sentence for §6.

---

## Deliverables (commit these)
- `data/results/eval_open_<A>.jsonl`, `eval_open_<B>.jsonl`
- `data/results/eval_ceiling_<A>.jsonl`, `eval_ceiling_<B>.jsonl`
- `data/results/probe_<A>.json/.png`, `probe_<B>.json/.png`
- `data/results/FINDINGS_interpretability.md`

```bash
git add data/results/eval_open_*.jsonl data/results/eval_ceiling_*.jsonl \
        data/results/probe_*.json data/results/probe_*.png \
        data/results/FINDINGS_interpretability.md
git commit -m "Larger-model interpretability: behavioural + probe results for the two A3B models"
git push    # tracks personal/main (ThomasK1018/fininteract)
```

## Guardrails
- **Never edit** `data/final/*.jsonl` (frozen eval set) or the answer-key sheets.
- Keep the simulator/grader on the OpenAI API; only the agent-under-test is local.
- If `MODEL_B_ID` is unresolved, **stop at Step 0 and report** — do not substitute
  a different-sized model silently, since the comparison is size-matched by design.
- Expected OpenAI spend: ~$1–2 per model (sim+grader). Local inference is free.
