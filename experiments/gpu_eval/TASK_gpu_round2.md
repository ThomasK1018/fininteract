# TASK: GPU round 2 — causality, reasoning, scaling, gap-decomposition

**For the cloud agent:** four experiment groups on the 8× A100-40GB box, against
this repo. The frozen eval set is `data/final/fininteract_v1.jsonl` (N=173). The
agent-under-test runs locally; the user-simulator (GPT-5) and grader/axis-judge
(GPT-4o-mini) stay on the OpenAI API (`OPENAI_API_KEY`). Commit every result file
under `data/results/` and push. Order below is by value; do as many as time allows.

---

## A. Causal steering along the ambiguity direction  (`steer_axis.py`)  ← centerpiece

Turns §6.6's *correlational* probe into a *causal* claim. Extracts the per-layer
ambiguity direction (mean final-token activation of `Q+C` minus bare `Q`), adds
`alpha*dir` to the residual stream, and measures whether the agent's **first
action** shifts toward asking (IR) and the right axis (AxisHit@1). alpha is swept
over both signs — the data picks the direction.

```bash
for M in "Qwen/Qwen3-30B-A3B" "Qwen/Qwen3.5-35B-A3B"; do
  lbl=$(basename "$M" | tr 'A-Z' 'a-z' | tr '.' 'p')
  python experiments/gpu_eval/steer_axis.py --model "$M" \
    --layers 0.4 0.6 0.8 --alphas -8 -4 0 4 8 --n-dir 40 \
    --out "data/results/steer_${lbl}.json"
done
```
**Report:** IR and AxisHit@1 as a function of alpha (and per steer-layer if you
sweep `--layers` separately). **The claim is earned only if** steering moves IR/
AxisHit monotonically with alpha while alpha=0 reproduces the baseline IR (72/77%).
If +alpha raises asking and −alpha suppresses it, that is direct causal evidence
the axis representation drives elicitation. Negative result (flat under steering)
is also publishable — it would mean the representation is read-only for the policy.
Tip: if effects are weak, raise alpha magnitude or steer a band of mid-late layers
(0.5–0.9) together.

## B. Reasoning ON vs OFF, same weights  (`evaluate.py --agent-thinking`)

The A3B run was thinking-OFF. Toggle thinking on the *identical* weights to isolate
whether test-time reasoning fixes recognition/commitment (esp. the 30B's Chinese
ask-loops, 76% never-commit). The flag injects
`chat_template_kwargs.enable_thinking` on agent calls only.

```bash
# serve as before (serve.sh), then for each model and each toggle:
for T in on off; do
  python scripts/evaluate.py --instances data/final/fininteract_v1.jsonl \
    --models qwen3-30b-a3b --modes answer+search+interact \
    --agent-base-url http://localhost:8000/v1 --agent-thinking $T \
    --out data/results/eval_think_${T}_qwen3-30b-a3b.jsonl \
    --summary data/results/eval_think_${T}_qwen3-30b-a3b.summary.json
done
```
**Report:** +interact accuracy, IR, AxisHit, and *commit rate* (1 − fraction of
transcripts ending on a question) for ON vs OFF. Key question: does thinking make
the 30B **commit** in Chinese, and does either model convert a resolved ambiguity
into a correct figure?

## C. Scaling sweep + GRPO scale-up

**C1 — scaling sweep (ready):** `scaling_sweep.sh` runs Qwen3 0.6B→32B through all
modes + ceiling. Produces the *"scale lifts the oracle ceiling, not self-
elicitation"* figure (§6.3). Then `python experiments/gpu_eval/rollup_open.py`.

**C2 — scale the GRPO ladder to 8B/14B/32B:** rerun the axis-guided SFT→KTO→GRPO
ladder (the training harness used for the 4B result, on this box) with the base
model swapped to `Qwen/Qwen3-8B`, `-14B`, `-32B`. Keep the **axis-guided teacher**
(private gold-axis hint, not stored in the trajectory) and score with the
**leak-proof** metrics (AxisHit@1, interaction rate) per §7.4. Goal: show the
recipe's gains hold (or that the teacher trick is still required) at larger scale.
Fits the box: 32B QLoRA training is comfortable across the 8 GPUs.

## D. Disentangle elicitation vs. recall  (`--modes interp-oracle` + ECE)

Addresses the reviewer confound: is +interact 0% about *not asking* or *not
recalling the value once the reading is known*? The new **interp-oracle** mode
injects the resolved interpretation `C` but **no** evidence spans and no search.

```
  full context-oracle (C + both spans)  ~91%   <- reading only
  interp-oracle       (C, no spans)        ?    <- recall given known interpretation
  +interact           (must elicit C)     0%    <- elicitation + recall
```
- **interp-oracle − +interact = elicitation cost**; **ceiling − interp-oracle = recall cost.**

```bash
python scripts/evaluate.py --instances data/final/fininteract_v1.jsonl \
  --models qwen3-30b-a3b --modes interp-oracle \
  --agent-base-url http://localhost:8000/v1 \
  --out data/results/eval_interp_oracle_qwen3-30b-a3b.jsonl \
  --summary data/results/eval_interp_oracle_qwen3-30b-a3b.summary.json
# also fill the ECE column for any model: add --elicit-confidence to its interact run
```
**Report:** the three-way decomposition per model (and EN/ZH), plus the now-filled
ECE cells. If recall cost dominates, the headline reframes from "can't ask" to
"can't recall even when told what to fetch" — important to state precisely.

---

## Deliverables (commit + push)
`data/results/steer_*.json`, `eval_think_{on,off}_*.jsonl`,
`eval_open_*` + `eval_ceiling_*` from the sweep, GRPO ladder metrics,
`eval_interp_oracle_*.jsonl`, and a short `FINDINGS_gpu_round2.md` with: the
steering IR/AxisHit-vs-alpha curve, the reasoning ON/OFF table, the scaling figure
data, and the elicitation-vs-recall decomposition. Send these back and the rows/
figures get folded into §6.3/§6.6/§7.4.

## Guardrails
- Never edit `data/final/*.jsonl` or answer-key sheets.
- Simulator/grader stay on the OpenAI API; only the agent-under-test is local.
- Steering and reasoning toggles are agent-only — do not apply them to the judge.
- Report negative results plainly; a flat steering curve or no reasoning gain is a
  real finding, not a failure.
