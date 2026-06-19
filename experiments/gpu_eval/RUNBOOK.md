# GPU eval runbook — open-weight breadth, scaling sweep, reasoning depth

Target box: **8× A100-SXM4-40GB, NVLink (320 GB total)**. Everything below is scoped
to fit it. The **agent-under-test runs locally** (vLLM, OpenAI-compatible); the
**user-simulator (GPT-5) and grader (GPT-4o-mini) stay on the OpenAI API**, so all
local rows are graded by the *same* judge as the existing OpenAI rows and are
directly comparable. All experiments run on the frozen `fininteract_v1.jsonl`
(N=173) so they line up with the published tables.

## 0. Prereqs (on the GPU box)
```bash
pip install "vllm>=0.6.3" openai          # + the repo's scripts/ deps
export OPENAI_API_KEY=sk-...               # sim + grader only
git clone <this repo> && cd finteractcomp  # needs data/final/ + scripts/
```
Cost: each model run issues ~4 GPT-5 sim calls + ~3 GPT-4o-mini grader calls per
instance in interact mode → roughly **$1–2 of OpenAI usage per model**, ~$15–25
for the whole roster. Local inference is free.

## 1. Smoke test (5 min, do this first)
```bash
# terminal A — serve the smallest model on 1 GPU
./experiments/gpu_eval/serve.sh qwen3-4b Qwen/Qwen3-4B 1
# terminal B — run 10 instances through all modes + ceiling
INSTANCES=data/final/fininteract_v1.jsonl \
  ./experiments/gpu_eval/eval_model.sh qwen3-4b   # (add --limit via evaluate.py if you fork it)
python experiments/gpu_eval/rollup_open.py qwen3-4b
```
If the LaTeX row prints, the wiring is good.

## 2. Scaling sweep (#2) — the headline figure
Serves Qwen3 0.6B→32B in turn, all modes + ceiling, tearing down between sizes:
```bash
OPENAI_API_KEY=$OPENAI_API_KEY ./experiments/gpu_eval/scaling_sweep.sh
python experiments/gpu_eval/rollup_open.py    # rolls up every eval_open_*.jsonl
```
**What to plot:** x = params (log), two lines — context-oracle ceiling (expected
to *rise* toward ~95%) and +interact accuracy / AxisHit (expected ~*flat*). The
widening gap is "scale buys latent capacity, not self-elicitation." Reuse
`scripts/plot_crossmodel_flow.py` / `analyze_breakdowns.py` for CIs.
Wall-clock: ≈45–70 min/size → ~5–6 h for the six sizes.

## 3. Large open models (#1) — fill the empty tab:main rows
Run individually (each needs most of the box). Quantized MoE checkpoints (AWQ)
keep the giants inside 320 GB.
```bash
# Llama-3.3-70B (dense, fp16, TP8)
./serve.sh llama70b meta-llama/Llama-3.3-70B-Instruct 8 ;  ./eval_model.sh llama70b
# GLM-4.5-Air (~106B MoE, fp16, TP8)
./serve.sh glm45air zai-org/GLM-4.5-Air 8 ;                ./eval_model.sh glm45air
# Qwen3-235B-A22B (MoE, 4-bit AWQ, TP8) — tighten MAXLEN if KV cache OOMs
MAXLEN=8192 ./serve.sh qwen3-235b Qwen/Qwen3-235B-A22B-Instruct-2507-AWQ 8 --quantization awq
./eval_model.sh qwen3-235b
python experiments/gpu_eval/rollup_open.py llama70b glm45air qwen3-235b
```
**Out of scope on this box:** DeepSeek-V3.1 (671B) — too large even 4-bit
(~350 GB+ weights before KV). Report it as API-only or omit; don't force it.

## 4. Reasoning vs non-reasoning (#4a)
Matched 32B pairs, no code change — the model is the only variable:
```bash
OPENAI_API_KEY=$OPENAI_API_KEY ./experiments/gpu_eval/reasoning_sweep.sh
python experiments/gpu_eval/rollup_open.py qwen2.5-32b qwq-32b r1-distill-32b
```
**Read it as:** if QwQ/R1-distill lift the *ceiling* but not *AxisHit/+interact*,
reasoning improves answering-once-resolved, not ambiguity *recognition* — a clean
result that reinforces §6.3/§7. *Advanced same-weights control:* serve Qwen3-32B
twice, toggling thinking via vLLM `--reasoning-parser deepseek_r1` and an
`enable_thinking` chat-template kwarg; needs a 1-line `extra_body` patch in
`evaluate.py:chat()` to pass `chat_template_kwargs`. Skip unless a reviewer asks.

## 5. Mechanistic probe at scale (#4b)
`experiments/sft_vs_rl/probe_sft_vs_base.py` already decodes the gold axis from
hidden states layer-by-layer (base vs base+SFT-adapter). On this box, rerun it on
Qwen3-8B/14B/32B to show (a) *where* axis-awareness concentrates (which layers)
and (b) that axis-guided SFT *raises* axis-decodability — the representational
correlate of the §7.4 behavioural gains. This loads weights directly (HF
`transformers`), no vLLM server needed.

## 6. Folding results into the paper
- `tab:main`: paste the LaTeX lines from `rollup_open.py` into the open-weight
  block; keep two decimals for AxisHit/ECE to match the OpenAI rows.
- Scaling figure: new `fig:scaling` near §6.3 (latent capacity) — it *is* the
  scale-invariance evidence that section currently asserts.
- `analyze_breakdowns.py` already globs `data/results/eval_*.jsonl`, so EN/ZH and
  per-axis splits + bootstrap CIs extend to the new models automatically.

## Roster cheat-sheet (fits 8× A100-40GB)
| Model | Params | dtype | TP | Role |
|---|---|---|---|---|
| Qwen3-0.6B … 32B | 0.6–32B | fp16 | 1–4 | scaling sweep |
| Llama-3.3-70B-Instruct | 70B dense | fp16 | 8 | tab:main row |
| GLM-4.5-Air | ~106B MoE | fp16 | 8 | tab:main row |
| Qwen3-235B-A22B (AWQ) | 235B MoE | 4-bit | 8 | tab:main row (tighten MAXLEN) |
| QwQ-32B / R1-Distill-32B | 32B | fp16 | 4 | reasoning vs Qwen2.5-32B |
| DeepSeek-V3.1 | 671B | — | — | **too big — API-only / omit** |
