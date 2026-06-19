# Larger-Model Interpretability — Qwen3-30B-A3B vs. Qwen3.5-35B-A3B

Self-contained execution of `experiments/gpu_eval/TASK_interpretability.md` on an
8× A100-40GB box. Frozen eval set `data/final/fininteract_v1.jsonl` (N=173, 53 EN / 120 ZH).

## Step 0 — Model ids (resolved)
- **Model A:** `Qwen/Qwen3-30B-A3B` (Qwen3 MoE, 48 layers, ~3B active) → label `qwen3-30b-a3b`
- **Model B:** `Qwen/Qwen3.5-35B-A3B` (Qwen3.5 MoE, 40 layers, ~3B active; arch
  `qwen3_5_moe`, served via its text pathway) → label `qwen3p5-35b-a3b`.
  Model B **was confirmed on the Hub and loads** in transformers 5.10, so the
  comparison proceeds (no stop-at-Step-0).

### Serving note (why not vLLM)
This box runs **driver 535 (CUDA 12.2)**, and the agent had to be served through a
small **custom HF-transformers OpenAI-compatible server** (`hf_openai_server.py`)
instead of vLLM, for two independent reasons:
1. vLLM 0.23 ships a **cu130** torch build that refuses to initialise on driver 535
   (`CUDA driver too old, found 12020`); and
2. vLLM 0.23's registry has **no `qwen3_5_moe`** entry — Model B is unservable on
   vLLM regardless of CUDA.
Both models therefore use the **same** HF serving stack (fairer A/B), greedy decoding,
**thinking-mode OFF** (these are hybrid-reasoning models; thinking-on is the optional
Step 3). Simulator (GPT-5) and grader (GPT-4o-mini) stayed on the OpenAI API, as required.

---

## Step 1 — Behavioural evaluation (`tab:main` rows)

```
Model              Ans-only  +Search  +Interact   IR   AxisHit@1   Ceiling
qwen3-30b-a3b         1.2      0.0       0.0       72     .23        90.2
qwen3p5-35b-a3b       0.0      2.3       0.0       77     .40        91.9
```
LaTeX (column order of `tab:main`):
```
qwen3-30b-a3b   & 1.2 & 0.0 & 0.0 & 72 & .23 & -- \\
qwen3p5-35b-a3b & 0.0 & 2.3 & 0.0 & 77 & .40 & -- \\
```
(ECE column `--`: confidence elicitation not run.)

**EN/ZH split (accuracy, bootstrap 95% CI from `analyze_breakdowns.py`):** both
models are **0.0%** in `+interact` in *both* languages (EN n=53, ZH n=120). For
calibration against the published ladder on the same judge: gpt-5 `+interact`
= 20.2% [14.5, 26.6] (interaction **helps**, Δ=+13.9, p<1e-4); gpt-4o = 4.6%;
gpt-5-mini = 0.0%. **Both A3B models pattern with gpt-5-mini: a flat +interact**
(interact−search Δ: 30B +0.0, p=1.00; 35B −2.3, p=0.07) sitting under a ~90% oracle ceiling.

### The elicitation *behaviour* differs sharply; the *outcome* does not
| | 30B | 35B |
|---|---|---|
| Interaction rate (IR) | 72% | 77% |
| AxisHit@1 (right axis when asking) | 0.23 | **0.40** |
| IR by language (EN / ZH) | **11% / 99%** | 85% / 74% |
| Finals that are still a *question* (never committed) | **131/173 (76%)** | **8/173 (5%)** |

- The **30B** is erratic and language-driven: in English it barely elicits (IR 11%,
  like prior work's 7%); in Chinese it asks almost always (IR 99%) but **gets stuck in
  ask-loops and refuses to commit** — 76% of its interactive transcripts end on a
  clarifying question *even after the harness explicitly says "Maximum questions
  reached. Please answer now."* (a genuine non-compliance failure, not a cutoff artifact).
- The **35B** elicits consistently across languages, hits the correct axis ~2× more
  often (0.40 vs 0.23), and **commits** to an answer (only 5% dangling). Its behaviour
  is the cleaner of the two.
- **Yet both end at 0% +interact.** Resolving *which* interpretation is wanted does not
  supply the underlying EDGAR/filing figure; the models then hallucinate the number
  (e.g. Tesla FY-revenue "$136.2B" vs gold "$97.69B"). Only the context-oracle ceiling —
  which injects the answer-bearing passages — recovers accuracy (90–92%). So the
  +interact↔ceiling gap is ~90 points for **both** models.

## Step 2 — Mechanistic probe (layer-wise axis decodability)

Per-layer linear probe (last-token, sklearn logistic regression) decoding the gold
axis `entity_scope` vs `metric_definition` (n=157 single-axis items; majority baseline
0.599). Figures: `probe_qwen3-30b-a3b.png`, `probe_qwen3p5-35b-a3b.png`, and the
overlay `probe_axis_overlay.png`.

```
Model              Peak axis-decode   Peak layer        vs baseline 0.599
qwen3-30b-a3b          0.714           L6/47  (~13% depth)     +0.115
qwen3p5-35b-a3b        0.740           L37/39 (~95% depth)     +0.141
```

Both decode the axis **well above chance** → the ambiguity *type* is linearly present
in each network. They differ in **where**: the 30B concentrates it **early** (~13%
depth), the 35B **late** (~95% depth) — the same depth shift seen in the earlier 4B/MoE
activation case study (30B ~27%, 35B ~85%): across the Qwen3→Qwen3.5 generation the axis
representation migrates toward the output end of the network.

## Step 3 — Thinking toggle: not run (optional). Both models evaluated thinking-off.

---

## Verdict (candidate *Finding* for §6)

> Across two comparably-sized A3B MoE models from different generations
> (Qwen3-30B-A3B, Qwen3.5-35B-A3B), the gold ambiguity axis is **linearly decodable
> from hidden states** (peak entity-vs-metric probe 0.71 / 0.74 vs 0.60 baseline; located
> early ~13% depth in Qwen3 and late ~95% depth in Qwen3.5), yet **+interaction
> accuracy stays flat at 0% under a ~91% context-oracle ceiling** for both — the
> elicitation gap is scale- and generation-invariant. The two models fail *differently*
> in the policy: the 30B under-elicits in English and loops without committing in Chinese
> (76% of transcripts never produce an answer), whereas the 35B elicits cleanly and on the
> correct axis ~2× more often (AxisHit 0.40 vs 0.23) and commits 95% of the time — but
> neither converts the resolved ambiguity into a correct figure. **The representation is
> present in both; the bottleneck is the policy that must turn recognition into a
> committed, correct answer — and better elicitation alone does not close it.**

### Caveats
- Agent served via HF transformers (not vLLM) with **thinking-off**; thinking-on (Step 3)
  could change commitment behaviour, especially the 30B's Chinese ask-loops.
- +interact is bounded by **factual recall**, not just elicitation: the simulator conveys
  the intended interpretation, not the answer value, so a model that lacks the specific
  filing figure cannot score even after a perfect clarifying question. The probe
  (representation) and the ceiling (reading) isolate this cleanly.
- Probe is correlational (decodability, not causal use); n=157 single-axis items.

## Deliverables
- `data/results/eval_open_{qwen3-30b-a3b,qwen3p5-35b-a3b}.jsonl` (519 rows each: 3 modes × 173)
- `data/results/eval_ceiling_{qwen3-30b-a3b,qwen3p5-35b-a3b}.jsonl` (173 each)
- `data/results/probe_{qwen3-30b-a3b,qwen3p5-35b-a3b}.json` + `.png`, `probe_axis_overlay.png`
