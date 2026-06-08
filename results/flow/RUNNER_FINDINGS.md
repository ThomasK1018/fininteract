# FinInteract Interpretability — Flow Experiments: Findings

**Date:** 2026-06-08
**Device:** Linux workstation, single NVIDIA RTX 4090 (24 GB)
**Kit:** `fininteract_interp_kit` (depth-flow + generation-flow studies)
**Benchmark:** `data/fininteract_v1.jsonl` — 173 frozen instances (EN/ZH financial-query ambiguity)

This document records the two **flow-based** experiments run on two open models, the full
results, and the caveats. The behavioral runner (`run_behavior.py`, needs an OpenAI key) and
the linear-probe analysis (`analyze_probes.py`) were **not** part of this run — only the flow
studies were requested.

---

## 1. Models tested

| Model | Role | Layers (+emb) | Size (bf16) | Env |
|---|---|---|---|---|
| `Qwen/Qwen3-4B-Instruct-2507` | baseline | 37 | ~8 GB | `thomas_env` (transformers 4.57.6) |
| `Qwen/Qwen3.5-4B` | newest / SOTA | 33 | 9.3 GB | `qwen35_env` (transformers 5.10.2) |

### Setup notes / gotchas (important for the other device)
- **The kit's default model name `Qwen/Qwen3-4B-Instruct` does NOT exist on HF (404).** The real
  bilingual instruct model is **`Qwen/Qwen3-4B-Instruct-2507`** — used as the baseline.
- **`Qwen/Qwen3.5-9B` was requested first but did not fit:** it is 19.3 GB bf16 and the 4090 had
  only ~17 GB free (a 7 GB `ollama` process was resident). Per the fallback instruction we used
  **`Qwen/Qwen3.5-4B`** (9.3 GB), which fit alongside ollama.
- **Qwen3.5 is a brand-new architecture:** `Qwen3_5ForConditionalGeneration` / `model_type:
  qwen3_5` — a multimodal VL model with **hybrid linear + full attention** (periodic full-attn
  every 4 layers). It is **not supported by transformers 4.57**; it needs **transformers ≥ 5.x**.
  We built a dedicated env (`qwen35_env`, cloned from `thomas_env` + `pip install -U
  transformers>=5.10` → 5.10.2, huggingface-hub 1.18) so the existing vLLM env was left intact.
- **No kit code changes were needed for Qwen3.5.** `AutoModelForCausalLM` auto-routes the config
  to `Qwen3_5ForCausalLM` for text-only input, `AutoTokenizer` works (Qwen2Tokenizer), and
  `out.hidden_states` is a clean 33-entry tuple — exactly the interface the kit expects. The only
  message is a perf warning (no `flash-linear-attention` installed → slower torch fallback).

---

## 2. Method recap

A single **"ambiguity score"** = projection of a hidden state onto the per-layer **diff-in-means
ambiguity direction** (mean activation on the bare ambiguous question − mean on the
question+disambiguating-context version), centered at the midpoint of the two class means.

- **Depth-flow** (`probe_activations.py` → `analyze_flow.py`): how that score separates ambiguous
  vs disambiguated prompts **across layers**, and at which depth the model distinguishes ambiguity
  *types* (entity_scope vs metric_definition).
- **Generation-flow** (`flow_generate.py` → `analyze_flow.py`): how the score evolves
  **token-by-token** as the model writes its answer, and whether the internal ambiguity peak
  precedes the first spoken clarification token (`?`, "which", "specify", 哪/请/年份…).

Per-axis instance counts (identical for both models, same benchmark):
`entity_scope=94, metric_definition=63, recognition_policy=9, temporal_scope=7`.
Only entity-vs-metric is statistically supported; the rarer axes are shown for reference only.

---

## 3. Headline comparison

| Metric | Qwen3-4B-Instruct-2507 (baseline) | **Qwen3.5-4B (SOTA)** |
|---|---|---|
| Hidden layers (+emb) | 37 | 33 |
| **Depth-flow peak separation layer** | 35 (≈97% depth) | **32 (100% depth)** |
| Peak separation magnitude | 91.6 | 48.6 |
| Entity-vs-metric peak layer | 35 | 32 |
| **Clarification rate** (asked / 173) | 49% (85) | **69% (120)** |
| Mean clarification onset token | 41.1 | 53.7 |
| "Peak before ask" (script metric) | 100% | 82% |
| Peak-at-token-0 (artifact check) | 97% | 79% |

### Interpretation
1. **Depth-flow replicates and is the robust finding on both models.** Ambiguity is linearly
   encoded, and the entity-vs-metric *type* distinction emerges, in the **final layers**
   (L35/37 baseline; L32/33 Qwen3.5 — i.e. the very top of the stack). Curves rise late and
   sharply; on Qwen3.5 the rise is flat until ~L25 then jumps steeply. Entity and metric
   directions split with opposite sign (entity positive, metric negative). → "The model
   represents ambiguity, and its kind, in mid-to-late layers" holds for both.
2. **The newer model asks more.** Qwen3.5-4B emits a clarifying token in **69%** of cases vs
   **49%** for the baseline — it defaults-and-answers less often. This is the desirable direction
   for the behavioral default-capture story.
3. **The generation-flow "recognition precedes the ask" headline is largely an ARTIFACT — do not
   cite it as evidence.** The script computes the ambiguity "peak token" over the last third of
   layers, but that projection is **maximal at the very first generated token and then decays**.
   So for 97% (baseline) / 79% (Qwen3.5) of instances the peak sits at token 0, making
   "lead = onset − 0 ≈ onset" trivially positive. It is **not** evidence of internal recognition
   firing just before the clarification. The depth-flow result is the one to report.

---

## 4. Figures (per model, in `results/<model>/figs/`)

| File | Shows |
|---|---|
| `fig_depth_separation.png` | ambiguous − disambiguated projection per layer; dashed line = peak layer |
| `fig_axis_depth_curves.png` | per-axis ambiguity-projection curves across layers |
| `fig_entity_vs_metric.png` | entity vs metric onto the axis-discriminative direction |
| `fig_generation_heatmap.png` | layer × generated-token ambiguity score; cyan line = mean clarify onset |
| `fig_recognition_onset.png` | histogram of (clarify-onset − ambiguity-peak); see artifact caveat above |

---

## 5. Caveats (state these in any write-up)
- **Diff-in-means is correlational, not causal.** A causal claim needs activation steering
  (push the direction, measure change in asking) — not done here.
- **Recognition-onset metric is artifactual** (see §3.3). Cite depth-flow, not generation-onset.
- **Open model ≠ closed frontier.** Mechanistic evidence on an open proxy only.
- **Axis support:** entity-vs-metric only; temporal (7) and recognition (9) shown qualitatively.
- **Qwen3.5 perf path:** ran on the torch fallback for linear attention (no flash-linear-attn);
  results are correct, generation was just slower.

---

## 6. File manifest (this bundle)

```
FINDINGS.md                                  this document
data/fininteract_v1.jsonl                    the 173-instance benchmark
src/                                         the kit scripts (for reproduction)
results/baseline_qwen3-4b-instruct-2507/
    flow_report.json                         depth + generation flow report
    genflow.npz                              [173 x 37 x 64] token flow (nan-padded) + onsets
    genflow.gen.jsonl                        per-instance generation text + clarify_onset
    figs/*.png                               5 figures
results/qwen3.5-4b/
    flow_report_qwen35.json
    genflow_qwen35.npz                       [173 x 33 x 64]
    genflow_qwen35.gen.jsonl
    figs/*.png
```

**Not included (large):** raw per-layer activation files
`acts_bare.npz` (157 MB) and `acts_bare_qwen35.npz` (139 MB). These are only needed to
*re-derive the ambiguity directions from scratch* (i.e. re-run the depth-flow). The depth-flow
results are already baked into `flow_report*.json`, and `genflow*.npz` is enough to re-do all
generation-flow / onset analysis. Ask if you want the raw activations shipped separately.

## 7. Reproduce / re-analyze on the other device
```bash
# Re-generate figures or re-run the depth/gen-flow analysis from the included results:
python src/analyze_flow.py --genflow results/qwen3.5-4b/genflow_qwen35.npz \
    --out /tmp/report.json          # (depth-flow needs the raw acts npz; gen-flow does not)
python src/plot_flow.py --report results/qwen3.5-4b/flow_report_qwen35.json \
    --genflow results/qwen3.5-4b/genflow_qwen35.npz --outdir /tmp/figs

# Full re-run from scratch needs a GPU + the env:
#   Qwen3.5  -> transformers >= 5.10  (model_type qwen3_5)
#   baseline -> transformers 4.57 is fine; model = Qwen/Qwen3-4B-Instruct-2507
```
