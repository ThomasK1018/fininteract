# FinInteract

A bilingual (EN/ZH) benchmark for evaluating financial search agents on **ambiguous
queries** — questions that are *easy to verify but ambiguous to resolve*. Each instance pairs a
question with a **default** interpretation (what a non-expert assumes) and an **intended**
interpretation (fixed by a disambiguating context), both grounded in verbatim filing spans.

## Repository layout

```
data/final/           Frozen benchmark: fininteract_v1.jsonl (173 instances) + DATASET_CARD.md
data/sources/         Passage pool (EDGAR XBRL, CNINFO/akshare, derived passages)
data/constructed/     Working construction output
scripts/              Construction, evaluation, analysis, and study harnesses
docs/                 methodology.md, motivation.md, dataset_analysis.md
paper/                LaTeX source (main.tex, refs.bib)
```

## Key scripts

| Script | Purpose |
|--------|---------|
| `construct_instances.py` | 3-role construction pipeline (constructor → 14-rule QC → adversarial verifier) |
| `evaluate.py` | Multi-model ReAct evaluation (3 modes; OpenAI + open-weight via compatible APIs) |
| `analyze_results.py` | Default-capture, IR-vs-H₀ calibration, first-action, item difficulty |
| `constructor_ablation.py` | How different LLMs generate under the pipeline (reproducibility study) |
| `probe_activations.py` / `analyze_probes.py` | Mechanistic interpretability probes |
| `dise.py` | Disambiguation Efficiency (DisE⁺) metric |

## Quick start

```bash
pip install openai tqdm edgartools akshare numpy scikit-learn
export OPENAI_API_KEY=...

# evaluate models on the frozen benchmark
python scripts/evaluate.py --instances data/final/fininteract_v1.jsonl \
    --models gpt-5 gpt-4o --modes answer-only answer+search answer+search+interact \
    --elicit-confidence --out data/results/eval.jsonl

# post-hoc analyses
python scripts/analyze_results.py --results data/results/eval.jsonl
```

## Data sources & licensing

Built from public SEC EDGAR filings (XBRL) and A-share annual reports (CNINFO/akshare).
Third-party datasets used during construction (FinEval, FinQA, FinanceQA, cflue, FinanceBench,
DocFinQA, BBT-FinCUGE) are **not redistributed here** — clone them from their original sources.

## Methodology

See `docs/methodology.md` for the full construction pipeline, the 14-rule QC layer, the five-axis
financial-ambiguity taxonomy, and the evaluation framework; `docs/dataset_analysis.md` for the
empirical analysis of the frozen dataset.
