# FinInteract — Supplementary Code and Data

This package accompanies the paper *FinInteract: A Fine-Grained Financial Ambiguity
Benchmark*. It contains the frozen benchmark, the full construction and evaluation
pipeline, the analysis and plotting scripts, and the raw evaluation outputs needed to
regenerate every table and figure in the paper.

This material is anonymized for double-blind review. It contains no author, institution,
or repository-identifying information, and no API keys.

## Package layout

```
README.md                     This file
requirements.txt              Python dependencies
configs/
  openrouter.example.json     Template for API access (copy to openrouter.json, add your key)
data/
  fininteract_v1.jsonl        Frozen benchmark used for the experiments (N = 173)
  fininteract_v1.1.jsonl      Public release (N = 172; one consensus-rejected item quarantined)
  DATASET_CARD.md             Datasheet: provenance, schema, intended use, licensing
  CHANGELOG.md                Version history (v1 -> v1.1)
scripts/                      Construction, evaluation, analysis, and plotting code
results/                      Raw per-model evaluation outputs (JSONL), one row per instance
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10+ recommended
pip install -r requirements.txt

# API access. Native OpenAI models read OPENAI_API_KEY from the environment.
# Cross-vendor models route through OpenRouter via a small JSON config:
cp configs/openrouter.example.json configs/openrouter.json
#   then edit configs/openrouter.json and insert your own OpenRouter key.
```

## Data schema

Each line of `data/fininteract_v1.jsonl` is one instance with fields including:
`instance_id`, `language`, `source`, `ticker`, `company`, `filing_type`, `filing_date`,
`question`, `context` (the disambiguating context C), `answer` (intended, exactly
verifiable), `default_answer`, `intended_evidence_span`, `default_evidence_span`,
`intended_interpretation` / `default_interpretation` (each an {entity, period, metric,
basis} tuple), `axes` (the ambiguity categories touched), `n_axes`, `h0`
(log2 of the plausible-interpretation count), and `qc` (per-rule pass flags plus the
verifier blind-solve rate). See `data/DATASET_CARD.md` for the complete datasheet.

## Reproducing the main results

**1. Evaluate a model on the benchmark** (three interaction modes):

```bash
export OPENAI_API_KEY=...
python scripts/evaluate.py \
    --instances data/fininteract_v1.jsonl \
    --models gpt-5 gpt-4o \
    --modes answer-only answer+search answer+search+interact \
    --elicit-confidence \
    --out results/eval_new.jsonl
```

Cross-vendor / open-weight models are run through OpenRouter:

```bash
python scripts/evaluate.py \
    --instances data/fininteract_v1.jsonl \
    --models anthropic/claude-sonnet-5 qwen/qwen3-30b-a3b \
    --modes answer+search+interact \
    --openrouter-config configs/openrouter.json \
    --out results/eval_crossvendor.jsonl
```

**2. Regenerate tables and figures** from the provided raw outputs in `results/`
(no re-running needed):

```bash
python scripts/analyze_results.py --results results/eval_gpt5_gpt4o.jsonl   # main breakdowns
python scripts/report_exp12.py                                             # 2x2 oracle (elicitation vs grounding)
python scripts/dise.py --results results/<file>.jsonl                      # DisE+ efficiency metric
python scripts/plot_findings.py                                            # headline figures
python scripts/plot_error_distribution.py                                  # error taxonomy
python scripts/plot_capability_radar.py                                    # per-category capability radar
```

The `results/` directory contains the raw per-instance outputs for every model reported
in the paper (OpenAI GPT-5 / GPT-4o / GPT-5-mini, the Qwen open-weight family, and the
cross-vendor panel), plus the oracle, human-baseline, simulator-robustness, and
category-aware ReAct ablations.

## Reconstructing the benchmark from source (optional)

The full construction pipeline is included for transparency. It pulls public filings,
builds a passage pool, and constructs instances through a constructor -> quality-control
-> adversarial-verifier loop. See `scripts/README.md` for the step-by-step commands
(`pull_edgar.py`, `pull_cninfo.py`, `extract_edgar_passages.py`, `construct_instances.py`).
Third-party datasets used only during construction are not redistributed here; obtain
them from their original public sources.

## Notes on reproducibility

- The user simulator runs an LLM at temperature 1.0 by default; a deterministic option and
  a free-form-answer option are available via `--user-sim`. Reported robustness checks
  (deterministic vs. stochastic vs. noisy simulators) are in `results/eval_simrobust_*`.
- Numeric grading uses an LLM grader with a +/-1% tolerance; grader-vs-human agreement is
  reported in the paper.
- Randomness in agent runs comes from the backbone models; set decoding temperature to 0
  where a provider supports it for closer reproduction.
