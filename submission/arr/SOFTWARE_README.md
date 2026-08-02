# FinInteract — Software

Code accompanying the submission *FinInteract: Benchmarking Clarification and Intent
Integration in Ambiguous Financial Question Answering*. This archive holds the
construction, evaluation, analysis, and plotting pipeline.

**The benchmark itself and the raw evaluation outputs are in the companion Data
archive**, uploaded to this submission's Data field. Unpack both side by side so that
`data/` sits next to `scripts/`, which is what every path below assumes:

```
<workdir>/
  scripts/      configs/    requirements.txt     <- this archive
  data/final/   data/results/                    <- Data archive
```

This material is anonymized for double-blind review: it contains no author,
institution, or repository-identifying information, and no API keys.

## Layout

```
README.md                          This file
requirements.txt                   Python dependencies
configs/openrouter.example.json    API config template (copy to openrouter.json, add your key)
scripts/                           Construction, evaluation, analysis, and plotting code
scripts/README.md                  Per-script index
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10+ recommended
pip install -r requirements.txt

# Native OpenAI models read OPENAI_API_KEY from the environment.
# Cross-vendor models route through OpenRouter via a small JSON config:
cp configs/openrouter.example.json configs/openrouter.json
#   then edit configs/openrouter.json and insert your own OpenRouter key.
```

## Reproducing the main results

**1. Evaluate a model on the benchmark** (three interaction modes):

```bash
export OPENAI_API_KEY=...
python scripts/evaluate.py \
    --instances data/final/fininteract_v1.jsonl \
    --models gpt-5 gpt-4o \
    --modes answer-only answer+search answer+search+interact \
    --elicit-confidence \
    --out data/results/eval_new.jsonl
```

Cross-vendor and open-weight models are run through OpenRouter:

```bash
python scripts/evaluate.py \
    --instances data/final/fininteract_v1.jsonl \
    --models anthropic/claude-sonnet-5 qwen/qwen3-30b-a3b \
    --modes answer+search+interact \
    --openrouter-config configs/openrouter.json \
    --out data/results/eval_crossvendor.jsonl
```

**2. Regenerate tables and figures** from the raw outputs shipped in the Data archive
(no model calls needed):

```bash
python scripts/analyze_results.py --results data/results/eval_gpt5_gpt4o.jsonl
python scripts/dise.py --results data/results/<file>.jsonl
python scripts/plot_findings.py
python scripts/plot_error_distribution.py
python scripts/plot_capability_radar.py
python scripts/plot_axis_performance.py
```

## Reconstructing the benchmark from source (optional)

The full construction pipeline is included for transparency. It pulls public filings,
builds a passage pool, and constructs instances through a constructor -> quality-control
-> adversarial-verifier loop. Third-party datasets used only during construction are not
redistributed here; obtain them from their original public sources.

The EDGAR pull scripts require a contact email in the SEC User-Agent header, per SEC
policy. The shipped placeholder is `your.email@school.edu`; replace it with your own
before running them.

## Notes on reproducibility

- The user simulator runs an LLM at temperature 1.0 by default; a deterministic option
  and a free-form-answer option are available via `--user-sim`. Robustness checks
  (deterministic vs. stochastic vs. noisy simulators) are in `data/results/eval_simrobust_*`.
- Numeric grading uses an LLM grader with a +/-1% tolerance; grader-vs-human agreement is
  reported in the paper.
- Randomness in agent runs comes from the backbone models; set decoding temperature to 0
  where a provider supports it for closer reproduction.
