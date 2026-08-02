# FinInteract — Data

Data accompanying the submission *FinInteract: Benchmarking Clarification and Intent
Integration in Ambiguous Financial Question Answering*. This archive holds the frozen
benchmark and the raw per-model evaluation outputs behind every table and figure in the
paper.

**The code that produces and consumes these files is in the companion Software
archive**, uploaded to this submission's Software field. Unpack both side by side so
that `data/` sits next to `scripts/`:

```
<workdir>/
  scripts/      configs/    requirements.txt     <- Software archive
  data/final/   data/results/                    <- this archive
```

This material is anonymized for double-blind review: it contains no author,
institution, or repository-identifying information, and no API keys.

## Layout

```
README.md                          This file
data/final/
  fininteract_v1.jsonl             Frozen benchmark used for the experiments (N = 173)
  fininteract_v1.1.jsonl           Public release (N = 172; one consensus-rejected item quarantined)
  fininteract_v1.quarantine.jsonl  The quarantined item
  DATASET_CARD.md                  Datasheet: provenance, schema, intended use, licensing
  CHANGELOG.md                     Version history (v1 -> v1.1)
data/results/                      Raw per-model evaluation outputs (JSONL/JSON)
```

Use `fininteract_v1.jsonl` to reproduce the numbers in the paper; it is the exact
frozen set the experiments were run on. Use `fininteract_v1.1.jsonl` for new work.

## Schema

Each line of `data/final/fininteract_v1.jsonl` is one instance with fields including:
`instance_id`, `language`, `source`, `ticker`, `company`, `filing_type`, `filing_date`,
`question`, `context` (the disambiguating context C), `answer` (intended, exactly
verifiable), `default_answer`, `intended_evidence_span`, `default_evidence_span`,
`intended_interpretation` / `default_interpretation` (each an {entity, period, metric,
basis} tuple), `axes` (the ambiguity categories touched), `n_axes`, `h0`
(log2 of the plausible-interpretation count), and `qc` (per-rule pass flags plus the
verifier blind-solve rate). See `data/final/DATASET_CARD.md` for the complete datasheet.

## Evaluation outputs

`data/results/` contains the raw per-instance outputs for every model reported in the
paper (OpenAI GPT-5 / GPT-4o / GPT-5-mini, the Qwen open-weight family, and the
cross-vendor panel), plus the oracle, human-baseline, simulator-robustness, per-category,
co-evolution, and ablation runs. The analysis and plotting scripts in the Software
archive read directly from this directory, so every table and figure in the paper can be
regenerated without issuing a single model call.

## Provenance and licensing

English instances derive from public SEC EDGAR filings (10-K, FY2022--2025) and a small
DocFinQA-sourced set; Chinese instances derive from public CSRC annual reports retrieved
via CNINFO/akshare. All source documents are public regulatory filings; no proprietary,
paywalled, or personal data is used. Third-party corpora used only during construction
are not redistributed here; obtain them from their original public sources.

Intended use: evaluating whether interactive search agents recognize and resolve query
ambiguity. Not a training corpus, and not financial advice.
