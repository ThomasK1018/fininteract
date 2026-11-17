# FinInteract v1 — Dataset Card

**File:** `fininteract_v1.jsonl` · **Frozen:** 2026-06-06 · **Instances:** 173

A bilingual (EN/ZH) benchmark of *ambiguous financial queries* built on the
"Easy to verify, Ambiguous to resolve" principle. Each instance pairs a question with a
**default** interpretation (what a non-expert assumes) and an **intended** interpretation
(fixed by a disambiguating context), both grounded in verbatim filing spans.

## Profile

| Property | Value |
|----------|-------|
| Total instances | **173** frozen evaluation records; v1.1 excludes one consensus-rejected record |
| Language | 53 EN (31%) / 120 ZH (69%) |
| Company metadata | 60 distinct nonempty labels; three DocFinQA records lack a company label. A normalized company count has not been established. |
| Sources | CNINFO/akshare 120, EDGAR 43, EDGAR-recognition 7, DocFinQA 3 |
| Filing dates | 2022 (40), 2023 (53), 2024 (58), 2025 (19); three undated DocFinQA records include older periods |
| Primary axis | entity_scope 94 (54%), metric_definition 63 (36%), recognition_policy 9 (5%), temporal_scope 7 (4%), filing_vintage 0 |
| n-axes | single 128 (74%) / two 45 (26%) |
| Mean H₀ | 2.20 bits |
| **Blind-solve rate** | **0.3%** (adversarial 10-trial verifier) |

## Construction checks and their limits

- The construction pipeline applies 14 checks. Stored final QC objects record acceptance and verifier counts, not a complete per-rule audit trail. Human adjudication subsequently rejected one record, removed in v1.1.
- The answer-separation check aims to distinguish intended and default values under the grader tolerance; it does not establish that both readings are equally plausible to users.
- Stored blind-verifier counts total 6 successful trials out of 1,730 (0.347%, rounded to 0.3%). This is a construction diagnostic, not evidence that all evaluation performance is caused by interaction.
- Most dated filings are from 2022–2025. Three DocFinQA-derived records have no filing date and include older financial periods. No contamination-free guarantee is made.

## Known scope limits (report honestly)

- **Axis skew is intrinsic, not a sampling failure.** entity_scope and metric_definition
  dominate because they are the axes with abundant *structured* financial ambiguity. The rare
  axes hit hard data-availability ceilings: recognition_policy exists only in mixed-revenue-model
  firms (~3–6 companies), filing_vintage needs rare restatement events with complete XBRL, and
  temporal_scope FY/CY ambiguity applies only to non-calendar-FY filers. See `dataset_analysis.md`.
- **ZH is a deep net-income probe**, not broad ZH coverage: ~99% of ZH instances concern 净利润
  (net income) scope/definition variants.
- **Single-axis dominant (74%)** — multi-axis (≥3) compositional instances were not reliably
  extractable from structured data; a future LLM-assisted multi-axis tier is possible.

## Schema (per line)

`instance_id, passage_id, language, source, ticker, company, filing_type, filing_date,
question, context, answer, default_answer, intended_evidence_span, default_evidence_span,
intended_interpretation{entity,period,metric,basis}, default_interpretation{...},
shared_attributes, distinctive_attributes, axes, n_axes, h0, qc{...}`

## Companion files

- `fininteract_v1.quarantine.jsonl` — 2 instances dropped for sub-QC scores (recoverable).
- GRPO/RLVR training kit (separate): stratified train/test split, paraphrase-augmented
  training data, and the paraphrase-consistency robustness probe.

## Provenance notes

- Built via a 3-role pipeline: GPT-5 constructor → 14-rule QC → adversarial verifier
  (5× GPT-5-mini + 5× GPT-5, two-condition rejection). OpenAI API only.
- IDs are clean and sequential (`fininteract_0001`..`fininteract_0173`); an earlier
  resume-counter collision was deduplicated at freeze time.

## License

Original benchmark records, annotations, and metadata: CC BY 4.0; see `LICENSE.txt`. Source filings and other third-party materials retain their original rights and terms.
