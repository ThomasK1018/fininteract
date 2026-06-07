# Data-loading and construction scripts

Full pipeline for pulling source filings, building a passage pool, and constructing
FinInteract benchmark instances via LLM-assisted generation.

## One-time setup

```bash
conda activate finteract
pip install datasets edgartools openai tqdm

# SEC requires identification — no API key needed
export EDGAR_IDENTITY="Your Name your.email@school.edu"
export OPENAI_API_KEY=...
```

---

## Step 1 — Pull source data

### DocFinQA (Kensho, ACL 2024)
123k-word 10-K contexts with gold QA pairs — backbone of the EN passage pool.

```bash
# Full validation split (~780 records)
python scripts/load_docfinqa.py --split validation
# Output: data/docfinqa/docfinqa_validation.jsonl
```

### EDGAR direct (2024-2025 10-Ks)
Contamination-safe filings beyond FinanceBench/FinQA.

```bash
# Smoke test first
python scripts/pull_edgar.py \
    --targets data/edgar/target_companies.txt \
    --forms 10-K --years 2024 2025 \
    --out data/edgar/edgar_filings_smoke.jsonl \
    --limit-tickers 2

# Full pull
python scripts/pull_edgar.py \
    --targets data/edgar/target_companies.txt \
    --forms 10-K --years 2024 2025 \
    --out data/edgar/edgar_filings.jsonl
# Output: data/edgar/edgar_filings.jsonl (Item 7 MD&A + Item 8 + XBRL facts)
```

### FinanceBench, FinQA (download manually)
```
data/financebench/   — place FinanceBench open-source JSONL here
data/finqa/FinQA/dataset/train.json + dev.json
```

### ZH sources (FinEval, CFLUE)
```
FinEval/   — place FinEval repo root here (relative to project root)
cflue/     — place CFLUE repo root here
```

---

## Step 2 — Extract EDGAR XBRL passages

Derives verified QA pairs from EDGAR XBRL facts for entity_scope, temporal_scope,
and metric_definition instances. Must be run after `pull_edgar.py`.

```bash
python scripts/extract_edgar_passages.py
# Output: data/sources/edgar_passages.jsonl (~3 passages per filing)
```

---

## Step 3 — Build passage pool

Merges all sources into a single pool. Reads `edgar_passages.jsonl` automatically
if present (preferred over raw EDGAR filings, which lack candidate answers).

```bash
python scripts/prepare_passages.py --out data/sources/passages.jsonl
# Output: data/sources/passages.jsonl
```

Axis distribution check (should see all 5 axes represented):
```bash
python3 -c "
import json; from collections import Counter
c=Counter()
[c.update(json.loads(l).get('candidate_axes',[])) for l in open('data/sources/passages.jsonl')]
print(dict(c))
"
```

---

## Step 4 — Construct instances

LLM-assisted construction with 8-rule constructor prompt, adversarial QC verifier,
and axis-diversity enforcement. OpenAI API only.

```bash
# Dry run (no API calls)
python scripts/construct_instances.py --limit 10 --dry-run

# Pilot (20-30 instances for QC review)
python scripts/construct_instances.py \
    --source data/sources/passages.jsonl \
    --out data/constructed/pilot.jsonl \
    --limit 100 --target 25

# Full run (target 500 accepted instances)
python scripts/construct_instances.py \
    --source data/sources/passages.jsonl \
    --out data/constructed/instances.jsonl \
    --target 500
```

Rejected instances are written to `*.rejected.jsonl`.
Target verifier rejection rate: 20-40%. If <10%, questions are too easy.

---

## Step 5 — Score pilot quality

Rule-based quality check (10 rules, no API calls). Run after each pilot batch.

```bash
python scripts/score_pilot.py data/constructed/pilot.jsonl
```

**AAAI acceptance bar:** ≥90% of instances should achieve 100% rule-pass score.
All 5 axes should each have ≥10% share.

Remaining issues after automated scoring always need human spot-check:
- Plausible default interpretation?
- Non-obvious intended interpretation?
- Context C specific enough to actually disambiguate?
- Answer A numerically correct?

---

## Construction pipeline rules (8 hard rules)

The constructor prompt enforces:

**Q rules:**
1. Not a yes/no question
2. No disambiguating terms, dates, or segment names in Q — only shared attributes
3. Use candidate_answer verbatim as A
4. Q must name the company
5. Q must ask about a financial metric (not page count, filing fee, etc.)

**A rules:**
6. Type-consistent with Q (count Q → integer A, rate Q → % A)

**C rules:**
7. C must not contain the answer value
8. C must describe the INTENDED interpretation (not the default)

**Code-level guards:**
- Skip passages with empty candidate_answer
- Reject answers with implausible % values (>500% or 0%)
- Filter axes_exercised to valid taxonomy names only

---

## DisE (Disambiguation Efficiency) metric

```python
from dise import compute_dise, aggregate_dise

# Per instance
result = compute_dise(
    axes_hit=["temporal_scope", "entity_scope"],
    n_asks=3,
)
print(result.dise)   # e.g. 0.86 (2.585 bits / 3 asks)

# Aggregate
agg = aggregate_dise(all_results)
print(agg)  # {"mean_dise": 0.45, "by_complexity": {1: 0.72, 2: 0.51, 3: 0.31}}
```

---

## Sanity checks

```bash
wc -l data/docfinqa/docfinqa_validation.jsonl   # expect ~780
wc -l data/edgar/edgar_filings.jsonl            # expect ~43
wc -l data/sources/edgar_passages.jsonl         # expect ~111
wc -l data/sources/passages.jsonl               # expect ~934+

# Inspect one EDGAR XBRL passage
python3 -c "
import json
r=json.loads(open('data/sources/edgar_passages.jsonl').readline())
print(r['passage_id'], r['company'], r['candidate_answer'], r['candidate_axes'])
"

# Inspect one constructed instance
python3 -c "
import json
r=json.loads(open('data/constructed/pilot.jsonl').readline())
print(json.dumps(r, indent=2)[:1500])
"
```
