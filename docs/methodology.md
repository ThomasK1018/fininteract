# Methodology

## Overview

FinInteract is constructed around the principle **"Easy to verify, Ambiguous to resolve"** (adapted from InteractComp). Each instance consists of a question $Q$, a disambiguating context $C$, and a verified answer $A$, such that:

- $Q$ is answerable in at least two plausible ways from publicly available financial disclosures.
- $A$ is the unique correct answer under the *intended* interpretation.
- $C$ is the minimal set of scope/definitional constraints that collapses $Q$ to $A$.
- An agent reading $Q$ alone (without $C$) must find it genuinely ambiguous — i.e., the *default* interpretation leads to a different, also defensible answer.

This *default-vs-intended interpretation pairing* is our generalization of InteractComp's target-distractor design to the financial domain.

---

## Financial Ambiguity Taxonomy

We define five axes of financial ambiguity, each grounded in standard accounting and disclosure practice:

| Axis | Definition | Example |
|------|-----------|---------|
| **Temporal scope** | Ambiguity over which time period is intended | Fiscal year ending September vs. calendar year; Q3 results vs. full year; TTM vs. point-in-time |
| **Metric definition / Reporting basis** | Ambiguity over which accounting basis applies | GAAP net income vs. non-GAAP adjusted income; organic revenue vs. as-reported; operating EBITDA vs. net income |
| **Entity scope** | Ambiguity over which entity or consolidation level is meant | Segment vs. consolidated total; parent-attributable vs. full consolidated; Class A vs. Class B shares; A-shares vs. H-shares |
| **Filing vintage / Disclosure source** | Ambiguity over which filing version or disclosure event to cite | Original 10-K vs. amended 10-K/A; as-reported vs. restated; preliminary earnings release vs. final filing |
| **Recognition / Accounting policy** | Ambiguity over which accounting policy was applied | Revenue recognized point-in-time vs. over-time; gross vs. net revenue; capitalized vs. expensed R&D; impairment timing |

Each instance exercises one primary axis (mandatory) and optionally one or two secondary axes where the passage genuinely supports additional ambiguity. The target distribution over primary axes is 25% temporal scope, 25% metric definition, 20% recognition policy, 15% entity scope, and 15% filing vintage — calibrated to counterbalance the over-representation of entity scope (71% of pool passages) and filing vintage (82%) in the raw data, and to ensure sufficient coverage of the harder, rarer axes.

### Disambiguation Entropy ($H_0$)

For each instance we compute the *prior disambiguation entropy*:
$$H_0 = \log_2 k$$
where $k$ is the number of plausible interpretations before any clarifying interaction. For a single-axis instance with two interpretations, $H_0 = 1$ bit. For a two-axis instance with three plausible values each, $H_0 = \log_2 9 \approx 3.17$ bits. This quantity anchors our **DisE** metric (§ Evaluation Framework).

---

## Dataset Construction Pipeline

### Data Sources

**English (EN, ~80% of pool):**
- *DocFinQA* — 780 long-context passages extracted from 10-K and 10-Q filings; provides verified candidate answers from annotated QA pairs.
- *EDGAR (XBRL-derived)* — 111 passages built from machine-readable XBRL facts for S&P 500 + Russell 1000 companies (FY2024–2025); candidate answers are drawn directly from XBRL values. Because instances are newly constructed from structured facts that require resolving freshly generated ambiguity pairs, memorization risk is substantially reduced relative to benchmarks that reuse published QA pairs directly — though we do not claim absolute contamination safety, as models may have broad web exposure.

**Chinese (ZH, ~20% of pool):**
- *CNINFO/akshare* — 237 passages derived from A-share annual reports (上证50 + 沪深300 constituents) via structured financial statement APIs. Three passage types: (1) metric definition — 扣非净利润 vs. 归母净利润; (2) entity scope — 归母净利润 vs. 净利润 (parent-attributable vs. full consolidated including minority interest); (3) temporal scope — year-over-year revenue growth.

The combined passage pool contains 1,128 passages (891 EN, 237 ZH). Each passage carries a `candidate_answer` (verified ground truth) and `candidate_axes` (axes the passage is likely to support), inferred from keyword signals and XBRL metadata.

### Three-Role Construction Pipeline

Instance construction follows a three-role pipeline. Each passage attempt costs approximately 10–14 minutes of wall time, dominated by two sequential GPT-5 calls — one for construction and one as the slowest concurrent verifier trial.

**Step 1 — Constructor (GPT-5, ~2–5 min).** A single GPT-5 call with a 4,096-token budget generates the full $(Q, C, A)$ triple given a filing passage, verified candidate answer, and mandatory primary ambiguity axis. The output JSON includes:

```
question, context, answer
default_answer           ← concrete value under the default interpretation
intended_evidence_span   ← verbatim passage quote (≤ 60 words) supporting A
default_evidence_span    ← verbatim passage quote supporting default_answer
intended_interpretation: {entity, period, metric, basis}
default_interpretation:  {entity, period, metric, basis}
axes_exercised: [primary_axis, ...]
```

The constructor prompt enforces eight hard rules: no yes/no questions, no disambiguating terms in $Q$, company must be named in $Q$, answer must be a substantive financial metric, context must not contain the answer value, and the intended interpretation must differ from the default on at least one axis. Axis-specific instructions are injected per primary axis to prevent the constructor from drifting toward whichever axis is easiest to construct. If GPT-5 returns an empty or malformed response, the call retries up to three times with a two-second backoff.

**Step 2 — Pre-verifier sanity checks (~instant).** Ten automated code-level rules are applied without any API calls (detailed in § Quality Control below). Any failure immediately rejects the passage and moves to the next one, avoiding wasted verifier budget.

**Step 3 — Adversarial Verifier (GPT-5-mini + GPT-5, 5 rounds each, parallelized, ~5–10 min).** Ten verifier calls fire simultaneously via a thread pool: five GPT-5-mini trials and five GPT-5 trials. Each trial asks the model to answer $Q$ *without* context $C$ and to also state the interpretation it assumed (period, entity, metric, basis). An instance is **rejected** if $\geq 2$ of the 10 trials both (a) produce the correct answer $A$ and (b) state an assumed interpretation that aligns with the intended interpretation. This two-condition rejection criterion prevents false rejections from lucky guesses: if a model produces the right number while assuming a different interpretation (e.g., the default), the coincidence is not counted against the instance. Because all ten calls run concurrently, wall time equals the latency of the slowest single GPT-5 call.

**Role 3 — Human validation.** Each accepted instance is independently annotated by two human annotators (the primary author and a domain-aware colleague) using an eight-question protocol (see § Human Validation Protocol below). Disagreements on any question are adjudicated by the primary author with reference to the source passage. Systematic failure patterns observed during annotation are fed back as prompt improvements to the constructor.

### Axis Diversity Enforcement

A naive construction loop over-represents axes that are common in the passage pool (entity scope appears in 66% of passages; filing vintage in 67%). We enforce target axis shares through:

1. **Priority function with 3× penalty:** passages whose primary axis is over-quota receive a multiplicative penalty of 3× the shortfall, ensuring they drop to the back of the processing queue.
2. **Hard exclusion at 2× quota:** passages whose only available axes are all at twice the target share are excluded from consideration until the distribution corrects.
3. **Rarity-adjusted initial sort:** passages are sorted upfront by $\sum_{\text{axes}} (\text{target\_share} / \text{pool\_frequency})$, so rare-but-needed axes (metric definition at 37% pool frequency vs. 30% target) are processed first.
4. **Dynamic re-sort every 10 instances:** the remaining queue is re-ranked periodically to adapt to the evolving axis distribution.

### Quality Control (Pre-Verifier Sanity Checks)

Before spending adversarial verifier budget, each generated instance passes through fourteen automated checks applied without any API calls:

| Check | Rule |
|-------|------|
| R1 | $Q$ must not be answerable with yes/no |
| R2 | $Q$ must not contain disambiguating terms (fiscal year, non-GAAP, consolidated, amended, etc.) or inline dates/years |
| R4 | $Q$ must name the company (capitalized English proper noun or CJK characters for ZH) |
| R5 | $Q$ must ask about a substantive financial metric, not administrative content |
| R6 | Answer type must be consistent with question type (count $Q$ ↛ percentage $A$) |
| R7 | Context $C$ must not contain the answer value |
| R8 | Intended and default interpretations must differ on at least one axis |
| R9 | All axis labels must be valid vocabulary items |
| R10 | Answer must be non-trivial (not empty, bare punctuation, or "0") |
| R11 | `default_answer` must be present and must differ from `answer` (ensures two distinct defensible values exist) |
| R12 | Both `intended_evidence_span` and `default_evidence_span` must be non-empty (ensures each interpretation is grounded in a verbatim passage quote) |
| R13 | Evidence spans must be meaningfully distinct: not identical, and not more than 85% prefix-overlapping (prevents constructions where both interpretations are grounded in the same passage excerpt) |
| R14 | Intended and default answers must differ by **more than the grader's tolerance**: $> 1.5$ percentage points for rate answers, or $> 3\%$ relative for value answers. This rejects *non-discriminating* instances where a model resolving to the wrong interpretation would still be graded correct. |

R14 is the most consequential post-pilot addition. Empirical audit of the first 180 instances (see `dataset_analysis.md`) found that growth-rate framings on the metric-definition axis sometimes collapse the intended and default answers to within grader tolerance (e.g., GAAP net-income growth vs. GAAP EPS growth often differ by only tenths of a percentage point). Without R14 such instances silently inflate measured accuracy, because answering the *default* (wrong) interpretation grades as correct. R14 enforces that every instance is genuinely discriminating at construction time.

Additional pre-verifier rejections: implausible percentage answers ($> 500\%$), EPS questions with percentage answers (unless asking for a growth rate), and primary axis mismatch (constructor ignored the mandatory primary axis).

A DocFinQA-specific pre-scan further skips passages whose primary axis is `filing_vintage`, `recognition_policy`, or `temporal_scope` (DocFinQA passages essentially never support these without paired structured values), and requires at least two distinct numeric values in the passage. EDGAR temporal yoy passages from calendar-fiscal-year companies are skipped because they carry no genuine fiscal-vs-calendar ambiguity. These gates conserve constructor budget for passages that can actually produce a valid instance.

---

## Human Validation Protocol

Each accepted instance is reviewed by two annotators who answer eight yes/no questions:

| # | Question | Validates |
|---|----------|-----------|
| H1 | Is $Q$ ambiguous without $C$ — i.e., does it have at least two plausible, distinct interpretations? | Core ambiguity claim |
| H2 | Is the default interpretation (yielding `default_answer`) plausible for a non-expert reader? | Default answer defensibility |
| H3 | Does $C$ uniquely identify the intended interpretation? | Context sufficiency |
| H4 | Does $C$ avoid directly stating the answer value? | Rule R7 |
| H5 | Is the intended answer $A$ correct under the intended interpretation? | Answer correctness |
| H6 | Is `default_answer` correct under the default interpretation? | Default answer correctness |
| H7 | Is the primary axis label correct? | Taxonomy validity |
| H8 | What yes/no clarifying question would a financial analyst naturally ask to resolve the ambiguity? (free text) | Ambiguity quality check |

**Reported statistics:** ambiguity validity rate (H1), default plausibility rate (H2), context sufficiency rate (H3), answer correctness rate (H5), axis-label agreement (H7), inter-annotator Cohen's $\kappa$ across H1–H7, and post-adjudication acceptance rate. H8 responses are used to compute human AxisHit (how often the analyst's natural question matches the gold axis).

---

## Evaluation Framework

### Agent Architecture

We evaluate agents under the ReAct framework in seven configurations:

**Primary evaluation modes:**

| Mode | Actions available | Description |
|------|------------------|-------------|
| **Answer-only** | answer | Pure parametric recall; no retrieval or interaction |
| **Answer+Search** | search, answer | Oracle retrieval augmented; tests if context access alone resolves ambiguity |
| **Answer+Search+Interact** | search, interact, answer | Full interaction; agent may ask yes/no/IDK questions to elicit $C$ |

The *interact* action presents a yes/no question to a simulated user. The user simulator (GPT-5 at temperature 1.0) is given the disambiguating context $C$ and responds yes, no, or "I don't know."

**Baseline and ablation modes:**

| Mode | Description | Purpose |
|------|-------------|---------|
| **Always-ask** | Agent must ask exactly one clarification before answering | Tests whether gains come from strategic recognition vs. any interaction |
| **Axis-oracle** | Agent is told the primary ambiguity axis but not the intended value | Upper bound on axis recognition; isolates the "what to ask" capability |
| **Template-oracle** | Fixed human-written question per axis; agent answers after that exchange | Tests whether models fail because they cannot identify what to ask, or because they cannot answer after receiving clarification |
| **Enumerate** | Agent lists all plausible interpretations and answers under each | Direct comparison with the enumeration paradigm rejected in § Motivation |

### Models

Eight models are evaluated across all three modes:

| Family | Models |
|--------|--------|
| OpenAI | GPT-5, GPT-5-mini, GPT-4o |
| Anthropic | Claude-Sonnet-4, Claude-Opus-4 |
| Open-weight | DeepSeek-V3.1, Qwen3-235B, GLM-4.5 |

### Metrics

**Primary:**
- **Accuracy (%)** — fraction of instances answered correctly, graded by GPT-4o-mini with financial-domain tolerance (±1% numeric, entity/ticker equivalence, currency normalization; fiscal-year mismatch = wrong).
- **DisE$^+_\text{all}$ (primary)** — $\mathbf{1}[\text{correct}] \times H_0 / \max(n_{\text{asks}}, 1)$. Zero when incorrect; rewards confident correct zero-ask answers ($n_{\text{asks}} = 0$ gives $H_0$) as well as efficient clarification. This is the primary reported metric.
- **DisE$^+_\text{interact}$ (secondary)** — $\mathbf{1}[\text{correct}] \times \mathbf{1}[n_{\text{asks}} > 0] \times H_0 / n_{\text{asks}}$. Only defined when the agent asked at least one question; specifically rewards successful clarification and is zero for zero-ask correct answers. Reported alongside DisE$^+_\text{all}$ to separately characterize interaction behavior.
- The original DisE $= H_0 / n_{\text{asks}}$ (no correctness gate) is retained as a reference metric in ablation comparisons.

**Secondary:**
- **AxisHit** — for each interact action, GPT-4o-mini classifies whether the clarifying question targets a true ambiguity axis. Each question is assigned one of three labels: **hit** (targets a correct axis), **generic** (vague clarification with no specific axis, e.g. "Can you clarify the question?"), or **wrong_axis** (targets a specific but incorrect axis). Four sub-metrics are reported:
  - **AxisHit@1** — first clarifying question is on-axis (most actionable; agent recognized ambiguity immediately)
  - **AnyAxisHit** — at least one question across the interaction is on-axis (per instance)
  - **WrongAxisRate** — fraction of interact actions targeting a specific but incorrect axis (indicates confident misdirection)
  - **GenericAskRate** — fraction of interact actions that are axis-free generic clarifications (indicates low ambiguity awareness)
  - A model with high AxisHit@1 but low accuracy confirms that interaction quality is decoupled from grounding ability; a model with high WrongAxisRate is confidently wrong about the source of ambiguity.
- **Round** — average number of conversation turns per instance.
- **IR (Interaction Rate)** — fraction of instances where the agent used at least one interact action.
- **Calibration Error** — 5-bin expected calibration error on stated confidence, following the sampling-based estimator of Cole et al. (EMNLP 2023).

### User Simulator Variants

To test robustness to the simulation assumption, each model is evaluated under three user simulator configurations:

| Simulator | Description |
|-----------|-------------|
| **LLM** (primary) | GPT-5 at temperature=1.0; answers yes/no/IDK based on context $C$ |
| **Oracle** | Deterministic rule-based; answers based on keyword matching against intended interpretation fields; no LLM call |
| **Noisy** | GPT-5 simulator with 15% random answer corruption; simulates users who occasionally misremember or misstate their intent |

The LLM simulator is the primary evaluation condition. Oracle provides an upper bound on interaction benefit (perfect user responses). Noisy tests robustness when users are unreliable.

### Ablation: Forced-Interaction Analysis

Following InteractComp, we run a forced-interaction ablation: the agent is required to ask $n \in \{2, 4, 6, 8, 10\}$ clarifying questions before being permitted to answer. This reveals the *latent capacity* of each model — the accuracy achievable when it cannot skip interaction — and distinguishes overconfidence (high skip rate, low accuracy) from genuine uncertainty (low skip rate, high accuracy).

### Per-Axis Breakdown

Accuracy is stratified by each of the five ambiguity axes. This identifies which types of financial ambiguity are hardest for current models. Based on InteractComp precedent and domain analysis, we expect metric definition and temporal scope to be the hardest axes (requiring domain knowledge to even recognize the ambiguity), while entity scope may be easier for models with strong SEC filing exposure.

---

## Bilingual Design

Approximately 80% of instances are in English (sourced from SEC EDGAR and DocFinQA) and 20% are in Chinese (sourced from A-share annual reports via CNINFO/akshare). Chinese instances exercise entity scope (归母净利润 vs. 净利润 disambiguation), metric definition (扣非净利润 vs. 归母净利润), and temporal scope (fiscal year revenue growth rate). Chinese questions are generated in Mandarin with Chinese company names, requiring models to handle cross-lingual financial terminology.

ZH instances are expected to be harder than EN instances due to sparser model training data for A-share corporate filings and more complex Chinese corporate structure nomenclature (e.g., H/A/B-share distinctions, 集团 vs. 子公司 scope).

---

## Dataset Statistics (Target)

| Property | Value |
|----------|-------|
| Total instances | 200–500 (build-dependent) |
| Language split | ~80% EN / ~20% ZH (final; transiently ZH-heavy during construction) |
| Single-axis instances | ~70% |
| Two-axis instances | ~30% |
| Three-plus-axis instances | stretch goal (rare; requires passages supporting ≥3 independent verifiable ambiguities) |
| Data sources | DocFinQA, EDGAR XBRL, EDGAR 10-K/A amendments, CNINFO/akshare |
| Verification | 14-rule pre-verifier QC + adversarial verifier + human spot-check |
| Human baseline | 50 instances (user + 1 colleague) |

The single/two/three-plus-axis split is revised from an earlier 30/50/20 target. Empirical
construction (see `dataset_analysis.md`) shows multi-axis instances are scarce: a passage must
simultaneously support two or more *independent, individually verifiable* ambiguities, which is
uncommon in real filings. The realized distribution is approximately 70% single-axis / 30%
two-axis, with three-plus-axis instances treated as an opportunistic stretch rather than a quota.
Single-axis dominance does not weaken the benchmark — each single-axis instance still requires the
agent to identify *which* of the five axes is under-specified, which is the core capability under
test.
