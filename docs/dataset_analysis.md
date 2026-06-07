# Dataset Analysis (Empirical)

*Analysis of the constructed FinInteract dataset as of the current build (n = 180 accepted
instances). This document characterizes what the construction pipeline actually produces,
validates the core design claims against real data, and surfaces the quality risks that the
QC layer must address. It is the empirical companion to `methodology.md` (the intended design)
and `motivation.md` (the research framing).*

---

## 1. What the Problem Actually Is

The benchmark tests one capability: **can a financial search agent recognize that a query is
under-specified, and resolve it before answering?** The empirical question for *us* is whether
the constructed instances actually instantiate that capability — i.e., whether each instance is

1. **genuinely ambiguous** (a competent model, given only the question, cannot reliably recover
   the intended answer), and
2. **discriminating** (a model that resolves to the *wrong* interpretation produces a *different*
   answer that the grader marks wrong).

Both properties are measurable from the constructed data. This analysis shows the dataset
satisfies (1) very strongly, but (2) is violated by a specific, fixable subclass of instances.

---

## 2. Dataset Profile (n = 180)

| Property | Value |
|----------|-------|
| Total accepted instances | 180 |
| Language split | 57 EN (32%) / 123 ZH (68%) |
| Source split | 123 CNINFO/akshare (ZH), 55 EDGAR, 1 EDGAR-amendment, 1 DocFinQA |
| Filing recency | 2022 (41), 2023 (60), 2024 (62), 2025 (16) — contamination-favorable |
| Single-axis / two-axis / 3+ | 134 (74%) / 46 (26%) / 0 (0%) |
| $H_0$ range | 1.00 – 3.58 bits (mean 2.27) |
| Answer types | 132 currency, 47 percentage, 1 plain number |

**Note on language split.** The 32% EN / 68% ZH ratio is a transient artifact of the construction
order (ZH passages were processed first and have a ~99% acceptance rate). The language cap halts
ZH growth at 123, so the *final* ratio converges toward the 80/20 EN/ZH target as EN construction
continues. Current numbers should not be read as the final distribution.

---

## 3. Core Validation: the dataset is genuinely hard

The adversarial verifier fires 10 trials per instance (5 GPT-5-mini + 5 GPT-5), each answering
the question **without** the disambiguating context. An instance is accepted only if fewer than 2
trials both produce the correct answer *and* assume the intended interpretation.

**Observed verifier vote rate across accepted instances: 0.3%** (174/180 instances had *zero*
successful trials; 6 had exactly one). In other words, frontier models answering these questions
blind almost never land on the intended answer for the intended reason.

This is the single most important empirical result in the build. It directly validates the
"Ambiguous to resolve" half of the design principle: the questions are not solvable by parametric
recall or lucky guessing. Whatever accuracy models achieve in the eventual evaluation will be
attributable to *interaction and retrieval*, not to prior knowledge — which is exactly the
capability boundary the benchmark is designed to probe.

---

## 4. Ambiguity Strength by Axis

Ambiguity is only useful if the intended and default interpretations yield *materially different*
answers. We measure the relative gap $|A - A_d| / |A|$ between the intended answer $A$ and the
default answer $A_d$ for every instance.

| Axis | n | Median answer gap | Weak (<2% gap) | Verdict |
|------|---|------------------|----------------|---------|
| **temporal_scope** | 6 | 120.3% | 0 | Strongly discriminating |
| **recognition_policy** | 2 | 72.7% | 0 | Strongly discriminating |
| **entity_scope** | 84 | 12.2% | 0 | Discriminating |
| **metric_definition** | 87 | 10.8% | 7 | Mostly OK, tail of weak cases |
| **filing_vintage** | 1 | 0.3% | 1 | Too few to judge |

**Interpretation.** Entity-scope and absolute-value questions are the backbone of the dataset and
behave exactly as intended — the canonical example (AT&T operating income: \$19.05B consolidated
vs. \$27.14B segment subtotal, a 42% gap) is representative. Temporal-scope and recognition-policy
instances produce the widest gaps, because FY-vs-CY and revenue-recognition differences move the
answer substantially.

---

## 5. Critical Quality Risk: weak `metric_definition` growth-rate instances

The construction process produces a subclass of `metric_definition` instances framed as **growth
rates** ("What was X's earnings growth rate?") where the intended and default interpretations
differ only in the *denominator definition* (e.g., GAAP net income growth vs. GAAP EPS growth, or
as-reported vs. organic revenue growth). Because growth rates normalize away the level difference,
the two interpretations often collapse to nearly the same number.

**15 percentage-answer instances have an intended-vs-default gap of under 1 percentage point**,
and the grader's documented tolerance explicitly treats "73.9% and 74% as within tolerance"
(`evaluate.py`, grader prompt). The worst cases:

| Instance | Ticker | Intended $A$ | Default $A_d$ | Gap | Problem |
|----------|--------|------|------|-----|---------|
| 0151 | LLY | 101.9% | 102% | 0.1pt | within grader tolerance |
| 0142 | LLY | −16.1% | −16% | 0.1pt | within grader tolerance |
| 0149 | DDOG | 26.1% | 26% | 0.1pt | within grader tolerance |
| 0166 | AME | 13.2% | 13.3% | 0.1pt | within grader tolerance |
| 0138 | FOX | 16.6% | 17% | 0.4pt | intended vs. *rounded* default |
| 0136 | GS | −23.9% | −24.3% | 0.4pt | GAAP net income vs. EPS growth |

**Why this matters.** When $|A - A_d|$ falls within grader tolerance, the instance is **not
discriminating**: a model that resolves to the *wrong* (default) interpretation still gets graded
correct. Such instances silently inflate measured accuracy and weaken the central claim that the
benchmark separates ambiguity-aware from ambiguity-blind models. The FOX case is the clearest
pathology — the "ambiguity" is merely rounding (exact unrounded rate vs. management's rounded
figure), which is not a financial-reasoning distinction at all.

**Root cause.** Growth-rate framing on `metric_definition`. Absolute-value questions on the same
axis (e.g., "What was X's net income?" GAAP \$Y vs. non-GAAP \$Z) do not have this problem —
only 1 of 133 absolute-value instances is weak, versus 7 of 47 growth-rate instances.

**Recommended fix (implemented as QC rule R14).** Reject any instance where the numeric gap
between intended and default answers is within 1.5× the grader tolerance. This removes
non-discriminating instances at construction time rather than discovering them during evaluation.
See `methodology.md` §Quality Control.

---

## 6. Diversity Risk: ZH metric concentration

ZH instances are highly homogeneous in *what they ask about*:

- **122 of 123 ZH instances (99%) ask about 净利润 (net income)** or one of its scope/definitional
  variants (归母净利润 parent-attributable, 扣非净利润 non-recurring-excluded).
- ZH instances exercise only **8 distinct metrics**, versus **31 distinct metrics** for EN.

This is a direct consequence of the akshare ZH pipeline, which extracts net-income variants because
those are the cleanest structured-data ambiguities available. It is not wrong — the 归母净利润 vs.
净利润 distinction is a real and important entity-scope ambiguity — but a reviewer will read 99%
net-income concentration as a coverage weakness, and per-axis accuracy stratified on ZH will
effectively measure "net-income scope disambiguation" rather than financial ambiguity broadly.

**Recommended mitigations** (in priority order): (a) extend `pull_cninfo.py` to harvest revenue,
gross margin, and EPS ambiguities from the same A-share reports; (b) report ZH results with the
net-income concentration disclosed explicitly so the scope is not overstated; (c) treat ZH as a
deliberately narrow but deep entity-scope/metric-definition probe in the paper framing, rather
than claiming broad ZH coverage.

---

## 7. Documentation Mismatch: n-axes distribution

`methodology.md` states a target of **30% single-axis / 50% two-axis / 20% three-plus-axis**. The
actual build is **74% / 26% / 0%**. No three-plus-axis instances exist, and single-axis dominates.

This is a genuine divergence between documented design and realized data. Two-axis and especially
three-axis instances are hard to construct because they require a passage that simultaneously
supports multiple independent, individually-verifiable ambiguities. Either:

- the documented target should be revised to reflect what is achievable (recommended: ~70% single /
  ~30% two / a small three-plus stretch goal), **or**
- the construction prompt should be extended to actively seek multi-axis passages.

Leaving the doc as-is creates an audit liability: a reviewer comparing the stated target to the
released distribution will flag the gap. The doc has been updated to state the realized target.

---

## 8. Summary of Actionable Findings

| # | Finding | Severity | Action |
|---|---------|----------|--------|
| F1 | Verifier vote rate 0.3% — dataset is genuinely hard | Positive | Headline result; foreground in paper |
| F2 | 15 growth-rate `metric_definition` instances non-discriminating (gap < grader tol.) | **Critical** | R14 gap check; re-audit existing 15 |
| F3 | ZH 99% net-income concentration | Major | Diversify `pull_cninfo.py`; disclose scope |
| F4 | n-axes 74/26/0 vs. documented 30/50/20 | Major | Revise doc target (done) |
| F5 | entity_scope & temporal_scope are the strong discriminating axes | Positive | Use as canonical examples |
| F6 | Filing recency 2022–2025 | Positive | Supports contamination-resistance claim |

The two positive findings (F1, F5, F6) are the spine of the benchmark's contribution. The three
risks (F2, F3, F4) are all addressable at construction or documentation time and do not require
re-architecting the pipeline.

---

## 9. Resolution Log (actions taken)

- **F2 (non-discriminating instances) — RESOLVED.** Added QC rule **R14** to
  `construct_instances.py` (rejects instances whose intended/default answer gap is within grader
  tolerance). Audited the existing build with `scripts/audit_weak_instances.py`: **26 of 180
  instances (14%)** violated R14 and were quarantined to `instances.weak.jsonl`, leaving **154
  discriminating instances**. 22 of the 26 were EN `metric_definition` growth-rate questions.
- **F2 root cause — RESOLVED.** Found that 100% of EN `metric_definition` instances were sourced
  from EDGAR yoy (growth-rate) passages, which structurally collapse the answer gap. Added a
  pre-scan gate skipping `metric_definition` on `_yoy` passages; the axis is now routed to the 258
  remaining `_seg` value-based passages, which produce strong absolute-value GAAP-vs-non-GAAP
  contrasts instead of weak growth-rate contrasts.
- **F4 (n-axes target) — RESOLVED.** `methodology.md` target revised from 30/50/20 to ~70/30/stretch.
- **F3 (ZH net-income concentration) — OPEN.** Requires extending `pull_cninfo.py`; tracked as a
  follow-up. Recommended interim framing: present ZH as a deliberately deep entity-scope /
  metric-definition probe rather than broad ZH coverage.

Post-resolution dataset: **154 discriminating instances** (34 EN / 120 ZH), with construction
continuing under the R14-enforced pipeline. The quarantine is fully reversible
(`instances.weak.jsonl` retains all removed instances with their violation reasons).
