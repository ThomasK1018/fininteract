# Motivation

## The Evaluation Gap in Financial Question Answering

Large language models are increasingly deployed in high-stakes financial workflows: earnings call analysis, SEC filing review, analyst report generation, and investor query handling. Yet the benchmarks used to evaluate these models share a critical blind spot — they assume that every query is fully specified before the model begins to answer.

Consider the question *"What was Apple's revenue growth rate?"* This question is unanswerable as stated. It is ambiguous across at least three independent dimensions: whether *revenue* means GAAP total net sales or a non-GAAP organic figure; whether *growth* is measured against the prior fiscal year (ending in September) or the calendar year; and whether the scope is worldwide consolidated revenue or a specific segment such as Services. A skilled financial analyst would not guess — they would ask a clarifying question. Current benchmarks reward the model that guesses correctly; they do not distinguish between a model that genuinely understands the ambiguity and a model that happens to pick the default interpretation.

## Limitations of Existing Financial QA Benchmarks

Existing financial QA benchmarks enforce a single-gold-answer convention and evaluate models in a single turn:

- **FinanceBench** (Islam et al., 2023) provides 150 carefully curated questions over SEC filings, each with a unique correct answer verified against the source document. Queries are pre-disambiguated by construction.
- **FinQA** (Chen et al., EMNLP 2021) targets numerical reasoning over earnings reports but gives the model the exact evidence table alongside the question.
- **DocFinQA** extends FinQA to long-context settings but maintains the same single-answer paradigm.
- **BizBench** and **FinBen** aggregate multiple financial tasks but treat QA as closed-book or retrieval-augmented reading comprehension, not as an interactive dialogue.

None of these benchmarks ask whether a model *recognizes* that a question is ambiguous, nor whether it can *resolve* that ambiguity through strategic interaction with a user.

## The Ambiguous-QA Gap in Finance

The ambiguous-QA literature has matured significantly in the general domain. AmbigQA (Min et al., EMNLP 2020) introduced multi-answer formulations for Wikipedia questions. CondAmbigQA (Li et al., EMNLP 2025) extended this with condition-structured annotations. CLAM (Kuhn et al., 2022) demonstrated that clarification pipelines improve answer accuracy on ambiguous queries. However, all of this work operates on general-domain factual questions (Wikipedia, news, encyclopedias) and relies on human-judged ambiguity labels.

Finance is structurally different in ways that make it a natural and important testbed:

1. **Ambiguity is endemic, not incidental.** Financial metrics are defined by accounting standards (GAAP vs. non-GAAP), reporting scope (consolidated vs. segment), filing vintage (original vs. restated), and temporal convention (fiscal vs. calendar year). A single number — "net income" — can legitimately differ by 30–40% depending on which of these dimensions the questioner intended. This is not a failure of the question; it is a property of the domain.

2. **Misanswers are costly.** An investor acting on the wrong revenue figure, a compliance officer citing the wrong filing version, or a research analyst computing a growth rate against the wrong base period can cause real financial harm. This raises the stakes for interaction far above what obtains in general-domain QA.

3. **Ground truth is verifiable without human judges.** SEC filings, XBRL facts, and structured financial databases provide exact, machine-readable values for every metric, period, and scope. A correct answer can be verified programmatically, satisfying the "Easy to verify" principle from InteractComp (arXiv:2510.24668).

## The Interaction Stagnation Hypothesis

InteractComp (ICLR 2026) established an important longitudinal finding: while retrieval accuracy on financial and web search benchmarks has improved rapidly with model scale, interaction capability — the ability to recognize when to ask rather than answer — has not kept pace. Across eight frontier models on general-domain tasks, fewer than 14% of questions were answered correctly in full-interaction mode, even though forced-interaction experiments demonstrated that the same models could achieve 60–70% accuracy when required to elicit disambiguating context before answering.

We hypothesize that this stagnation is more severe in finance than in the general domain. Financial queries require technical domain knowledge to even *recognize* the axes of ambiguity: a model that does not know what "non-GAAP" means cannot recognize that "adjusted net income" is ambiguous. Likewise, an agent unaware of Microsoft's June 30 fiscal year end cannot identify that a question about "FY2023 revenue" is temporally under-specified for a user who may mean the calendar year.

**FinInteract** tests this hypothesis directly: we adapt the InteractComp methodology to the finance domain, extend the taxonomy of ambiguity from general-domain entity disambiguation to a five-axis financial ambiguity framework, and measure whether the stagnation observed by InteractComp is replicated — and amplified — in finance.

## Why Interaction, Not Enumeration

An alternative approach would enumerate all plausible interpretations and return a set of answers (the paradigm of AmbigQA and CondAmbigQA). We reject this design for three reasons. First, real analyst workflows are dialogic: an analyst receiving an ambiguous question asks for clarification, they do not preemptively list all possible answers. Second, enumeration scales poorly with the number of interpretable dimensions: five axes with two-to-three values each yield up to $3^5 = 243$ combinations per question, making enumeration impractical. Third, enumeration does not test the core capability we care about — the model's ability to identify *which* dimension is under-specified and formulate a targeted clarifying question.

We therefore adopt the ReAct-style three-action protocol from InteractComp: the agent may **search** for information, **interact** with the user to elicit a clarifying constraint, or **answer** with a final response. This design directly mirrors how a skilled financial analyst would handle an ambiguous query.
