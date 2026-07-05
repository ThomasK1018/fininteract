# General-domain replication of the single-gold illusion (AmbigQA)

**Goal (Route A).** Show the single-gold illusion is not finance-specific by running
the *same methodology* (LLM answer-grader + simulated responder) on a real
general-domain ambiguous-QA benchmark.

**Data.** AmbigQA (AmbigNQ light, dev). 1,149 genuinely ambiguous questions with
≥2 readings that have **distinct answers** (avg 3.36 readings/instance). Each
instance pairs the ambiguous question $Q$ with an *intended* reading (fixed,
disclosed rule: the most-specific = longest disambiguated question) and the
remaining valid readings as *defaults*. Sample = 120 (deterministic stride).

**Modes.** `answer-only` (bare $Q$), `oracle` (the disambiguated $Q$ — is the
knowledge there?), `interact` (agent may ask ONE clarifying question; an
informative responder steers toward the intended reading without revealing the
answer). Grader = gpt-4o-mini.

## Results (n=120)

| model | mode | intended acc | any-valid acc | default-capture | ask-rate |
|---|---|--:|--:|--:|--:|
| gpt-4o-mini | answer-only | **0.083** | **0.267** | 0.217 | 0.000 |
| gpt-4o-mini | oracle | 0.133 | 0.242 | 0.133 | 0.000 |
| gpt-4o-mini | interact | 0.108 | 0.267 | 0.208 | 0.125 |
| gpt-4o | answer-only | **0.183** | **0.475** | 0.375 | 0.000 |
| gpt-4o | oracle | 0.308 | 0.425 | 0.208 | 0.000 |
| gpt-4o | interact | 0.258 | 0.492 | 0.358 | 0.008 |

## What replicates (all three core FinInteract findings)

1. **The single-gold illusion generalizes.** Grading identical answer-only outputs
   against *any* accepted reading (what a lenient single-gold benchmark credits)
   vs. a *specified* target reading overstates competence by **3.2×** (gpt-4o-mini,
   0.267/0.083) and **2.6×** (gpt-4o, 0.475/0.183) — matching the finance "up to
   3×."
2. **Default-capture is near-identical to finance.** Models produce a valid-but-not-
   intended reading **22–38%** of the time; gpt-4o's **0.375** is essentially the
   finance gpt-4o figure (0.37). Direct evidence the error is ambiguity-blindness,
   not random noise.
3. **Elicitation is underused, and interaction doesn't convert.** Even *allowed* to
   ask, models clarify on **<13%** of ambiguous queries (gpt-4o: **0.8%** — it
   essentially never asks), and `interact` intended accuracy stays far below the
   any-valid ceiling. Same "don't ask; don't resolve" pattern FinInteract measures.

## Honest difference (and why it *supports* using finance as the instrument)

The `oracle` ceiling is modest in the general domain (0.13 / 0.31), not ~0.9 as in
finance. Reason: AmbigQA answers are obscure open-domain factoids that remain hard
*even when disambiguated*, so general-domain knowledge is itself a limiter. Resolving
the ambiguity still helps substantially (intended acc ~1.6–1.7× vs answer-only), but
one cannot cleanly separate "failed to elicit" from "didn't know" without verifiable
ground truth. **Finance is the better instrument precisely because its ground truth
is exact and injectable** (oracle 93–95%), isolating elicitation as the bottleneck in
a way general-domain QA cannot. The *illusion itself*, however, is domain-general.

## Caveats
- Intended reading chosen by a fixed disclosed rule (longest disambiguation); the
  choice-independent evidence is **default-capture** (a valid non-target answer) and
  the any-valid≫intended gap, both large.
- n=120 sample of 1,149; grader = gpt-4o-mini (same family as gpt-4o-mini policy).
- Artifacts: `scripts/general_domain/{build_ambig_instances,run_illusion}.py`,
  `data/general_domain/ambig_instances.jsonl`, `illusion_gpt4o{,mini}.{jsonl,summary.json}`.
