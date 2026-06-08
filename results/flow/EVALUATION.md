# Flow Experiments — Evaluation & Robustness Verdict

**Reviewer pass over the GPU runner's two-model flow study** (Qwen3-4B-Instruct-2507 dense, 37L;
Qwen3.5-4B hybrid-attention, 33L). Question: does this improve the paper's robustness?

## TL;DR
- **YES, the depth-flow result improves robustness** — it now *replicates across two independently
  architected open models*. Cite it.
- **NO, do not cite anything from the generation-flow branch** (clarification rate, onset). Both
  the runner's onset metric *and* the clarify-rate headline are artifacts. The runner caught the
  first; I caught the second.

## What is robust (cite this)
Depth-flow replicates cleanly on both models:
| | Qwen3-4B-Instruct (dense 37L) | Qwen3.5-4B (hybrid-attn 33L) |
|---|---|---|
| Peak ambiguous−disambiguated separation | layer 35 (≈95% depth) | layer 32 (≈97% depth) |
| Entity vs. metric peak layer | 35 | 32 |
| Entity / metric sign at peak | + / − | + / − |
| Shape | flat to ~L20, sharp late rise | flat to ~L25, sharper late rise |

Two models with **different attention mechanisms, layer counts, and transformers versions** both
encode (a) ambiguous-vs-disambiguated separation and (b) the entity-vs-metric *type* distinction
linearly, in the **final ~5% of the stack**, with **opposite sign** for the two axes. That
cross-architecture agreement is the robustness win — it makes "the model represents ambiguity and
its kind in late layers" a replicated claim rather than a one-model curiosity.

Figure: `paper/fig_crossmodel_depthflow.png` (normalized overlay; regenerate with
`scripts/plot_crossmodel_flow.py`).

## What is NOT robust (do not cite)
The runner's FINDINGS.md §3 makes two generation-flow claims. **Both are artifacts:**

1. **"Recognition precedes the ask" (onset metric)** — *runner already flagged this.* The
   late-layer ambiguity projection is maximal at the very first generated token (97% baseline /
   79% Qwen3.5 peak at token 0) then decays, so `lead = onset − 0 ≈ onset` is trivially positive.
   Not evidence of internal recognition firing before a clarification.

2. **"The newer model asks more — 69% vs 49%" — THIS IS ALSO AN ARTIFACT (runner missed it).**
   The clarify-detector is a regex (`?|which|specify|哪|请|具体|年份|期间`). Breakdown of what
   actually fired:
   - **Qwen3.5: 120/120 "clarifications" matched `?` inside a visible chain-of-thought**
     ("Thinking Process: … Question: What was their operating income?"). The `?` is in the
     model's *reasoning trace*, not a question to the user. 120/120 contained a "Thinking
     Process"/"Analyze the Request" block. Qwen3.5 emits CoT by default; the baseline does not.
   - **Baseline: 76/85 matched 具体 ("specifically")** inside hedge phrases like
     "具体的净利润数据需要参考最新财报" ("the specific figure should be checked against the latest
     report") — a refusal/hedge, not a clarifying question. Only ~9 matched genuine ask words.

   So the 49%/69% gap measures *CoT formatting differences between the two models*, not asking
   behavior. **Do not report a clarification-rate comparison from this data.**

## Net effect on the paper
- **Strengthens** the (planned) mechanistic subsection: replication across architectures.
- Use the **closed-model behavioral numbers** (GPT-5/GPT-4o, already in §5) for the
  "models don't ask / capture the default" claim — that is the right evidence for asking behavior.
- The open-model flow study is the **internal-representation** complement: they *represent*
  ambiguity (and its type) late in the stack. Keep the two evidence streams separate; do not let
  the open-model regex clarify-rate stand in for behavioral asking.
- Honest reporting of the caught artifacts is itself a rigor signal for the resource track.

## If you want to rescue a real asking-behavior signal later
The regex is too loose and CoT-blind. A clean redo would: strip any `<think>`/"Thinking Process"
block before matching, drop 具体/年份 from the lexicon (too common in hedges), and ideally grade
"did it ask the user a clarifying question" with the GPT-4o grader you already use in the main
eval — not a regex. Not needed for the paper; the depth-flow result stands on its own.
