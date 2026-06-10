# Larger-Model Interp + GRPO — Independent Evaluation

Reviewer pass over the two new bundles:
`fininteract-larger-model-interp-main.zip` (Qwen3-30B-A3B, Qwen3.5-35B-A3B added to the flow
study, plus real behavioral runs + mismatch probes) and `fininteract-results-main.zip` (same
interp tree + a full **GRPO/RLVR ladder** report). I re-derived every headline number from the
raw JSON/jsonl/npz rather than trusting the summary tables.

---

## A. Interpretability — what replicates, what is artifact

### ✅ ROBUST (cite these)

**1. Axis-decoding probe replicates at scale and is the substantive static result.**
Both MoE models linearly decode the ambiguity *type* (entity vs metric) well above the 0.599
majority baseline — 0.796 (30B) and 0.803 (35B), ~5σ above chance at n=157. Combined with the
two 4B models, the "model represents ambiguity *and its kind*" claim now holds across **four
models spanning two size classes (4B → 35B) and two architectures (dense, MoE)**. That is a
genuine robustness upgrade.

**2. Depth-flow separation peaks at the very top of the stack on every model.**
Peak separation layer: 35/37, 32/33, 48/49, 40/41 — i.e. ≈95–98 % relative depth in all four.
The "ambiguity emerges late" geometry is architecture- and scale-invariant.

**3. Behavioral ask-rate differs sharply across the model generation — and this part is real.**
From the multi-turn ReAct run (counts actual `interact` JSON actions, not a regex):
Qwen3-30B asks **7 %** (12/173) vs Qwen3.5-35B **90 %** (156/173). A genuine generational jump
in clarifying *propensity*.

### ❌ DO NOT CITE (artifacts / degenerate metrics) — two the runner caught, two it did not

**4. Detection probe is trivial.** Saturates at 1.00 from **layer 1** → it is detecting
context-token *presence*, not a semantic ambiguity feature. *(Runner flagged this — correct.)*

**5. Generation recognition-onset is still the token-0 artifact.** I re-checked the new npz:
the late-layer projection peaks at token 0 in **100 % (30B) / 97 % (35B)** of instances, so any
"peak precedes the ask" lead is mechanical. The 35B's "onset = token 42.5" is not evidence.
*(Runner flagged the artifact in the 4B round; it silently reappears in the 35B table — drop it.)*

**6. ⚠️ Behavioral CORRECTNESS is contaminated by an answer-leaking search tool — runner's
"action bottleneck" gloss is not supported by it.** `run_behavior.py`'s `search` action returns
`intended_evidence_span` — the gold passage for the *intended* answer. Cross-tabulating proves
the leak:

| | asked | correct\|asked | NOT asked | correct\|not-asked |
|---|---:|---:|---:|---:|
| Qwen3-30B | 12 | 50 % | 161 | **93 %** |
| Qwen3.5-35B | 156 | 56 % | 17 | 47 % |

A non-asking model cannot legitimately be **93 % correct** on a benchmark that is ambiguous by
construction (R14 forces intended ≠ default). It is reading the answer off the leaked span.
Consequences:
- The "Qwen3 represents-but-doesn't-act = **failure**" framing is wrong: the non-asking 30B
  scores *higher*, and asking *doesn't* raise accuracy in either model (50 % vs 93 %; 56 % vs 47 %).
- The **mismatch metric (93.1 % / 9.8 %) is degenerate**: detection flags all 173, so it is just
  `1 − ask_rate` restated, not an independent measurement.

**Bottom line for the case study:** keep the *static* claim (axis-decoding + late-peak depth-flow,
now replicated 4-models) and the *descriptive* ask-rate jump (7 %→90 %). Do **not** present the
behavioral correctness, the mismatch %, or the generation-onset as evidence of a recognition-vs-
action causal split — the leaky search tool and the trivial detection probe make those circular.
A real causal claim still needs activation steering, which was not run.

---

## B. GRPO / RLVR ladder — assessment

The runner reached the **same** search-leak conclusion independently ("accuracy has a low
ceiling-to-baseline gap… the honest signal is AxisHit and Interaction"). That cross-check is
reassuring. Honest signals (held-out n=51):

| Stage | AxisHit@1 | Interaction | (Accuracy*) |
|---|---:|---:|---:|
| Qwen3-4B base | 25.0 % | 54.9 % | 84.3 % |
| SFT | 68.6 % | 100 % | 88.2 % |
| KTO | **70.6 %** | 100 % | 84.3 % |
| GRPO | 66.7 % | 100 % | 88.2 % |

\*Accuracy is leak-inflated; ignore as a learning signal.

- **SFT does the heavy lifting** — the leak-proof metrics jump base→SFT (AxisHit 25→69, Interaction
  55→100). KTO/GRPO are within run-to-run noise of SFT at n=51. Honest, unembellished read.
- The decisive fix was the **axis-guided teacher** (privately give the teacher the gold axis when
  generating SFT demos): on-axis yield ~0 %→79 %. This is a reusable methodological insight — an
  uninformed teacher only ever asks the obvious (period) axis.
- **Path B (true multi-turn GRPO in verl)** is real engineering: proof-of-life on 0.6B, then a
  stable **15-step 4B run on 6 GPUs** where reward rose 0.55→~1.05 and turns 5.7→8.6 (learning to
  ask more). Caveat: 15 steps is a learning-signal demo, not a converged model.

**Caveats to state if any GRPO number enters the paper:** n=51 test, **all judges/teacher are
local Qwen2.5-32B/72B-AWQ** (not GPT-4o — cheaper but weaker, and the policy is a Qwen sibling of
the judge), and terminal reward is partly leaked. Frame as "the dataset is *trainable* and SFT
instills axis-targeted asking," not as SOTA accuracy.

---

## C. Net effect on the paper
- **Strengthens** the mechanistic subsection: upgrade Finding 6 from "two 4B models" to "four
  models, 4B→35B, dense+MoE" for the *static* representation + late-peak geometry. Add the
  descriptive 7 %→90 % ask-rate shift across the Qwen3→Qwen3.5 generation.
- **Do not** add a recognition-vs-action *causal* claim from the behavioral correctness/mismatch
  — it is confounded here. Keep behavior claims on the graded closed-model eval (§5).
- **GRPO** is a credible optional "the benchmark is trainable" appendix/companion-method note,
  led by AxisHit/Interaction (not accuracy), with the axis-guided-teacher trick as the takeaway.
