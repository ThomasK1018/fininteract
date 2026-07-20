# FinInteract vs. Supervisor's Benchmark-Paper Template — Fit & Restructure Plan

Assessment of how the current paper (`paper/main.tex`) maps to the DIAL-lab Benchmark/Evaluation
template (StatQA / nvBench 2.0 / VisJudge-Bench lineage), and what to restructure.

## Verdict
**FinInteract fits this template unusually well — it already contains all five core elements,
including the rare Element 5 (a specialized model).** The work is not under-built for the
template; it is if anything *over*-built and sprawling. The main job is **re-organization and
three additions (explicit RQs, Design Goals, a Discussion section)**, not new research.

## The five core elements — all present
| Element | Template expectation | FinInteract | Status |
|---|---|---|---|
| 1. Research Gap | a real evaluation blind spot | **single-gold illusion**: financial QA benchmarks use one gold answer, blind to ambiguity + interaction (direct parallel to nvBench 2.0's "single-correct-answer ignores query ambiguity") | ✅ strong |
| 2. Construction Pipeline | scalable, high-quality, novel | three-role Constructor→Verifier from EDGAR; **paired intended/default** = controllable ambiguity injection (à la nvBench); `fig_framework.pdf` | ✅ strong |
| 3. Evaluation Framework (taxonomy) | fine-grained diagnostic taxonomy | **5-axis ambiguity taxonomy** (entity/metric/temporal/filing/recognition) × 3 modes × metrics (Acc, IR, AxisHit, DisE⁺, ECE) | ✅ strong |
| 4. Empirical Findings | insights, not just a leaderboard | Findings 0–10, Human-vs-LLM, cross-vendor, single-gold illusion *quantified*; already uses **bold "Finding X:"** style | ✅ strong |
| 5. Specialized Model (optional) | a method that improves the ability | **TWO**: inference-time **Axis-Aware ReAct** + training-time **axis-guided SFT/GRPO** & co-evolution | ✅✅ rare strength |

## Section mapping: current → template (tight 7-section form)
| Template section | Current material | Action |
|---|---|---|
| **1. Introduction** | §1 Introduction | Add explicit **RQs**, **Design Considerations** framing, contributions↔RQs mapping |
| **2. The Proposed Benchmark** | §2 Task Definition + §3 Taxonomy + §4 Dataset Construction | Merge into one section; add **Design Goals G1–G4**; keep pipeline `fig_framework` as **Figure 2**; stats = `tab:stats` |
| **3. Specialized Model / Methodology** | §7 Diagnosis & Intervention (Axis-Aware ReAct + Axis-Guided SFT/GRPO + Co-Evolution) | **Reposition earlier** as its own section; this is Element 5 |
| **4. Experiments & Empirical Findings** | §5 Evaluation Framework (→ 4.1 Setup) + §6 Experiments | Overall table = `tab:main`; organize **fine-grained analysis by RQ**; case studies; keep Finding-X |
| **5. Discussion & Research Opportunities** | (currently folded into Limitations/Conclusion + co-evolution) | **NEW dedicated section** |
| **6. Related Work** | §8 Related Work + `tab:comparison` | Keep; reference `tab:comparison` earlier (intro) per template |
| **7. Conclusion** | §10 Conclusion | Keep, tighten |

## Checklist — what's already there vs. the gaps
Already satisfied (✅): Running-example figure (`fig_motivation.pdf` = Figure 1), pipeline figure
(`fig_framework.pdf`), **benchmark comparison table** (`tab:comparison`), overall-performance
table (`tab:main`), error taxonomy (`tab:errors`), Human-vs-LLM (human baseline + new judge
validation), case study (mechanistic), Finding-X summaries, open data/code.

**Gaps to fill (the actual work):**
1. **Explicit Research Questions (2–3) in the Introduction — MISSING (highest priority).** Proposed:
   - **RQ1** — Do frontier models *detect and resolve* financial-report ambiguity, and does interaction help?
   - **RQ2** — Is the bottleneck knowing *when/what to ask* (elicitation) rather than evidence access?
   - **RQ3** — Can targeted inference-time and training-time interventions on the ambiguity axis improve resolution?
   Then map contributions and §4 analysis subsections onto RQ1/RQ2/RQ3.
2. **Design Goals G1–G4 — MISSING as such.** Reframe §2.2 "Design Principles" as explicit goals
   (e.g., G1 representativeness/coverage, G2 fine-grained per-axis evaluation, G3 low-cost/scalable
   leak-safe construction, G4 high data quality / human-validated).
3. **Discussion & Research Opportunities — MISSING as a section.** Pull forward-looking content
   (benchmark self-refresh/co-evolution, Human-AI collaboration, extending axes/languages).
4. **Structure is sprawling (10 sections).** To hit the template's crisp Gap→Benchmark→Method→
   Eval→Insights→Opportunities line, consider moving the **mechanistic probe** and **co-evolution
   self-refresh** to Discussion/Appendix so the main narrative stays tight.

## How the recent reviewer-response work slots in
- **Judge–human validation** (grader κ=0.85; AxisHit classifier fixed) → strengthens §4.1 Setup /
  reliability (template explicitly values Human-vs-LLM checks).
- **Cross-vendor pilot** (`tab:crossvendor`) → broadens Overall Performance (Element 4; template
  wants open + closed, multiple scales).
- **Axis-Aware ReAct results + elicitation/grounding 2×2** → the core of §3 Specialized Model and a
  fine-grained "is elicitation the bottleneck?" analysis (RQ2).
- **Company-clustered CIs, single-gold AmbigQA re-grade** → statistical rigor the checklist rewards.
- **Blind-default human validation (66%)** → empirically grounds the Gap (Element 1).

## Bottom line
No new experiments are needed to fit the template. The restructure is: (a) add explicit RQs +
Design Goals + a Discussion section, (b) reorder so the specialized-model content is its own section
before Experiments, (c) optionally demote the mechanistic/co-evolution material to keep the spine
tight. The paper's unusual asset is having **both** a benchmark *and* two intervention methods — lead
with that.
