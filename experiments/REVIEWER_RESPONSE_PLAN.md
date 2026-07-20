# Reviewer-Response Experiment Plan (FinInteract / AAAI)

Response to the 9-point experimental-isolation critique. Grounded in a full codebase
map (Sept scan). **Headline finding: the harness already implements far more than the
critique assumes.** Axis-Aware ReAct, all baseline/oracle modes, wrong-axis/generic-ask
metrics, both simulator variants, and the steering intervention are already coded. Many
"experiments" reduce to *run the existing mode and report*; the real new work is a
handful of small builds plus the human-annotation and GPU-training studies.

Legend for each item: **RUN** (code exists, just execute) · **BUILD-S** (small script,
<1 day) · **BUILD-L** (substantial) · **HUMAN** (needs annotators) · **GPU** (needs the
8×A100 box) · **$$** (needs API budget).

---

## PROGRESS LOG

**DONE (local, no budget):**
- **Exp 8 — company-clustered bootstrap.** Built `scripts/analyze_company_cluster.py`; ran on
  all committed results. Result: clustered CIs only ~1.0–1.3× wider than i.i.d. (median
  ~1.1×); **no headline conclusion moves** (GPT-5 interact 20.2%, i.i.d. [14.5,26.6] →
  clustered [14.4,26.2]). Reviewer-8 sensitivity check passes.
- **Exp 9 (lexical part) — bag-of-words / length controls.** Built `scripts/probe_bow_control.py`.
  Result: **detection (Q vs Q+C) BoW = 1.00 AUROC** (raw detection is lexically trivial — do
  not lean on it); **axis-decoding lexical floor is high** — BoW 0.73 AUROC combined (EN 0.97
  but n=37 overfit; ZH 0.57≈chance), length-only 0.64, company-disjoint BoW 0.80. The neural
  axis probe must be shown clearly ABOVE this floor; current cited ~0.71–0.74 margin is thin.

**BUILT, awaiting annotators (human track designed):**
- **Exp 4a — blind default annotation:** `annotation/make_default_sheet.py` + `score_default.py`
  (%-match to A_d, A_i; Gwet AC1; answer entropy; noticed-second-answer rate). Sheets verified.
- **Exp 3a — judge–human agreement:** `annotation/make_judge_sheet.py` + `score_judge_agreement.py`
  (balanced correct/incorrect + hit/miss sheets; raw agreement, κ, AC1, judge FP/FN vs human
  consensus, human-human ceiling). Sheets verified.
- **Exp 7 — turn-matched human baseline:** `annotation/make_baseline_multiturn.py` +
  `score_baseline_multiturn.py` (up to 6 rounds, annotator_id, 1-turn vs matched ceiling,
  AxisHit@1 + AnyAxisHit, per-language/annotator). Sheets verified.
- Next human-track step is recruitment (≥3 annotators/language incl. extra bilinguals for ZH),
  then run the `score_*` scorers (need `OPENAI_API_KEY` or `--self-graded`).

**PENDING:** Exp 1, 2, 3b, 4b (API budget); Exp 5, 6, 9-activations (GPU); Exp 6 KTO/GRPO
(external code blocker).

---

## Resource buckets (what gates each experiment)

| Bucket | Experiments | Blocker |
|---|---|---|
| **Local / free** (CPU, cached data) | 8, parts of 9 | none — can start immediately |
| **API budget** (OpenRouter/OpenAI) | 1, 2, 3b, 4b | OpenAI key is dry; OpenRouter ~$11 and in use |
| **GPU (8×A100)** | 5, 6, 9-activations | cluster access |
| **Human annotators** | 3a, 4a, 7 | recruit + bilingual coverage |
| **External code not in repo** | 6 (KTO/GRPO) | re-obtain `fininteract_grpo_kit` |

---

## HIGHEST PRIORITY

### Exp 1 — Fill the Axis-Aware ReAct main-results row  ·  RUN + small metric fold-in  ·  $$
**Reviewer:** method row `main.tex:683` is empty; report acc, IR, AxisHit@1, wrong-axis,
generic-ask, turns, ECE vs standard ReAct, always-ask, axis-oracle, template-oracle.

**Already exists** (`scripts/evaluate.py`):
- Proposed method = **`axis-aware` mode** (`AGENT_SYSTEM_AXIS_AWARE` :200, dispatch :711;
  4-step protocol with a `state` action handler :803).
- Baselines are all modes: standard ReAct = `answer+search+interact`; `always-ask` :146;
  `axis-oracle` :158; `template-oracle` :329/:666.
- Metrics in `compute_metrics()` :876 — accuracy, `ir_rate`, `axis_hit_at1` :924,
  `wrong_axis_rate` :922, `generic_ask_rate` :921, `avg_turns`. All present.
- ECE lives in `scripts/analyze_results.py::confidence_calibration()` :101 (needs runs
  produced with `--elicit-confidence`).

**Build (small):** fold ECE into the `--summary`/RESULTS TABLE so all metrics land in one
place (port `confidence_calibration` into `compute_metrics`, add a column at :1239).

**Run:** one sweep over the 5 conditions × full 173 (or a fixed n), `--elicit-confidence`
on. Cost ≈ the standard per-model full-run rate (~$15–20/model equivalent through the
OpenRouter judge pipeline). **This is the single highest-value run — it is the paper's
central "we provide an intervention" claim and is currently unsupported.**

---

### Exp 2 — Factorial 2×2 isolating elicitation from retrieval/grounding  ·  BUILD-S + RUN  ·  $$
**Reviewer:** context-oracle currently supplies *both* the resolved interpretation C and
both evidence spans (`main.tex:918`), so the 93–95% cannot attribute the bottleneck to
elicitation vs evidence access. Want Context {absent, present} × Evidence {retrieval,
oracle}, retrieved evidence held fixed when adding C; also report axis-oracle and
template-oracle.

**Already exists:** `interp-oracle` mode injects C but *withholds* spans/search
(:713/:725) — that is the "C present, evidence retrieval" cell we lacked. `axis-oracle`,
`template-oracle` present. Oracle retrieval via `oracle_search()` :579 + `--passage-file`.

**Build (small):** add a `context-oracle` mode = C **plus** oracle evidence spans **plus**
ordinary retrieval, and a `context-present/retrieval` cell that holds retrieved evidence
fixed while adding C. Hook next to the `interp-oracle` branch (:713–727); instance fields
`intended_interpretation`, `default_interpretation`, `passage_text` already loaded (:659).

**2×2 grid to run:**
| | Retrieval evidence | Oracle evidence |
|---|---|---|
| **C absent** | `answer+search+interact` | oracle-evidence + interact |
| **C present** | `interp-oracle` (exists) | `context-oracle` (build) |

Plus `axis-oracle`, `template-oracle` as the "what axis / question-formulation" rungs.
Decomposes: when-to-ask, what-axis, question-formulation, response-integration, grounding.

---

### Exp 3 — Validate grader / AxisHit judge / simulator against humans  ·  (a) BUILD-S+HUMAN, (b) RUN  ·  $$
**Reviewer:** primary metrics depend on GPT-4o-mini + GPT-5 sim; judge-human agreement
missing; also report the promised deterministic & 15%-noisy simulator robustness.

**3a — Judge-human agreement (BUILD-S + HUMAN).** Human-label kit exists (`annotation/`,
`score_validation.py` computes Gwet AC1 / Cohen κ, but **human-human only**). Judges are
`evaluate.grade()` :404 and `classify_axis_hit()` :531. **Build:** a script that runs the
LLM grader + AxisHit judge over a blind-double-annotated stratified sample and reports
judge-vs-human agreement, FP/FN rates, and whether headline conclusions flip under human
labels. Needs annotators.

**3b — Simulator robustness (RUN).** Both variants already CLI-wired: deterministic =
`--user-sim oracle` (`_oracle_user_answer` :423); 15%-noisy = `--user-sim noisy`
(`_noisy_user_answer` :498, `noise_rate=0.15`). Just re-run the headline conditions under
each and add a comparison table. Pure execution — this closes the "defined but never
shown" gap with no new code.

---

### Exp 4 — Empirically validate the "default interpretation"  ·  (a) HUMAN, (b) RUN/BUILD-S  ·  $$
**Reviewer:** single-gold argument assumes a conventional benchmark picks the constructed
default (`main.tex:723`); test with blind annotators (see Q + passage, not C) choosing the
most natural answer; report % matching A_d, agreement, entropy. Also re-run AmbigQA with
exactly one predetermined gold (random-gold, human-majority-gold) instead of any-valid.

**4a — Blind default annotation (HUMAN).** Validation sheet already has a blind "default
plausible" item (`make_validation_sheet.py`, H2). **Build:** dedicated blind task — show
Q + passage, hide C, annotator produces/selects the natural answer; scorer computes
%-match to `default_answer`, inter-annotator agreement, answer entropy. Needs annotators.

**4b — Single-gold AmbigQA re-grade (RUN + BUILD-S).** `run_illusion.py::eval_instance()`
:103 already grades intended / default / any-valid three ways; **random-gold already
supported** via `build_ambig_instances.py --intended hash` (`robust_hash.jsonl` exists).
So single-gold vs any-valid and random-gold are **RUN**. Human-majority-gold needs human
reading-preference labels (BUILD-S selection rule + HUMAN).

---

## IMPORTANT ROBUSTNESS

### Exp 5 — Matched sample sizes + multiple seeds for the salience/co-evolution claim  ·  BUILD-S + GPU
**Reviewer:** recognition=9, temporal=7, filing=0 instances (`main.tex:561`) is too thin to
claim recognition is scale-invariant or that salience gates co-evolvability. Want
entity/metric downsampled to 9/26; recognition curves at 30/60/90; 3–5 seeds; equal-token
& equal-example controls. Reframe "salience predicts data need" as hypothesis unless shown.

**Already exists:** `mr_coevolve_v2.py` slice mechanism — `--slice-size` (30 default),
`--train-window` (→30/60/90), `--val-size`, `--seed` (:135). One alt seed ("seed7") already
run. Analysis `analyze_learnability.py`.

**Build (small):** an explicit equal-N subsampler (current code slices by pool order, not a
target-N sampler) to cut *exactly* 9/26/30/60/90 across axes; a driver loop over ≥3–5 seeds.
**Caveat to state in-paper:** recognition pool = 41 generated + 8–9 human; curves will stay
noisy — either construct a larger recognition probe (`pull_recognition_policy.py`) or
explicitly frame recognition scale-invariance as a *hypothesis*. **GPU-bound** (Qwen3-4B
QLoRA per point × seeds).

### Exp 6 — Proper SFT/RL baselines + non-leaky end-to-end eval  ·  BUILD-L + GPU + external code
**Reviewer:** training ladder is one leaky run (`main.tex:1285`); compare axis-guided SFT
vs generic-clarification SFT, random-axis SFT, always-ask SFT, axis-aware-prompt-no-train,
SFT-without-KTO/GRPO; company/filing-disjoint test + ordinary retrieval; mean±sd over seeds.

**Already exists:** SFT trainer `experiments/gpu_eval/c2_sft_train_gc.py` (TRL+QLoRA); axis
demo generator `gen_axis_guided_demos.py`; non-leaky retrieval `search_no_leak.py`;
guided-vs-unguided spec (`experiments/sft_vs_rl/README.md`).

**Build:** (1) three new demo generators as teacher-prompt variants of
`gen_axis_guided_demos.py` — generic-clarification (no hint), random-axis (wrong hint),
always-ask (force ask, axis-agnostic). (2) Non-leaky eval wiring via `search_no_leak.py`.
(3) Seed loop + company/filing-disjoint split (fields exist on rows).
**Blocker:** the **KTO and GRPO trainers are not in the repo** (external `fininteract_grpo_kit`
/ verl). The SFT-without-KTO/GRPO ablation and any KTO/GRPO benefit claim require
re-obtaining that kit or reimplementing (TRL `KTOTrainer`; verl multi-turn GRPO).
**Honest current state:** evidence supports axis-guided SFT; a distinct KTO/GRPO benefit is
not yet isolable — plan should either produce it or soften the claim.

### Exp 7 — Strengthen the human baseline  ·  BUILD-S + HUMAN
**Reviewer:** humans get 1 yes/no; agents get up to 10 rounds (`main.tex:964`) — not
comparable. Run humans under 1-turn AND agent-matched budgets, several annotators/language;
current ZH result confounded (only one bilingual annotator).

**Already exists:** `annotation/make_baseline_sheet.py` + `score_baseline.py` (reuses
`evaluate.grade`/`classify_axis_hit`). **Build (small):** an agent-matched multi-turn
baseline sheet (up to N clarifications). **Need:** ≥2–3 annotators/language incl. more
bilinguals for ZH. HUMAN-bound.

### Exp 8 — Company-clustered bootstrap  ·  BUILD-S · LOCAL (start now)
**Reviewer:** 173 instances / 61 companies (`main.tex:537`); instance-level CIs too narrow.
Add company-clustered bootstrap + company-disjoint sensitivity.

**Already exists:** `analyze_breakdowns.py::boot_ci()` :38 (B=2000) and
`paired_boot_pvalue()` :48 — but resample **individual instances** (i.i.d.). `company` /
`source` fields are on every result row (`evaluate.py` :694/:868).
**Build (small):** a cluster/block bootstrap that resamples the 61 companies (not 173 rows)
+ a company-disjoint sensitivity split. **Pure local analysis, no API/GPU/humans — this is
the cheapest, fastest credibility win; do it first.**

---

## MECHANISTIC

### Exp 9 — Probe controls / causal intervention  ·  BUILD-S (analysis) + GPU (activations) 
**Reviewer:** probe Q vs Q+C may just detect context presence; axis decoding may exploit
lexical/source cues (`main.tex:1085`). Add company/source/template-disjoint splits,
shuffled-label & bag-of-words baselines, length-matched irrelevant-context controls,
cross-language transfer; preferably activation steering showing decoded-direction→asking.

**Already exists:** extraction `probe_activations.py` (stores all-layer acts, GroupKFold by
instance); analysis `analyze_probes.py` (detection + `behavioral_mismatch()` = the
"represented but not acted on" result); per-axis AUROC `probe_per_axis.py`; **steering
already coded** — `experiments/gpu_eval/steer_axis.py` (`build_directions` :38, forward-hook
`steer` :51, sweeps α/layers, measures IR + AxisHit@1). Splits are currently random only.

**Build (small, mostly CPU on cached `.npz`):** shuffled-label null for the *linear* probe;
bag-of-words baseline; length-matched irrelevant-context control; cross-language (EN/ZH)
train→test split; company/source/template-disjoint splits (langs stored, company derivable).
**GPU:** re-extract activations for the length-matched control; **run the existing
`steer_axis.py` at scale** to convert Finding 10 from correlational to causal (the code is
done — this is the highest-leverage mechanistic upgrade and is currently left as "future
work" in `main.tex:1140`).

---

## Suggested sequencing

1. **Now, local/free:** Exp 8 (company bootstrap) + Exp 9 CPU controls on cached activations
   (shuffled-label, bag-of-words, length-matched, cross-lang, disjoint splits). No budget.
2. **Small builds, then queue for API budget:** Exp 1 metric fold-in + Exp 2 `context-oracle`
   mode + Exp 4b random-gold re-grade. Run when OpenRouter is topped up (batch with the
   current model sweep to amortize judge cost).
3. **Pure runs when budget allows:** Exp 3b (simulator robustness flags) — trivial, high ROI.
4. **GPU queue:** Exp 9 steering at scale (causal upgrade), Exp 5 matched-size/seed curves,
   Exp 6 SFT baselines (+ resolve KTO/GRPO external-code blocker before promising RL claims).
5. **Human track (parallel, long lead):** recruit annotators → Exp 3a (judge-human), Exp 4a
   (blind default), Exp 7 (turn-matched baseline, extra bilinguals). Design sheets now.

## Claims to soften unless the experiment lands
- "Salience predicts data need" → **hypothesis** until Exp 5 (matched N + seeds) (recognition
  n≈9 cannot support scale-invariance).
- "KTO/GRPO add benefit" → **not isolable** until Exp 6 baselines (currently one leaky run).
- "Represented but not acted on" (probe) → **correlational** until Exp 9 steering is run.
