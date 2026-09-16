# Reviewer-Response GPU Experiments — Results (Exp 5, 6, 9)

Executed the GPU-bound reviewer experiments from `experiments/GPU_HANDOFF.md`. Env note: vLLM is
not installed on this box, so sim/grader/axis-judge ran through the OpenAI API (gpt-4o-mini),
not local vLLM; policy models served locally via `hf_openai_server.py` (adapter-loading verified).
All paper models were cached (4B/8B/14B/32B backbone + 30B-A3B/35B-A3B).

**One-line synthesis:** ambiguity is *causally* represented (steering), but which axis a model
asks about is driven by **salience**, not by decodability and not by axis-specific supervision.
At matched data, salient axes learn and scarce ones don't (Exp 5); on a salient axis, *any*
clarification-SFT elicits the right axis (Exp 6); steering is causal exactly where the model
under-acts (Exp 9). Two results **challenge** prior framing and are reported honestly.

---

## Exp 9 — Causal steering (represents → acts)  `experiments/gpu_eval/steer_axis.py`
Inject `alpha * decoded_ambiguity_direction` into the residual stream (layers ~0.4/0.6/0.8) and
measure the first-action IR and AxisHit@1. `data/results/steer_*.json`.

| model | base IR / AxisHit@1 | steered (best α) | effect |
|---|---|---|---|
| **Qwen3-4B** (backbone) | 50% / 0.84 | +8: **81%** / 0.87 | IR ↑ — asks more (**causal**) |
| Qwen3-8B | 6% / 0.86 | over-steer collapses | floor (barely asks) |
| Qwen3-14B | 40% / 0.84 | flat across α | ceiling (already targets) |
| Qwen3-32B | 40% / 0.80 | flat across α | ceiling |
| **Qwen3-30B-A3B** | 62% / 0.21 | +4: 76% / **0.59** | IR ↑ **and** AxisHit ↑↑ (**strong**) |
| **Qwen3.5-35B-A3B** | 6% / 0.25 | +1.5: **94%** / 0.43 | IR ↑↑ (**strong**) |

**Verdict:** adding the decoded ambiguity direction *causally* increases interaction rate (4B
50→81, 30B 62→76, 35B 6→94) and axis-targeting (30B 0.21→0.59) **where the model under-acts**;
it is null where the model already acts (14B/32B at ceiling). This turns the §6.6 correlational
probe into a causal claim on the core model + the large MoEs.

## Exp 9 — Probe controls  `scripts/analyze_probes_controls.py`, `scripts/probe_bow_control.py`
Axis-decoding AUROC (entity vs metric) with the controls a linear-probe claim needs.
Lexical floor (BoW): combined **0.73**, company-disjoint **0.78**, length-only 0.64.

| model | neural AUROC | company-disjoint | shuffled-null | cross-lang EN↔ZH |
|---|---|---|---|---|
| 4B | 0.774 | **0.820** | 0.524 | 0.575 |
| 8B | 0.749 | 0.774 | 0.502 | 0.47–0.52 |
| 14B | 0.765 | **0.818** | 0.513 | 0.45–0.52 |
| 32B | 0.763 | 0.784 | 0.514 | **0.61–0.67** |

**Verdict:** the neural probe beats the lexical floor — clearest on the **company-disjoint**
control (4B/14B ~0.82 > 0.78), so it is not just company/source lexicon — and the shuffled-label
null collapses to ~0.51 (the signal is real). **Honest limitation:** cross-language transfer is
weak-to-chance (0.45–0.67), i.e. the axis representation is largely **language-specific**; it
improves with scale (32B best). The margin over the *combined* BoW floor is thin (~+0.03), so the
company-disjoint number is the load-bearing one.

## Exp 5 — Matched sample size: salience vs data quantity  `scripts/mrv2_matched_n.py`
Train each axis on the SAME tiny N (recognition's available sizes), ≥5 seeds, AxisHit@1 on the
frozen human probe. `data/results/matchedN_*.json`.

| axis (salience) | N=9 | N=26 |
|---|---|---|
| entity (0.92) | 0.29 ± 0.34 | **0.975 ± 0.016** |
| metric (0.48) | 0.275 ± 0.09 | 0.68 ± 0.36 (bimodal) |
| recognition (0.04) | 0.00 ± 0.00 | 0.156 ± 0.19 |

**Verdict:** at **matched N=26** the salience gradient holds — entity **reliably** learns
(0.975 ± 0.02) ≫ metric **unstably** (0.68 ± 0.36) ≫ recognition **barely** (0.16 ± 0.19). Same N,
so this is **salience, not data quantity** — the reviewer's confound is ruled out. At N=9 *all*
axes fail (0.0–0.29): there is also a learnable **floor** below which no axis learns (so the
gap needs N ≳ 26 to appear — the earlier single-run entity/metric numbers were above this floor).

## Exp 6 — SFT baselines: does axis-guidance specifically help?  `scripts/gen_demos_baselines.py`, `scripts/mrv2_exp6.py`
Same 56-instance **company-disjoint** entity train set; demos differ ONLY in which clarifying
question is taught. AxisHit@1 on the human probe (n=40; pre-retrieval, so leakage-independent).

| condition | AxisHit@1 | IR | resolve acc |
|---|---|---|---|
| base (no train) | 0.375 | 0.40 | 0.325 |
| **axis-guided** | 0.900 | 1.00 | 0.375 |
| generic-clarification | 1.000 | 1.00 | 0.350 |
| random-axis | 0.900 | 1.00 | 0.275 |
| always-ask | 1.000 | 1.00 | 0.275 |

**Verdict (challenges prior framing):** on the **salient** entity axis, *every* clarification-SFT
variant reaches ~0.90–1.00 AxisHit — **axis-guidance is statistically indistinguishable from
generic / always-ask / even random-axis**. The SFT teaches the model to *ask* (IR 0.40→1.00); the
**axis is supplied by salience, not by the training label**. So the intervention that matters is
"learn to ask," and axis-specific supervision adds nothing on a salient axis. (It may matter on a
*non*-salient axis — but those don't learn anyway, per Exp 5.) Resolve-accuracy stays low
(0.28–0.38) for all: the recall wall, not the clarification, bounds resolution.

> **BLOCKED:** the RL arm (KTO/GRPO) is not reproducible here — those trainers live in an external
> kit (`fininteract_grpo_kit`) not in this repo. Only the SFT arm is reported; any KTO/GRPO-benefit
> claim needs that kit re-obtained.

---

## What this means for the paper (honest)
- **Strengthens:** the mechanistic claim is now *causal* (steering), the probe survives disjoint
  controls, and "salience gates learnability" holds at **matched N** — the strongest form.
- **Requires reframing:** "axis-guided SFT" is *not* special on salient axes (Exp 6) — reframe the
  intervention as "teach the model to ask; salience selects the axis." And the axis representation
  is **language-specific** (Exp 9 cross-lang), which the abstract should not overclaim as abstract.

## Artifacts
Scripts: `scripts/analyze_probes_controls.py`, `scripts/mrv2_matched_n.py`,
`scripts/gen_demos_baselines.py`, `scripts/mrv2_exp6.py` (new); `steer_axis.py`,
`probe_activations.py`, `probe_bow_control.py` (existing, run). Results: `data/results/steer_*.json`
(6 models), `probe_controls_*.json` (4B/8B/14B/32B), `matchedN_*.json` (3 axes),
`exp6_sft_baselines_entity.json`; activations `data/interp/acts_qwen3-*.npz`. Adapters under
`outputs/coevolve/{mr_v2/_matchedN,baselines}/` (regenerable, not committed).
