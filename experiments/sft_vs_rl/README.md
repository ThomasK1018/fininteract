# SFT-vs-RL: what does training actually teach on FinInteract?

**Question.** The RLVR ladder (`results/grpo/RESULTS_REPORT.md`) shows the leak-proof
signals (AxisHit@1, Interaction) jump almost entirely at **SFT** (AxisHit 25→69,
Interaction 55→100), while KTO/GRPO move within noise. The naive reading is
"SFT does the heavy lifting." But two things confound that conclusion, and
disentangling them is the actual research contribution:

1. **The reward is leaked.** The training env's `search` returns the disambiguated
   gold evidence span, so the *untrained base* already scores ~84% accuracy /
   ~0.96 reward. A leaked terminal reward gives GRPO almost no gradient on
   accuracy — so "RL adds little" may be an artifact, not a finding.
2. **Did SFT teach *recognition* or *formatting*?** Asking the right axis
   (AxisHit) looks like a capability, but it could be shallow imitation of the
   axis-guided teacher's demonstrations. "SFT = format, RL = capability" is a
   heuristic (superficial-alignment hypothesis), not a law — it has to be tested.

This folder specifies three experiments, in priority order, to run on the GPU box
that holds the GRPO kit (8×A100). Scripts here are self-contained except where
they must patch the kit's env code (clearly marked).

---

## Experiment 1 — Fix the reward leak, re-run RL  *(must-do; decides everything)*

**Hypothesis.** If RL still adds little after the leak is fixed → "SFT does the
heavy lifting" is a real finding. If RL suddenly gains headroom (AxisHit/reward
climb during GRPO, and *accuracy* becomes a meaningful, non-saturated signal) →
the original conclusion was a leak artifact.

**Change.** Replace the `search` action's return value everywhere it currently
returns the single disambiguated gold span with a passage containing **both** the
intended and default evidence spans (shuffled), so the model can no longer read
the answer off the search result — it must *clarify* to disambiguate. Use
`search_no_leak.py::nonleaky_search` as the drop-in.

**Where to patch (grep the kit):**
- The TRL single-turn env search handler (the `search` action branch).
- `verl_integration/fininteract_agent_loop.py` — the `search` branch of the
  ReAct loop.
- (Also `src/run_behavior.py` if you re-run the behavioral eval, same leak.)

**Re-run.** Re-generate eval on the held-out `test.jsonl` (n=51) for: base, SFT,
KTO, GRPO — now with the non-leaky env. Then run a fresh GRPO from the KTO/SFT
adapter for ≥100 steps (the 30-step run had no headroom *because of* the leak).

**Report back:** the 4-stage table (Accuracy, Interaction, AxisHit@1, reward) under
the non-leaky env, plus the per-step reward / reward_std / num_turns history of the
new GRPO run (CSV or the raw verl/wandb log — this is what lets us plot a real
learning curve instead of 4 points).

---

## Experiment 2 — Guided vs. unguided SFT ablation  *(cleanest format-vs-content test)*

**Hypothesis.** If AxisHit only rises with the **axis-guided** teacher's demos and
**not** with unguided demos, then the model is imitating the demonstrated axis
choice, not independently recognizing the ambiguity — the "formatting" reading.

**Change.** Train two SFT adapters from identical hyperparameters:
- `sft_guided` — the existing 372 axis-guided demos.
- `sft_unguided` — demos from the *uninformed* teacher (no gold-axis hint).
  (The kit already generates these; `gen_trajectories_par.py` without
  `--axis-guided`.)

**Report back:** AxisHit@1 and Interaction on test.jsonl for both adapters, plus
the on-axis yield of each demo set.

---

## Experiment 3 — Probe the axis representation, base vs. SFT  *(most on-brand; ties to paper §6.6)*

**Hypothesis.** Our paper already shows base open models *linearly represent* the
ambiguity axis but don't act on it. This asks whether SFT changed the
**representation** (capability) or only the **output policy** (formatting):
- SFT peak axis-decoding accuracy **≫** base peak → SFT *sharpened the
  representation* (capability gain).
- SFT peak **≈** base peak (behavior changed, representation did not) → SFT only
  rewired the **readout/policy** (formatting-like) — the most likely outcome, and
  itself a clean, publishable result that makes the "represents-but-doesn't-act →
  now-acts" story causal-adjacent.

**Run.** `probe_sft_vs_base.py` (self-contained, no kit env needed):

```bash
python experiments/sft_vs_rl/probe_sft_vs_base.py \
    --instances data/final/fininteract_v1.jsonl \
    --base-model Qwen/Qwen3-4B-Instruct-2507 \
    --adapter   outputs/sft \
    --rep last \
    --out  experiments/sft_vs_rl/probe_sft_vs_base.json \
    --fig  experiments/sft_vs_rl/probe_sft_vs_base.png
```

It extracts per-layer last-token activations for the **bare ambiguous question**
on the entity_scope-vs-metric_definition split, trains a per-layer linear probe
(stratified CV) on base and on base+SFT-adapter, and reports per-layer + peak
axis-decoding accuracy for both, with the delta.

**Report back:** `probe_sft_vs_base.json` and the figure.

---

## Caveats to keep attached to any result
- n=51 test; **all judges/teacher are local Qwen2.5-32B/72B-AWQ** (weaker than
  GPT-4o, and a sibling of the policy).
- GRPO in the original run was 15–30 steps — not converged.
- The probe (Exp 3) is correlational on the *representation*; it shows whether the
  decodable axis signal moved, not a causal intervention.

## Scope note
Experiment 3 is small and folds into the paper's mechanistic section (§6.6) as one
figure. Experiments 1–2 are a **stronger standalone follow-up** ("does the reward
leak hide RL's value, and is axis-asking learned as capability or format?") rather
than something to squeeze into the benchmark paper.
