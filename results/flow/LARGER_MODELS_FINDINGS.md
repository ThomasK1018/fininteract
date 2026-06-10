# Larger-model mechanistic case study: Qwen3-30B-A3B vs Qwen3.5-35B-A3B

Extends the 4B activation case study (`baseline_qwen3-4b-instruct-2507/`, `qwen3.5-4b/`) to the
larger MoE models. Run on 8x A100-40GB in bf16 (HF transformers `output_hidden_states`,
`device_map=auto`). Behavioral runner used a local 32B-AWQ judge for sim/grader (no API cost).
Generation-flow used a single-forward variant (`scripts/flow_generate_fast.py`) because the
original per-step `generate(output_hidden_states=True)` is pathologically slow on sharded models.

**Headline: both models internally represent the ambiguity equally; Qwen3.5 *acts* on it, Qwen3 does not.**

| Metric | Qwen3-30B-A3B (49 layers) | Qwen3.5-35B-A3B (41 layers) |
|--------|---------------------------|-----------------------------|
| Ambiguity-detection probe (peak) | ~1.00 (saturated; partly context-presence) | ~1.00 (saturated) |
| Axis-decoding entity-vs-metric | 0.796 @ layer 13 (~27% depth) | 0.803 @ layer 35 (~85% depth) |
| Depth-flow separation peak layer | 48/49 | 40/41 |
| Behavioral interaction rate | 7% (12/173) | 90% (156/173) |
| **Behavioral-internal MISMATCH** (represents-ambiguous but did NOT ask) | **93.1%** | **9.8%** |
| Generation recognition-onset | none (never clarifies) | token 42.5 (clarifies) |
| Figures | 4 (no onset fig) | 5 (full set incl recognition_onset) |

## Reading it
- **Recognition is present in both** (detection ~1.0, axis-type ~0.80) at similar strength.
  Qwen3.5 encodes the ambiguity *type* notably later in depth (~85% vs ~27%).
- **Action differs drastically.** Qwen3-30B exhibits the "represents-but-doesn't-act" failure
  (93.1%): it internally knows the question is ambiguous yet answers directly 93% of the time.
  Qwen3.5-35B closes this to 9.8% — it asks 90% of the time, and therefore is the only one of the
  two that produces a `fig_recognition_onset.png` (you can't align an ask that never happens).
- => Across Qwen3 -> Qwen3.5 the *recognition* is preserved while the *action bottleneck is fixed* —
  the causal story this case study is built to expose, now observed across model generations.

## Caveats
- Diff-in-means is correlational; detection probe is partly trivial (context-presence) at this scale —
  the substantive static signal is the **axis-decoding** probe.
- Qwen3-235B-A22B was **not run**: A100 (compute 8.0) can't run its FP8 (needs >=8.9), and the
  AWQ-MoE Triton dequant kernel was broken in the tested transformers; the reliable bf16+bnb-4bit
  path (470GB) was deferred.

## Artifacts (this directory tree)
- `qwen3-30b-a3b-instruct-2507/` and `qwen3.5-35b-a3b/`, each with:
  `probes_{bare,enum,mismatch}.json`, `flow_report.json`, `genflow.npz` (+`.gen.jsonl`),
  `behavior_*.jsonl`, and `figs/`.
- Activation tensors (`acts_bare.npz`, `acts_enum.npz`, ~140-176MB each) are intentionally omitted;
  regenerate with `scripts/probe_activations.py --model <id> --prompt-mode {bare,enumerate}`.
