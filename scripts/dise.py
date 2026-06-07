"""
Disambiguation Efficiency metrics for FinInteract.

Three metrics are implemented:

1. DisE (original) = H0 / n_asks
   H0 = sum_i log2(|options_i|) — prior interpretation entropy before any interaction.
   Higher is better; max = 1.0 when agent asks exactly H0 yes/no questions.
   Weakness (flagged by AAAI reviewer): rewards efficient questioning even when the
   final answer is wrong. A model that asks one irrelevant question and answers
   incorrectly still gets DisE > 0.

2. DisE+_all (correctness-gated, all instances) = 1[correct] × H0 / max(n_asks, 1)
   Only rewards disambiguation efficiency when the agent actually answers correctly.
   max(n_asks, 1) means a correct zero-ask answer still gets H0 / 1, rewarding
   confident correct answers regardless of whether the agent asked.
   This is the primary reported metric.

3. DisE+_interact (correctness-gated, interaction-only) = 1[correct] × 1[n_asks>0] × H0 / n_asks
   Like DisE+_all but only defined when the agent actually asked at least one question.
   Specifically rewards SUCCESSFUL CLARIFICATION — a model that answers correctly
   without asking gets 0. Reported as a secondary metric alongside DisE+_all to
   show the quality of the agent's interaction behavior separately from its baseline.

Edge cases:
    n_asks == 0  →  DisE = 0.0  (agent never asked, even if H0 > 0)
                 →  DisE+_all = 1[correct] × H0 (correct no-ask = H0; wrong = 0)
                 →  DisE+_interact = 0.0 (no interaction occurred)
    H0 == 0      →  instance is unambiguous; all DisE metrics = None

Axis option counts (conservative defaults — override per instance if needed):
    1. temporal_scope:          2  (FY vs CY; or quarterly vs annual)
    2. metric_definition:       4  (GAAP / non-GAAP / organic / constant-currency)
    3. entity_scope:            3  (parent / consolidated / segment)
    4. filing_vintage:          2  (original vs amended/restated)
    5. recognition_policy:      2  (e.g., one of two treatment choices)
"""

import math
from dataclasses import dataclass, field

# Default number of plausible interpretations per axis.
# Override per instance when the actual construction data has more information.
DEFAULT_AXIS_OPTIONS: dict[str, int] = {
    "temporal_scope":      2,
    "metric_definition":   4,
    "entity_scope":        3,
    "filing_vintage":      2,
    "recognition_policy":  2,
}


@dataclass
class DiseResult:
    h0: float                       # initial interpretation entropy (bits)
    n_asks: int                     # actual number of ASK actions by the agent
    dise: float | None              # original DisE = H0/n_asks; None if h0 == 0
    dise_plus_all: float | None     # DisE+_all  = 1[correct]*H0/max(n_asks,1)
    dise_plus_interact: float | None  # DisE+_interact = 1[correct]*1[n_asks>0]*H0/n_asks
    axes_hit: list[str]             # which axes are ambiguous in this instance
    axis_options: dict[str, int]    # options count per axis used in computation
    correct: bool = False           # whether the agent's final answer was correct

    @property
    def dise_plus(self) -> float | None:
        """Alias for dise_plus_all (backward compatibility)."""
        return self.dise_plus_all


def compute_h0(axes_hit: list[str],
               axis_options: dict[str, int] | None = None) -> float:
    """Sum of log2(options) across all ambiguous axes in this instance."""
    opts = {**DEFAULT_AXIS_OPTIONS, **(axis_options or {})}
    return sum(math.log2(opts.get(ax, 2)) for ax in axes_hit)


def compute_dise(axes_hit: list[str],
                 n_asks: int,
                 axis_options: dict[str, int] | None = None,
                 correct: bool = False) -> DiseResult:
    """
    Compute DisE and DisE+ for one agent-instance interaction.

    Args:
        axes_hit:     list of ambiguous axis names for this instance
        n_asks:       number of `ask` actions the agent used
        axis_options: optional per-instance override of option counts
        correct:      whether the agent's final answer was correct

    Returns:
        DiseResult with h0, dise (original), dise_plus (correctness-gated)
    """
    opts = {**DEFAULT_AXIS_OPTIONS, **(axis_options or {})}
    h0 = compute_h0(axes_hit, opts)

    if h0 == 0.0:
        return DiseResult(h0=0.0, n_asks=n_asks, dise=None,
                          dise_plus_all=None, dise_plus_interact=None,
                          axes_hit=axes_hit, axis_options=opts, correct=correct)

    dise               = h0 / n_asks if n_asks > 0 else 0.0
    dise_plus_all      = (h0 / max(n_asks, 1)) if correct else 0.0
    dise_plus_interact = (h0 / n_asks) if (correct and n_asks > 0) else 0.0

    return DiseResult(h0=h0, n_asks=n_asks, dise=dise,
                      dise_plus_all=dise_plus_all,
                      dise_plus_interact=dise_plus_interact,
                      axes_hit=axes_hit, axis_options=opts, correct=correct)


def aggregate_dise(results: list[DiseResult]) -> dict:
    """
    Aggregate DisE across a set of instances (e.g., per model, per axis subset).

    Returns a dict with mean, median, and per-axis breakdowns.
    """
    import statistics

    valid = [r for r in results if r.dise is not None]
    if not valid:
        return {"mean_dise_plus_all": None, "mean_dise_plus_interact": None,
                "mean_dise": None, "n_valid": 0}

    scores               = [r.dise for r in valid]
    scores_plus_all      = [r.dise_plus_all      for r in valid if r.dise_plus_all      is not None]
    scores_plus_interact = [r.dise_plus_interact for r in valid if r.dise_plus_interact is not None]
    # For complexity breakdown use DisE+_all
    by_n_axes: dict[int, list[float]] = {}
    for r in valid:
        k = len(r.axes_hit)
        by_n_axes.setdefault(k, []).append(r.dise_plus_all if r.dise_plus_all is not None else 0.0)

    return {
        # Primary metric: rewards correct answers, efficient OR zero-ask
        "mean_dise_plus_all":      round(statistics.mean(scores_plus_all), 4)      if scores_plus_all      else None,
        "median_dise_plus_all":    round(statistics.median(scores_plus_all), 4)    if scores_plus_all      else None,
        # Secondary metric: rewards correct answers that required clarification
        "mean_dise_plus_interact":   round(statistics.mean(scores_plus_interact), 4)   if scores_plus_interact else None,
        "median_dise_plus_interact": round(statistics.median(scores_plus_interact), 4) if scores_plus_interact else None,
        # Original metric (reference / ablation)
        "mean_dise":    round(statistics.mean(scores), 4),
        "median_dise":  round(statistics.median(scores), 4),
        "n_valid":      len(valid),
        "n_skipped":    len(results) - len(valid),
        "by_complexity": {
            k: round(statistics.mean(vs), 4)
            for k, vs in sorted(by_n_axes.items())
        },
    }
