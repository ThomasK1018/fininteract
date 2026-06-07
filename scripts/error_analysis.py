"""
Post-hoc error taxonomy classification for FinInteract evaluation results.

Error types (E1–E7):
  E1 Ambiguity blindness    — model never interacted, answered wrong
  E2 Wrong-axis clarification — model asked but targeted the wrong axis, answered wrong
  E3 Generic clarification  — model asked vague/non-axis questions, answered wrong
  E4 Retrieval dominance    — model searched but skipped interaction, answered wrong
  E5 Clarification misuse   — model asked a correct-axis question but still answered wrong
  E6 Evidence grounding     — template-oracle correct but agent incorrect (evidence/math failure)
  E7 Over-interaction       — answered correctly but asked too many questions (low DisE+)

Usage:
    python scripts/error_analysis.py \\
        --results data/results/gpt-5_interact.jsonl \\
        --template-oracle-results data/results/gpt-5_template-oracle.jsonl

The --template-oracle-results file is used to identify E6 (grounding errors): instances
where the template-oracle mode answered correctly but the agent mode did not.
"""

import json
import argparse
import statistics
from pathlib import Path
from collections import Counter, defaultdict


ERROR_TYPES = {
    "E1": "Ambiguity blindness       — no interaction, wrong answer",
    "E2": "Wrong-axis clarification  — asked, targeted wrong axis, wrong answer",
    "E3": "Generic clarification     — asked, no axis target, wrong answer",
    "E4": "Retrieval dominance       — searched but no interact, wrong answer",
    "E5": "Clarification misuse      — asked correct axis, still wrong answer",
    "E6": "Evidence grounding error  — template-oracle correct, agent wrong",
    "E7": "Over-interaction          — correct but DisE+ < 0.5 (too many asks)",
}


def classify_error(result: dict,
                   template_correct: dict[str, bool] | None = None) -> list[str]:
    """
    Classify a single evaluation result into zero or more error types.
    Returns a list of error type codes (may overlap for ambiguous failures).

    Args:
        result: one result record from evaluate.py output
        template_correct: map from instance_id → bool for template-oracle results
    """
    correct    = result.get("correct", False)
    n_asks     = result.get("n_asks", 0)
    n_searches = result.get("n_searches", 0)
    axis_hits  = result.get("axis_hits", [])
    h0         = result.get("h0", 1.0)

    errors = []

    if not correct:
        # E1: No interaction at all
        if n_asks == 0:
            errors.append("E1")

        # E2/E3 — asked at least once, classify by axis quality
        if n_asks > 0:
            any_hit     = any(h.get("is_hit", False)    for h in axis_hits)
            any_generic = any(h.get("is_generic", False) for h in axis_hits)
            any_wrong   = any(h.get("is_wrong_axis", False) for h in axis_hits)

            if any_wrong and not any_hit:
                errors.append("E2")
            if any_generic and not any_hit:
                errors.append("E3")

            # E4: searched but never interacted
            if n_searches > 0 and n_asks == 0:
                errors.append("E4")

            # E5: on-axis question asked, but still wrong
            if any_hit:
                errors.append("E5")

        # E6: template-oracle answered this correctly but agent did not
        if template_correct is not None:
            iid = result.get("instance_id")
            if template_correct.get(iid, False):
                errors.append("E6")

    else:
        # Correct answers can still be E7 if interaction was wasteful
        n_asks_max1 = max(n_asks, 1)
        dise_plus = h0 / n_asks_max1
        if dise_plus < 0.5 and n_asks > 0:
            errors.append("E7")

    return errors if errors else (["correct"] if correct else ["E1"])


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def diagnostic_map(results: list[dict]) -> dict:
    """
    Build the metric-to-error-type diagnostic mapping from evaluation metrics.
    Returns a dict of diagnostic observations with interpretations.
    """
    n = len(results)
    if n == 0:
        return {}

    acc   = sum(1 for r in results if r.get("correct")) / n
    ir    = sum(1 for r in results if r.get("n_asks", 0) > 0) / n
    ah1   = [r.get("axis_hit_rate", 0) for r in results if r.get("n_asks", 0) > 0]
    any_ah = [r for r in results if any(h.get("is_hit") for h in r.get("axis_hits", []))]
    wrong_ax = [r for r in results if any(h.get("is_wrong_axis") for h in r.get("axis_hits", []))]
    generic  = [r for r in results if any(h.get("is_generic") for h in r.get("axis_hits", []))]

    diag = {}

    # Low IR + low accuracy → ambiguity blindness (E1)
    if ir < 0.3 and acc < 0.3:
        diag["low_ir_low_acc"] = (
            f"IR={ir:.0%}, Acc={acc:.0%} → E1 (ambiguity blindness): "
            "model rarely interacts and rarely answers correctly; primary bottleneck is "
            "failure to recognize financial ambiguity."
        )

    # High IR + low AxisHit@1 → model knows to ask but doesn't know what
    mean_ah1 = statistics.mean(ah1) if ah1 else None
    if ir > 0.5 and mean_ah1 is not None and mean_ah1 < 0.4:
        diag["high_ir_low_axishit1"] = (
            f"IR={ir:.0%}, AxisHit@1={mean_ah1:.0%} → E2/E3: "
            "model interacts frequently but targets wrong or generic axes; "
            "axis-conditioned clarification needed."
        )

    # High AxisHit@1 + low accuracy → grounding/retrieval failure after correct question
    if mean_ah1 is not None and mean_ah1 > 0.6 and acc < 0.4:
        diag["high_axishit_low_acc"] = (
            f"AxisHit@1={mean_ah1:.0%}, Acc={acc:.0%} → E5/E6: "
            "model asks the right question but fails to extract or compute the correct answer; "
            "grounding and numerical reasoning are the residual bottleneck."
        )

    # High GenericAskRate → low ambiguity awareness
    gen_rate = len(generic) / n
    if gen_rate > 0.3:
        diag["high_generic_ask"] = (
            f"GenericAskRate={gen_rate:.0%} → E3: "
            "model shows uncertainty but lacks financial-domain structured clarification ability."
        )

    # High WrongAxisRate → confidently misdirected
    wrong_rate = len(wrong_ax) / n
    if wrong_rate > 0.2:
        diag["high_wrong_axis"] = (
            f"WrongAxisRate={wrong_rate:.0%} → E2: "
            "model asks axis-specific questions but misidentifies the source of ambiguity."
        )

    return diag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="JSONL file of evaluate.py results (interact or axis-aware mode)")
    parser.add_argument("--template-oracle-results",
                        help="JSONL file of template-oracle results (used to detect E6)")
    parser.add_argument("--out", help="Optional JSONL output with per-instance error labels")
    args = parser.parse_args()

    results = _load_jsonl(Path(args.results))

    template_correct: dict[str, bool] | None = None
    if args.template_oracle_results:
        tpl = _load_jsonl(Path(args.template_oracle_results))
        template_correct = {r["instance_id"]: r.get("correct", False) for r in tpl}

    # Per-instance classification
    labeled = []
    error_counts: Counter = Counter()
    for r in results:
        errors = classify_error(r, template_correct)
        labeled.append({**r, "error_types": errors})
        for e in errors:
            if e != "correct":
                error_counts[e] += 1

    n = len(results)
    n_correct = sum(1 for r in results if r.get("correct"))

    print(f"\n{'='*60}")
    print(f"Error Analysis: {Path(args.results).name}  ({n} instances)")
    print(f"{'='*60}")
    print(f"Accuracy: {n_correct}/{n} = {n_correct/n:.1%}")
    print(f"\nError distribution:")
    for code in sorted(error_counts):
        cnt = error_counts[code]
        desc = ERROR_TYPES.get(code, code)
        print(f"  {code}  {cnt:>3}/{n}  ({cnt/n:.0%})   {desc}")

    # Diagnostic mapping
    diag = diagnostic_map(results)
    if diag:
        print(f"\nDiagnostic observations:")
        for obs in diag.values():
            print(f"  • {obs}")

    # Per-axis breakdown
    axis_errors: dict[str, Counter] = defaultdict(Counter)
    for r in labeled:
        primary_axis = (r.get("axes") or ["?"])[0]
        for e in r.get("error_types", []):
            if e != "correct":
                axis_errors[primary_axis][e] += 1
    if axis_errors:
        print(f"\nError breakdown by axis:")
        for ax, cnt_map in sorted(axis_errors.items()):
            total_ax = sum(cnt_map.values())
            dominant = cnt_map.most_common(1)[0][0] if cnt_map else "-"
            print(f"  {ax:<30} n={total_ax}  dominant={dominant}")

    if args.out:
        out_path = Path(args.out)
        with out_path.open("w") as f:
            for r in labeled:
                f.write(json.dumps(r) + "\n")
        print(f"\nLabeled results written to {out_path}")


if __name__ == "__main__":
    main()
