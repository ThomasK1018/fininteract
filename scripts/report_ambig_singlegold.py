"""Exp 4b: AmbigQA single-gold vs any-valid grading (reviewer point 4).

Any-valid grading (credit for ANY defensible reading) is more permissive than literal
single-gold grading (credit only for one predetermined intended reading) that a
conventional benchmark applies. This re-analyzes existing illusion runs to report BOTH,
and the inflation any-valid grading would introduce, plus a random-gold robustness variant.

Pure re-analysis of saved rows (fields: intended_correct, any_valid, default_captured).
No API calls.

Usage:
  python scripts/report_ambig_singlegold.py
"""
import json, collections
from pathlib import Path

FILES = {
    "gpt-4o-mini (longest-gold)": "data/general_domain/illusion_gpt4omini.jsonl",
    "gpt-4o (longest-gold)":      "data/general_domain/illusion_gpt4o.jsonl",
    "random-gold (hash)":         "data/general_domain/robust_hash.jsonl",
}
MODES = ["answer-only", "oracle", "interact"]


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def rate(rows, field):
    return 100.0 * sum(bool(r.get(field)) for r in rows) / len(rows) if rows else float("nan")


def main():
    print("=" * 92)
    print("EXP 4b -- Single-gold (intended) vs any-valid grading on AmbigQA")
    print("=" * 92)
    print(f"{'run':28s} {'mode':12s} {'n':>4s} {'single-gold':>11s} {'any-valid':>10s} "
          f"{'inflation':>10s} {'dflt-cap':>9s}")
    for label, path in FILES.items():
        if not Path(path).exists():
            print(f"{label:28s}  (missing {path})")
            continue
        rows = load(path)
        by_mode = collections.defaultdict(list)
        for r in rows:
            by_mode[r.get("mode")].append(r)
        for mode in MODES:
            rs = by_mode.get(mode)
            if not rs:
                continue
            sg = rate(rs, "intended_correct")
            av = rate(rs, "any_valid")
            dc = rate(rs, "default_captured")
            print(f"{label:28s} {mode:12s} {len(rs):4d} {sg:10.1f}% {av:9.1f}% "
                  f"{av-sg:+9.1f} {dc:8.1f}%")
        print()

    print("Reading:")
    print(" - 'single-gold' = accuracy crediting only the ONE predetermined intended reading")
    print("   (what a conventional benchmark reports). 'any-valid' credits any defensible")
    print("   reading. 'inflation' = how many points any-valid grading would over-report.")
    print(" - random-gold (hash) designates a RANDOM reading as gold instead of the longest;")
    print("   a similar single-gold accuracy there shows the illusion is not an artifact of")
    print("   WHICH reading we pick. Human-majority-gold needs annotators (see Exp 4a sheet).")


if __name__ == "__main__":
    main()
