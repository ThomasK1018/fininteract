"""
Audit accepted instances for the R14 grader-tolerance gap violation.

Flags (and optionally removes) instances where the intended and default answers fall
within the grader's numeric tolerance — i.e., a model resolving to the WRONG
interpretation would still be graded correct, so the instance cannot discriminate.

These are instances accepted BEFORE the R14 check was added to construct_instances.py.

Usage:
    # Report only
    python scripts/audit_weak_instances.py --in data/constructed/instances.jsonl

    # Move flagged instances out to a quarantine file and rewrite the main file
    python scripts/audit_weak_instances.py --in data/constructed/instances.jsonl --fix
"""

import json
import re
import argparse
from pathlib import Path


def parse_numeric(s) -> float | None:
    s = str(s).strip()
    m = re.search(r'-?[\d,]+(?:\.\d+)?', s.replace(",", ""))
    if not m:
        m = re.search(r'-?[\d.]+', s)
        if not m:
            return None
    val = float(m.group().replace(",", ""))
    sl = s.lower()
    if "亿" in s:
        val *= 1e8
    elif "万" in s:
        val *= 1e4
    elif "b" in sl or "billion" in sl:
        val *= 1e9
    elif ("m" in sl or "million" in sl) and "%" not in s:
        val *= 1e6
    elif "k" in sl and "%" not in s:
        val *= 1e3
    return val


def is_weak(inst: dict) -> tuple[bool, str]:
    a = parse_numeric(inst.get("answer"))
    d = parse_numeric(inst.get("default_answer"))
    if a is None or d is None:
        return False, ""
    gap = abs(a - d)
    is_pct = "%" in str(inst.get("answer")) and "%" in str(inst.get("default_answer"))
    if is_pct:
        if gap < 1.5:
            return True, f"{gap:.2f}pt gap (rate answer, grader tol ~1pt)"
    else:
        rel = gap / abs(a) if a else None
        if rel is not None and rel < 0.03:
            return True, f"{rel:.1%} relative gap (value answer, grader tol ~1%)"
    return False, ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=Path,
                   default=Path("data/constructed/instances.jsonl"))
    p.add_argument("--fix", action="store_true",
                   help="Quarantine flagged instances and rewrite the main file")
    args = p.parse_args()

    instances = [json.loads(l) for l in args.inp.open() if l.strip()]
    weak, keep = [], []
    for inst in instances:
        flag, reason = is_weak(inst)
        if flag:
            weak.append((inst, reason))
        else:
            keep.append(inst)

    print(f"Audited {len(instances)} instances")
    print(f"Weak (R14 violation): {len(weak)}")
    print(f"Discriminating:       {len(keep)}\n")
    for inst, reason in weak:
        print(f"  {inst['instance_id']} [{inst.get('ticker','?')}] "
              f"{(inst.get('axes') or ['?'])[0]:<18} "
              f"A={inst.get('answer')} default={inst.get('default_answer')}  → {reason}")

    if args.fix and weak:
        quarantine = args.inp.with_suffix(".weak.jsonl")
        with quarantine.open("w", encoding="utf-8") as f:
            for inst, reason in weak:
                f.write(json.dumps({**inst, "_weak_reason": reason}, ensure_ascii=False) + "\n")
        with args.inp.open("w", encoding="utf-8") as f:
            for inst in keep:
                f.write(json.dumps(inst, ensure_ascii=False) + "\n")
        print(f"\nMoved {len(weak)} weak instances → {quarantine}")
        print(f"Rewrote {args.inp} with {len(keep)} discriminating instances")


if __name__ == "__main__":
    main()
