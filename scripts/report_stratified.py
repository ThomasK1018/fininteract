#!/usr/bin/env python3
"""Stratified reporting for the reviewer response (per-language and per-category).

Reads every data/results/eval_*.jsonl (plus or_eval_*.jsonl) and prints, per model and
mode, accuracy against the intended reading, default-capture, interaction rate, and the
single-gold inflation ratio (default / intended, answer+search mode), split by language
(EN / ZH) and by primary category. Also prints the inflation distribution over all
full-scale models so the "3.1x" headline can be reported as a range, not one cell.

Usage:
  python scripts/report_stratified.py [--models gpt-5 gpt-4o ...] [--md out.md]
"""
import argparse, glob, json, collections, math, random
from pathlib import Path

FULL_SCALE = ["gpt-5", "gpt-4o", "gpt-5-mini", "qwen3-4b", "qwen3-8b", "qwen3-14b",
              "qwen3-32b", "qwen3-30b-a3b", "qwen3p5-35b-a3b"]
MODES = ["answer-only", "answer+search", "answer+search+interact"]


def load(paths):
    rows = []
    for f in paths:
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("error") or r.get("forced_n", 0):
                continue
            if r.get("user_sim") not in (None, "llm"):
                continue
            rows.append(r)
    # de-duplicate on (model, mode, instance): keep the last occurrence
    seen = {}
    for r in rows:
        seen[(r["model"], r["mode"], r["instance_id"])] = r
    return list(seen.values())


def boot_ci(vals, n=2000, seed=0):
    if not vals:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    m = len(vals)
    bs = sorted(sum(rng.choice(vals) for _ in range(m)) / m for _ in range(n))
    return (100 * bs[int(0.025 * n)], 100 * bs[int(0.975 * n)])


def stats(rs):
    n = len(rs)
    if n == 0:
        return dict(n=0)
    acc = sum(bool(r["correct"]) for r in rs) / n
    dft = sum(bool(r.get("default_captured")) for r in rs) / n
    ir = sum(bool(r.get("interacted")) for r in rs) / n
    lo, hi = boot_ci([int(bool(r["correct"])) for r in rs])
    return dict(n=n, acc=100 * acc, dft=100 * dft, ir=100 * ir, lo=lo, hi=hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=FULL_SCALE)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    # Canonical main-table runs only (LLM simulator, thinking-off open models); the
    # protocol/simulator ablation files (think_*, simrobust, freeform, askonce, ...) are
    # reported separately and must not overwrite the canonical rows.
    paths = (sorted(glob.glob("data/results/eval_open_*.jsonl"))
             + sorted(glob.glob("data/results/eval_gpt5_gpt4o.jsonl"))
             + sorted(glob.glob("data/results/eval_gpt5mini*.jsonl"))
             + sorted(glob.glob("data/results/or_eval_*.jsonl")))
    rows = [r for r in load(paths) if r["mode"] in MODES]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["model"], r["mode"])].append(r)
    models = [m for m in a.models if any((m, mo) in by for mo in MODES)]
    out = []

    def emit(s=""):
        out.append(s)
        print(s)

    emit("## Per-language results (accuracy vs intended, default-capture, interaction rate)")
    emit("")
    emit("| Model | Mode | n EN | EN Acc [95% CI] | EN Default | EN IR | n ZH | ZH Acc [95% CI] | ZH Default | ZH IR |")
    emit("|---|---|---|---|---|---|---|---|---|---|")
    for m in models:
        for mo in MODES:
            rs = by.get((m, mo), [])
            if not rs:
                continue
            en = stats([r for r in rs if r.get("language") == "en"])
            zh = stats([r for r in rs if r.get("language") == "zh"])
            def fmt(s):
                if s["n"] == 0:
                    return "0 | -- | -- | --"
                return f"{s['n']} | {s['acc']:.1f} [{s['lo']:.0f}, {s['hi']:.0f}] | {s['dft']:.1f} | {s['ir']:.0f}"
            emit(f"| {m} | {mo} | {fmt(en)} | {fmt(zh)} |")
    emit("")
    emit("## Single-gold inflation per language (answer+search: default-capture / intended accuracy)")
    emit("")
    emit("| Model | EN intended | EN default | EN ratio | ZH intended | ZH default | ZH ratio | All ratio |")
    emit("|---|---|---|---|---|---|---|---|")
    ratios = {}
    for m in models:
        rs = by.get((m, "answer+search"), [])
        if not rs:
            continue
        cells = []
        for lang in ("en", "zh", None):
            sub = [r for r in rs if lang is None or r.get("language") == lang]
            s = stats(sub)
            ratio = (s["dft"] / s["acc"]) if s["n"] and s["acc"] > 0 else float("nan")
            if lang is None:
                ratios[m] = ratio
                cells.append(f"{ratio:.1f}x" if not math.isnan(ratio) else "n/a (acc 0)")
            else:
                cells.append(f"{s['acc']:.1f} | {s['dft']:.1f} | " + (f"{ratio:.1f}x" if not math.isnan(ratio) else "n/a"))
        emit(f"| {m} | " + " | ".join(cells) + " |")
    finite = sorted(v for v in ratios.values() if not math.isnan(v))
    if finite:
        emit("")
        emit(f"Inflation ratio over {len(finite)} full-scale models with non-zero intended accuracy: "
             f"min {finite[0]:.1f}x, median {finite[len(finite)//2]:.1f}x, max {finite[-1]:.1f}x; "
             f"models where default-capture exceeds intended accuracy: "
             f"{sum(v > 1 for v in finite)}/{len(finite)}.")
    emit("")
    emit("## Per-category results, +interact mode (accuracy vs intended / interaction rate)")
    emit("")
    cats = ["entity_scope", "metric_definition", "temporal_scope", "recognition_policy"]
    emit("| Model | " + " | ".join(f"{c} (n)" for c in cats) + " |")
    emit("|---|" + "---|" * len(cats))
    for m in models:
        rs = by.get((m, "answer+search+interact"), [])
        if not rs:
            continue
        cells = []
        for c in cats:
            sub = [r for r in rs if (r.get("axes") or [None])[0] == c]
            s = stats(sub)
            cells.append(f"{s['acc']:.1f} / IR {s['ir']:.0f} ({s['n']})" if s["n"] else "--")
        emit(f"| {m} | " + " | ".join(cells) + " |")
    emit("")
    emit("## Elicitation gap per language (context-oracle ceiling vs +interact)")
    emit("")
    ceil = collections.defaultdict(list)
    for f in glob.glob("data/results/eval_ceiling_*.jsonl"):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                ceil[r["model"]].append(r)
    emit("| Model | EN ceiling | EN +interact | EN gap | ZH ceiling | ZH +interact | ZH gap |")
    emit("|---|---|---|---|---|---|---|")
    for m in models:
        if m not in ceil:
            continue
        cells = []
        for lang in ("en", "zh"):
            c = stats([r for r in ceil[m] if r.get("language") == lang])
            i = stats([r for r in by.get((m, "answer+search+interact"), []) if r.get("language") == lang])
            if c["n"] and i["n"]:
                cells.append(f"{c['acc']:.1f} | {i['acc']:.1f} | {c['acc'] - i['acc']:.1f}")
            else:
                cells.append("-- | -- | --")
        emit(f"| {m} | " + " | ".join(cells) + " |")
    if a.md:
        Path(a.md).write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"\nwrote {a.md}")


if __name__ == "__main__":
    main()
