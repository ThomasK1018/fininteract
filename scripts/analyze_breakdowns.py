"""Breakdowns + bootstrap CIs + significance tests for the FinInteract leaderboard.

Reads every data/results/eval_*.jsonl (so it picks up new model runs automatically),
and reports, per model x mode:
  - accuracy with bootstrap 95% CI
  - EN vs ZH split
  - per-primary-axis accuracy with n
  - default-capture vs intended (the single-gold-illusion gap)
plus paired-bootstrap significance tests for the headline comparisons.

No API calls; pure re-analysis of saved eval rows. Addresses reviewer Tier-2 gaps
(confidence intervals, significance, EN/ZH split, per-axis n).
"""
import json, glob, random, collections
random.seed(0)

MODES = ["answer-only", "answer+search", "answer+search+interact"]
AXES = ["temporal_scope", "metric_definition", "entity_scope", "filing_vintage", "recognition_policy"]

def load():
    rows = []
    for f in sorted(glob.glob("data/results/eval_*.jsonl")):
        for l in open(f):
            l = l.strip()
            if l:
                try: rows.append(json.loads(l))
                except json.JSONDecodeError: pass
    # de-dup on (model, mode, instance_id, forced_n) keeping last
    seen = {}
    for r in rows:
        k = (r.get("model"), r.get("mode"), r.get("instance_id"), r.get("forced_n", 0))
        seen[k] = r
    return list(seen.values())

def acc(rs):
    return 100.0 * sum(bool(r["correct"]) for r in rs) / len(rs) if rs else float("nan")

def boot_ci(rs, stat=acc, B=2000):
    if not rs: return (float("nan"), float("nan"))
    n = len(rs); vals = []
    idx = list(range(n))
    for _ in range(B):
        samp = [rs[random.choice(idx)] for _ in range(n)]
        vals.append(stat(samp))
    vals.sort()
    return (vals[int(0.025 * B)], vals[int(0.975 * B)])

def paired_boot_pvalue(rs_a, rs_b, B=5000):
    """Two-sided paired bootstrap on accuracy difference, matched by instance_id."""
    da = {r["instance_id"]: bool(r["correct"]) for r in rs_a}
    db = {r["instance_id"]: bool(r["correct"]) for r in rs_b}
    keys = sorted(set(da) & set(db))
    if not keys: return float("nan"), 0
    diffs = [(da[k] - db[k]) for k in keys]
    obs = sum(diffs) / len(diffs)
    # center under H0 by subtracting observed mean, count |boot| >= |obs|
    n = len(diffs); idx = list(range(n)); ge = 0
    for _ in range(B):
        m = sum(diffs[random.choice(idx)] for _ in range(n)) / n
        if abs(m - obs) >= abs(obs): ge += 1
    return ge / B, len(keys)

rows = load()
by = collections.defaultdict(list)
for r in rows:
    if r.get("forced_n", 0) == 0:
        by[(r["model"], r["mode"])].append(r)
models = sorted({m for (m, _) in by})
print(f"loaded {len(rows)} rows | models: {models}\n")

print("="*86)
print("ACCURACY WITH BOOTSTRAP 95% CI  (forced_n=0)")
print("="*86)
print(f"{'model':14s} {'mode':24s} {'n':>4s} {'acc%':>6s} {'95% CI':>16s} {'default%':>9s}")
for m in models:
    for mode in MODES:
        rs = by.get((m, mode), [])
        if not rs: continue
        lo, hi = boot_ci(rs)
        dcap = 100.0 * sum(bool(r.get("default_captured")) for r in rs) / len(rs)
        print(f"{m:14s} {mode:24s} {len(rs):4d} {acc(rs):6.1f} [{lo:5.1f},{hi:5.1f}] {dcap:9.1f}")

print("\n" + "="*86)
print("EN vs ZH  (accuracy by language)")
print("="*86)
print(f"{'model':14s} {'mode':24s} {'EN n':>5s} {'EN%':>6s} {'ZH n':>5s} {'ZH%':>6s}")
for m in models:
    for mode in MODES:
        rs = by.get((m, mode), [])
        if not rs: continue
        en = [r for r in rs if r.get("language") == "en"]
        zh = [r for r in rs if r.get("language") == "zh"]
        print(f"{m:14s} {mode:24s} {len(en):5d} {acc(en):6.1f} {len(zh):5d} {acc(zh):6.1f}")

print("\n" + "="*86)
print("PER-PRIMARY-AXIS ACCURACY  (interact mode)  [primary axis = axes[0]]")
print("="*86)
hdr = f"{'model':14s} " + " ".join(f"{a.split('_')[0][:6]:>8s}" for a in AXES)
print(hdr)
for m in models:
    rs = by.get((m, "answer+search+interact"), [])
    if not rs: continue
    cells = []
    for a in AXES:
        sub = [r for r in rs if (r.get("axes") or [None])[0] == a]
        cells.append(f"{acc(sub):5.1f}({len(sub):2d})" if sub else "   -    ")
    print(f"{m:14s} " + " ".join(f"{c:>8s}" for c in cells))

print("\n" + "="*86)
print("SIGNIFICANCE (paired bootstrap, two-sided p)")
print("="*86)
def cmp(m, mode_a, mode_b, label):
    ra, rb = by.get((m, mode_a), []), by.get((m, mode_b), [])
    if not ra or not rb: return
    p, n = paired_boot_pvalue(ra, rb)
    print(f"  {m:10s} {label:34s} Δacc={acc(ra)-acc(rb):+5.1f}  p={p:.4f} (n={n})")
for m in models:
    cmp(m, "answer+search+interact", "answer+search", "interact vs search")
    cmp(m, "answer+search", "answer-only", "search vs answer-only")
print("\n(intended-vs-default gap is within-row: see 'default%' column vs acc%)")
