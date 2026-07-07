#!/usr/bin/env python3
"""
Characterize each ambiguity axis on two proposed conditions for co-evolution:
  SOURCE-ABUNDANCE  = corpus supply the data arm can draw on (candidate_axes)
  LEARNABILITY      = scale-slope of Targeting (AxisHit@1) across the model sweep
Cross-reference against the two run outcomes (entity GO, recognition NO-GO) to see
whether (abundance x learnability) predicts co-evolution success for the other axes.
"""
import json, math
from collections import defaultdict
from pathlib import Path

AXES = ["entity_scope", "metric_definition", "recognition_policy",
        "temporal_scope", "filing_vintage"]
SIZES = [("0.6b", 0.6), ("1.7b", 1.7), ("4b", 4), ("8b", 8), ("14b", 14), ("32b", 32)]
R = "data/results"


def first_ax(r):
    ax = r.get("true_axes") or r.get("axes") or ([r.get("axis")] if r.get("axis") else [])
    return ax[0] if ax else None


# --- 1. SOURCE ABUNDANCE: passages whose candidate_axes contain / lead with the axis
prim = defaultdict(int); anyc = defaultdict(int); tot = 0
for l in open("data/sources/passages.jsonl"):
    d = json.loads(l); tot += 1
    ca = d.get("candidate_axes") or []
    if ca:
        prim[ca[0]] += 1
    for a in set(ca):
        anyc[a] += 1

# --- 2. HUMAN instances per axis (frozen benchmark)
human = defaultdict(int)
for l in open("data/final/fininteract_v1.jsonl"):
    for a in (json.loads(l).get("axes") or []):
        human[a] += 1

# --- 3. LEARNABILITY: per-axis AxisHit@1 (over askers) at each model size
def axishit_by_size():
    out = defaultdict(dict)  # axis -> {size: (rate, n_askers)}
    for tag, _ in SIZES:
        f = Path(f"{R}/eval_open_qwen3-{tag}.jsonl")
        if not f.exists():
            continue
        hit = defaultdict(int); ask = defaultdict(int)
        for l in open(f):
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("mode") != "answer+search+interact":
                continue
            hs = r.get("axis_hits") or []
            if not hs:
                continue
            a = first_ax(r)
            ask[a] += 1
            hit[a] += 1 if hs[0].get("is_hit") else 0
        for a in AXES:
            if ask[a]:
                out[a][tag] = (hit[a] / ask[a], ask[a])
    return out

byhit = axishit_by_size()


def slope(axis):
    """OLS slope of AxisHit vs log10(params); None if <3 points."""
    pts = [(math.log10(p), byhit[axis][t][0]) for t, p in SIZES if t in byhit.get(axis, {})]
    if len(pts) < 3:
        return None, len(pts)
    n = len(pts); sx = sum(x for x, _ in pts); sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts); sxy = sum(x * y for x, y in pts)
    d = n * sxx - sx * sx
    return ((n * sxy - sx * sy) / d if d else 0.0), n


print(f"corpus passages = {tot}\n")
hdr = f"{'axis':20} {'prim':>5} {'anyC':>5} {'human':>5} | {'4B':>5} {'32B':>5} {'slope':>7} {'npts':>4}"
print(hdr); print("-" * len(hdr))
rows = {}
for a in AXES:
    s, npts = slope(a)
    h4 = byhit.get(a, {}).get("4b", (None,))[0]
    h32 = byhit.get(a, {}).get("32b", (None,))[0]
    f = lambda x: f"{x:.2f}" if isinstance(x, float) else "  -"
    rows[a] = dict(prim=prim[a], anyc=anyc[a], human=human[a], h4=h4, h32=h32,
                   slope=s, npts=npts)
    ss = f"{s:+.2f}" if s is not None else "   -"
    print(f"{a:20} {prim[a]:>5} {anyc[a]:>5} {human[a]:>5} | {f(h4):>5} {f(h32):>5} {ss:>7} {npts:>4}")

# --- 4. 2x2 placement + prediction
print("\n=== (abundance x learnability) 2x2 ===")
AB = 100   # >=100 primary passages = abundant
for a in AXES:
    r = rows[a]
    abundant = r["prim"] >= AB
    learn = (r["slope"] is not None and r["slope"] > 0.10)
    # temporal special-case: already-solved (high at all scales) -> co-evo moot
    note = ""
    if a == "temporal_scope":
        note = " (already solved zero-shot; co-evo moot)"
    if a == "filing_vintage":
        note = " (0 human instances; untestable)"
    verdict = "GO-predicted" if (abundant and learn) else "NO-GO-predicted"
    print(f"  {a:20} abundant={str(abundant):5} learnable={str(learn):5} -> {verdict}{note}")
print("\nobserved: entity_scope=GO, recognition_policy=NO-GO")
