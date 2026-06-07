"""
Extract COMPOSITIONAL multi-axis passages (entity_scope x temporal_scope) from EDGAR.

A multi-segment company that reports each segment across multiple fiscal years exposes a
segment x period grid in XBRL. From it we build a genuinely 2-axis-ambiguous question:

  "What was <Company>'s revenue?"
      intended : <segment S>, fiscal year Y1   (a specific off-default cell)
      default  : consolidated total, latest FY  (the naive reading)

The two interpretations differ on BOTH entity scope (segment vs consolidated) and temporal
scope (Y1 vs latest), so the agent must resolve two independent axes — a harder difficulty
tier than any single-axis instance. Both values are verbatim-present in the grid (so each
interpretation is groundable), and they differ by a large margin (R14-clean).

Usage:
  export EDGAR_IDENTITY="Name email@school.edu"
  python scripts/extract_multiaxis_passages.py \\
      --targets data/edgar/target_companies_b2.txt \\
      --out data/sources/edgar_multiaxis_passages.jsonl

Then: cat ... >> data/sources/edgar_passages.jsonl ; python scripts/prepare_passages.py
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("EDGAR_LOCAL_DATA_DIR", str(PROJECT_ROOT / "data" / "edgar" / ".edgar"))
os.environ.setdefault("EDGAR_CACHE_DIR", str(PROJECT_ROOT / "data" / "edgar" / ".edgar_cache"))

try:
    from edgar import Company, set_identity
except ImportError:
    sys.exit("pip install edgartools")


def _fmt(v: float) -> str:
    if abs(v) >= 1e9:
        return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v/1e6:.1f}M"
    return f"${v:.0f}"


# Labels that are aggregates/non-segments — exclude from the "segment" axis.
_SKIP_LABELS = ("total", "net sales", "revenues", "consolidated", "point in time",
                "over time", "geographic", "united states", "international")


def build_passage(ticker: str, company: str, filing_date: str, grid: dict) -> dict | None:
    """grid: {segment_label: {fy_year: value}}. Returns one 2-axis passage or None."""
    # consolidated = the largest aggregate across the latest common year
    years = sorted({y for d in grid.values() for y in d})
    if len(years) < 2:
        return None
    latest = years[-1]

    # segments = non-aggregate labels present in BOTH the latest and an earlier year
    segs = {}
    for lbl, d in grid.items():
        ll = lbl.lower()
        if any(s in ll for s in _SKIP_LABELS):
            continue
        if latest in d and any(y in d for y in years[:-1]):
            segs[lbl] = d
    if not segs:
        return None

    # consolidated proxy = max total value at latest year among aggregate labels
    consol = None
    for lbl, d in grid.items():
        if "total" in lbl.lower() or "net sales" in lbl.lower() or lbl.lower() in ("revenues", "revenue"):
            if latest in d and (consol is None or abs(d[latest]) > abs(consol[1])):
                consol = (lbl, d[latest])
    if consol is None:
        # fall back: sum of segments at latest
        consol = ("consolidated total", sum(d.get(latest, 0) for d in segs.values()))

    # intended cell: largest segment at an EARLIER year (differs from default on both axes)
    earlier = years[0]
    cand = [(lbl, d[earlier]) for lbl, d in segs.items() if earlier in d and abs(d[earlier]) > 50e6]
    if not cand:
        return None
    intended_seg, intended_val = max(cand, key=lambda x: abs(x[1]))
    default_lbl, default_val = consol

    # require a material gap (R14)
    if abs(intended_val - default_val) / max(abs(default_val), 1) < 0.05:
        return None

    # entropy: |entity values| x |years|
    k_entity = len(segs) + 1          # segments + consolidated
    k_time   = len(years)
    h0 = round(math.log2(max(k_entity * k_time, 2)), 3)

    # passage text: the grid, verbatim-quotable
    lines = [f"{company} — Revenue by reportable segment and fiscal year (from the 10-K)."]
    for lbl, d in list(segs.items())[:8]:
        cells = "; ".join(f"FY{y}: {_fmt(d[y])}" for y in years if y in d)
        lines.append(f"  {lbl}: {cells}")
    lines.append(f"  {default_lbl} (consolidated): FY{latest}: {_fmt(default_val)}")
    passage_text = "\n".join(lines)

    return {
        "passage_id":       f"edgar_multi_{ticker}_{intended_seg[:10].replace(' ','')}_{earlier}",
        "source":           "edgar_multiaxis",
        "language":         "en",
        "ticker":           ticker,
        "company":          company,
        "period":           str(latest),
        "filing_type":      "10-K",
        "filing_date":      filing_date,
        "passage_text":     passage_text,
        "candidate_answer": _fmt(intended_val),
        "candidate_axes":   ["entity_scope", "temporal_scope"],
        "multi_axis":       True,
        "h0":               h0,
        "_xbrl_hint": {
            "intended_entity": intended_seg,
            "intended_period": f"fiscal year {earlier}",
            "intended_val":    _fmt(intended_val),
            "default_entity":  "consolidated total",
            "default_period":  f"fiscal year {latest}",
            "default_val":     _fmt(default_val),
        },
    }


def extract_for_ticker(ticker: str, years: list[int]) -> list[dict]:
    out = []
    try:
        co = Company(ticker)
        company = co.name
        filing = co.get_filings(form="10-K").latest(1)
        fdate = str(filing.filing_date)
        if int(fdate.split("-")[0]) not in years:
            return out
        df = filing.xbrl().facts.query().to_dataframe()
    except Exception as e:
        print(f"  [err] {ticker}: {e}")
        return out

    rev = df[(df["concept"].astype(str).str.contains("Revenue", case=False, na=False))
             & df["numeric_value"].notna() & df["is_dimensioned"]]
    grid: dict = {}
    for _, r in rev.iterrows():
        lbl = str(r["label"])[:40]
        y = str(r.get("period_end"))[:4]
        if not y.isdigit():
            continue
        grid.setdefault(lbl, {})[y] = float(r["numeric_value"])

    p = build_passage(ticker, company, fdate, grid)
    if p:
        h = p["_xbrl_hint"]
        print(f"  [multiaxis] {ticker}: intended={h['intended_entity']} {h['intended_period']} "
              f"({h['intended_val']}) vs default=consolidated {h['default_period']} ({h['default_val']})")
        out.append(p)
    return out


def read_tickers(path: Path) -> list[str]:
    out, seen = [], set()
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        t = line.split()[0].upper()
        if t not in seen:
            seen.add(t); out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=Path, default=Path("data/edgar/target_companies_b2.txt"))
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    ap.add_argument("--out", type=Path, default=Path("data/sources/edgar_multiaxis_passages.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    ident = os.environ.get("EDGAR_IDENTITY")
    if not ident:
        sys.exit("Set EDGAR_IDENTITY='Name email@school.edu'")
    set_identity(ident)

    tickers = read_tickers(args.targets)
    if args.limit:
        tickers = tickers[:args.limit]
    print(f"Scanning {len(tickers)} tickers for entity x temporal grids")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if args.out.exists():
        for l in args.out.open():
            try: existing.add(json.loads(l)["passage_id"])
            except Exception: pass

    total = 0
    with args.out.open("a", encoding="utf-8") as f:
        for t in tickers:
            print(f"[{t}]")
            for p in extract_for_ticker(t, args.years):
                if p["passage_id"] in existing:
                    continue
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
                existing.add(p["passage_id"]); total += 1

    print(f"\nDone. {total} multi-axis (entity x temporal) passages -> {args.out}")


if __name__ == "__main__":
    main()
