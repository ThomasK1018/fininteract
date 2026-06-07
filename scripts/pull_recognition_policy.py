"""
Pull recognition_policy passages from EDGAR via ASC 606 revenue-timing disaggregation.

Under ASC 606, companies disaggregate revenue by *timing of recognition* — "Revenue
Recognized at a Point in Time" vs "Revenue Recognized Over Time" — and tag both as
dimensioned XBRL facts. This is a clean, structurally-guaranteed recognition_policy
ambiguity:

  "What was <Company>'s revenue?"
     intended (point-in-time) : $39.2B   |   default (over-time) : $6.5B   [Deere FY2025]

Both values are verbatim-present in the filing's revenue-disaggregation note, so the
default-vs-intended pair is guaranteed and R14-discriminating.

Usage:
  export EDGAR_IDENTITY="Name email@school.edu"
  python scripts/pull_recognition_policy.py \\
      --targets data/edgar/target_companies_b2.txt \\
      --out data/sources/edgar_recognition_passages.jsonl

Then merge:
  cat data/sources/edgar_recognition_passages.jsonl >> data/sources/edgar_passages.jsonl
  python scripts/prepare_passages.py
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("EDGAR_LOCAL_DATA_DIR", str(PROJECT_ROOT / "data" / "edgar" / ".edgar"))
os.environ.setdefault("EDGAR_CACHE_DIR", str(PROJECT_ROOT / "data" / "edgar" / ".edgar_cache"))

try:
    from edgar import Company, set_identity
except ImportError:
    sys.exit("pip install edgartools")

POINT_RE = re.compile(r"point\s*in\s*time", re.I)
OVER_RE  = re.compile(r"over\s*time|over\s*a\s*period", re.I)


def _fmt(v: float) -> str:
    if abs(v) >= 1e9:
        return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v/1e6:.1f}M"
    return f"${v:.0f}"


def extract_for_ticker(ticker: str, years: list[int]) -> list[dict]:
    out = []
    try:
        co = Company(ticker)
    except Exception as e:
        print(f"  [err] {ticker}: {e}")
        return out
    company = co.name

    try:
        filings = co.get_filings(form="10-K")
    except Exception as e:
        print(f"  [warn] {ticker} 10-K list: {e}")
        return out

    seen_fy = set()
    for filing in filings:
        try:
            fdate = str(filing.filing_date)
            if int(fdate.split("-")[0]) not in years:
                continue
            df = filing.xbrl().facts.query().to_dataframe()
        except Exception:
            continue

        # revenue facts tagged with a timing label
        rev = df[df["concept"].astype(str).str.contains("Revenue", case=False, na=False)].copy()
        rev = rev[rev["numeric_value"].notna()]
        if rev.empty:
            continue
        rev["is_point"] = rev["label"].astype(str).apply(lambda s: bool(POINT_RE.search(s)))
        rev["is_over"]  = rev["label"].astype(str).apply(lambda s: bool(OVER_RE.search(s)))

        # group by fiscal-year end; need both a point-in-time and an over-time value
        for fy_end, grp in rev.groupby("period_end"):
            if fy_end in seen_fy:
                continue
            pts = grp[grp["is_point"]]
            ovs = grp[grp["is_over"]]
            if pts.empty or ovs.empty:
                continue
            # use the largest of each (top-level totals, not segment splits)
            pv = float(pts["numeric_value"].abs().max())
            ov = float(ovs["numeric_value"].abs().max())
            # Require both values to be real revenue magnitudes (≥ $50M) — guards
            # against the label match grabbing tiny non-revenue facts (ratios, counts).
            if pv < 50e6 or ov < 50e6:
                continue
            # require a material gap (R14 will also enforce this downstream)
            if abs(pv - ov) / max(pv, ov) < 0.05:
                continue
            seen_fy.add(fy_end)

            passage_text = (
                f"{company} — Disaggregation of Revenue (ASC 606), fiscal year ended {fy_end}.\n"
                f"The Company disaggregates revenue by the timing of revenue recognition.\n"
                f"Revenue Recognized at a Point in Time: {_fmt(pv)}.\n"
                f"Revenue Recognized Over Time: {_fmt(ov)}.\n"
                f"Total net revenue is the sum of revenue recognized at a point in time and "
                f"revenue recognized over time."
            )
            print(f"  [recognition] {ticker} FY{fy_end}: point={_fmt(pv)} over={_fmt(ov)}")
            out.append({
                "passage_id":       f"edgar_recog_{ticker}_{str(fy_end)[:4]}",
                "source":           "edgar_recognition",
                "language":         "en",
                "ticker":           ticker,
                "company":          company,
                "period":           str(fy_end),
                "filing_type":      "10-K",
                "filing_date":      fdate,
                "passage_text":     passage_text,
                "candidate_answer": _fmt(pv),
                "candidate_axes":   ["recognition_policy"],
                "_xbrl_hint": {
                    "concept":        "Revenues",
                    "point_in_time":  _fmt(pv),
                    "over_time":      _fmt(ov),
                    "interpretation": "point_in_time",
                    "default_interp": "over_time",
                    "fiscal_year_end": str(fy_end),
                },
            })
            break  # one (latest) FY per ticker is enough
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
    ap.add_argument("--out", type=Path, default=Path("data/sources/edgar_recognition_passages.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    ident = os.environ.get("EDGAR_IDENTITY")
    if not ident:
        sys.exit("Set EDGAR_IDENTITY='Name email@school.edu'")
    set_identity(ident)

    tickers = read_tickers(args.targets)
    if args.limit:
        tickers = tickers[:args.limit]
    print(f"Scanning {len(tickers)} tickers for ASC 606 revenue-timing disaggregation")

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

    print(f"\nDone. {total} recognition_policy passages -> {args.out}")


if __name__ == "__main__":
    main()
