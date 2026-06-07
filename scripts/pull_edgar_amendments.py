"""
Pull 10-K/A (amended annual report) filings from EDGAR for filing_vintage passages.

These filings are the richest source of filing_vintage ambiguity:
  - "default" answer  = value as originally reported in the 10-K
  - "intended" answer = restated/corrected value in the 10-K/A

Strategy
--------
For each ticker:
  1. Pull the most recent 10-K filing and extract a key financial metric (XBRL).
  2. Look for a 10-K/A filed within 18 months of that 10-K.
  3. Extract the same metric from the 10-K/A.
  4. If the values differ by ≥1%, emit a passage record with both values in the
     XBRL hint, so the constructor can build a genuine default-vs-intended pair.

If no amendment is found, fall back to extracting passages that contain
restatement language ("as restated", "revised", "prior period adjustment")
directly from Item 7 MD&A of the original 10-K.

Usage
-----
    export EDGAR_IDENTITY="Your Name your.email@school.edu"
    python scripts/pull_edgar_amendments.py \\
        --targets data/edgar/target_companies_b2.txt \\
        --years 2022 2023 2024 \\
        --out data/sources/edgar_amendment_passages.jsonl

Output passages are in the standard passage pool schema with
  source         = "edgar_amendment"
  candidate_axes = ["filing_vintage"]
  _xbrl_hint     = { original_val, restated_val, concept, diff_pct }

Merge into the main pool:
    cat data/sources/edgar_amendment_passages.jsonl \\
        >> data/sources/edgar_passages.jsonl
    python scripts/prepare_passages.py
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("EDGAR_LOCAL_DATA_DIR",
                      str(PROJECT_ROOT / "data" / "edgar" / ".edgar"))
os.environ.setdefault("EDGAR_CACHE_DIR",
                      str(PROJECT_ROOT / "data" / "edgar" / ".edgar_cache"))

try:
    from edgar import Company, set_identity
except ImportError:
    sys.exit("Missing edgartools. Install with: pip install edgartools")

RESTATEMENT_RE = re.compile(
    r"as restated|as revised|prior period|restatement|"
    r"we have restated|correction|amendment|amended|"
    r"revised to reflect|retrospective|recast",
    re.I,
)

INCOME_CONCEPTS = [
    "NetIncomeLoss",
    "OperatingIncomeLoss",
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "GrossProfit",
    "EarningsPerShareDiluted",
]


def _fmt(val: float, concept: str) -> str:
    if any(c in concept for c in ("PerShare", "EPS")):
        return f"${val:.2f}"
    if abs(val) >= 1e9:
        return f"${val / 1e9:.2f}B"
    if abs(val) >= 1e6:
        return f"${val / 1e6:.1f}M"
    return f"${val:.2f}"


def _extract_xbrl_metric(filing) -> tuple[str, float, str] | None:
    """Return (concept_name, numeric_value, formatted) for the first available metric."""
    try:
        xbrl = filing.xbrl()
        facts = getattr(xbrl, "facts", None)
        if facts is None:
            return None
        for concept in INCOME_CONCEPTS:
            try:
                df = facts.get_facts_by_concept(concept, exact=False)
                if df is None or df.empty:
                    continue
                undim = df[~df.get("is_dimensioned", False)] if "is_dimensioned" in df.columns else df
                fy_rows = undim[undim.get("fiscal_period", "") == "FY"] if "fiscal_period" in undim.columns else undim
                if fy_rows.empty:
                    fy_rows = undim
                best = fy_rows.sort_values("fiscal_year", ascending=False).iloc[0]
                val = float(best.get("numeric_value", 0) or 0)
                if val == 0:
                    continue
                return (concept, val, _fmt(val, concept))
            except Exception:
                continue
    except Exception:
        pass
    return None


def _get_mda_text(filing) -> str:
    try:
        obj = filing.obj()
        for key in ("Item 7", "Management's Discussion and Analysis", "MD&A"):
            try:
                val = obj[key]
                if val and len(str(val)) > 300:
                    return str(val)[:8000]
            except Exception:
                continue
    except Exception:
        pass
    try:
        return filing.text()[:8000]
    except Exception:
        return ""


def pull_amendments_for_ticker(ticker: str, years: list[int]) -> list[dict]:
    passages = []
    try:
        co = Company(ticker)
    except Exception as e:
        print(f"  [err] {ticker}: {e}")
        return passages

    company_name = co.name

    # --- Strategy A: find actual 10-K/A amendment filings ---
    try:
        amendments = co.get_filings(form="10-K/A")
        for amend in amendments:
            try:
                adate = str(amend.filing_date)
                if int(adate.split("-")[0]) not in years:
                    continue
                print(f"  [amendment] {ticker} 10-K/A filed {adate}")
                amend_metric = _extract_xbrl_metric(amend)
                if amend_metric is None:
                    continue
                concept, restated_val, restated_fmt = amend_metric

                # Try to find the original 10-K filed before this amendment
                originals = co.get_filings(form="10-K")
                original_metric = None
                for orig in originals:
                    try:
                        odate = str(orig.filing_date)
                        if odate < adate:  # filed before the amendment
                            original_metric = _extract_xbrl_metric(orig)
                            if original_metric and original_metric[0] == concept:
                                break
                    except Exception:
                        continue

                if original_metric is None or original_metric[0] != concept:
                    continue
                _, orig_val, orig_fmt = original_metric

                # Only emit if values actually differ (≥1% difference)
                diff_pct = abs(restated_val - orig_val) / (abs(orig_val) + 1e-9) * 100
                if diff_pct < 1.0:
                    continue

                text = _get_mda_text(amend)
                if len(text) < 200:
                    continue

                passages.append({
                    "passage_id":       f"edgar_amend_{ticker}_{adate[:4]}_10ka",
                    "source":           "edgar_amendment",
                    "language":         "en",
                    "ticker":           ticker,
                    "company":          company_name,
                    "period":           str(getattr(amend, "period_of_report", "") or ""),
                    "filing_type":      "10-K/A",
                    "filing_date":      adate,
                    "passage_text":     text,
                    "candidate_answer": restated_fmt,
                    "candidate_axes":   ["filing_vintage"],
                    "_xbrl_hint": {
                        "concept":       concept,
                        "original_val":  orig_fmt,
                        "restated_val":  restated_fmt,
                        "diff_pct":      f"{diff_pct:.1f}%",
                        "amendment_date": adate,
                        "interpretation": "restated",
                    },
                })
                time.sleep(0.3)
            except Exception as e:
                print(f"  [warn] {ticker} amendment: {e}")
                continue
    except Exception as e:
        print(f"  [warn] {ticker} 10-K/A list: {e}")

    # --- Strategy B: 10-K filings with restatement language in MD&A ---
    try:
        tenks = co.get_filings(form="10-K")
        for filing in tenks:
            try:
                fdate = str(filing.filing_date)
                if int(fdate.split("-")[0]) not in years:
                    continue
                text = _get_mda_text(filing)
                if len(text) < 300:
                    continue
                if not RESTATEMENT_RE.search(text):
                    continue

                metric = _extract_xbrl_metric(filing)
                if metric is None:
                    continue
                concept, val, fmt_val = metric

                print(f"  [restatement-language] {ticker} 10-K filed {fdate}")
                passages.append({
                    "passage_id":       f"edgar_amend_{ticker}_{fdate[:4]}_restate",
                    "source":           "edgar_amendment",
                    "language":         "en",
                    "ticker":           ticker,
                    "company":          company_name,
                    "period":           str(getattr(filing, "period_of_report", "") or ""),
                    "filing_type":      "10-K",
                    "filing_date":      fdate,
                    "passage_text":     text,
                    "candidate_answer": fmt_val,
                    "candidate_axes":   ["filing_vintage"],
                    "_xbrl_hint": {
                        "concept":       concept,
                        "restated_val":  fmt_val,
                        "original_val":  None,
                        "diff_pct":      None,
                        "interpretation": "as_restated_in_filing",
                    },
                })
                time.sleep(0.3)
                break  # one per ticker is enough for Strategy B
            except Exception as e:
                print(f"  [warn] {ticker} 10-K restate scan: {e}")
                continue
    except Exception as e:
        print(f"  [warn] {ticker} 10-K list: {e}")

    return passages


def read_tickers(path: Path) -> list[str]:
    tickers = []
    seen: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        t = line.split()[0].upper()
        if t not in seen:
            seen.add(t)
            tickers.append(t)
    return tickers


def main(targets: Path, years: list[int], out: Path, limit: int | None):
    identity = os.environ.get("EDGAR_IDENTITY")
    if not identity:
        sys.exit("Set EDGAR_IDENTITY='Name email@school.edu'")
    set_identity(identity)

    tickers = read_tickers(targets)
    if limit:
        tickers = tickers[:limit]
    print(f"Scanning {len(tickers)} tickers for 10-K/A amendments and restatements, years={years}")

    out.parent.mkdir(parents=True, exist_ok=True)
    # Append so re-runs don't lose previously collected passages
    existing = set()
    if out.exists():
        with out.open() as f:
            for line in f:
                try:
                    existing.add(json.loads(line.strip()).get("passage_id", ""))
                except Exception:
                    pass
    total = 0
    with out.open("a", encoding="utf-8") as f_out:
        for ticker in tickers:
            print(f"[{ticker}]")
            passages = pull_amendments_for_ticker(ticker, years)
            for p in passages:
                if p.get("passage_id") in existing:
                    continue
                f_out.write(json.dumps(p, ensure_ascii=False) + "\n")
                existing.add(p.get("passage_id", ""))
                total += 1

    print(f"\nDone. {total} filing_vintage passages → {out}")
    print("Next steps:")
    print(f"  cat {out} >> data/sources/edgar_passages.jsonl")
    print("  python scripts/prepare_passages.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Pull 10-K/A amendments for filing_vintage passages")
    p.add_argument("--targets", type=Path,
                   default=Path("data/edgar/target_companies_b2.txt"))
    p.add_argument("--years",   nargs="+", type=int, default=[2022, 2023, 2024])
    p.add_argument("--out",     type=Path,
                   default=Path("data/sources/edgar_amendment_passages.jsonl"))
    p.add_argument("--limit",   type=int, default=None,
                   help="Process only first N tickers (for testing)")
    args = p.parse_args()
    main(args.targets, args.years, args.out, args.limit)
