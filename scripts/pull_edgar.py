"""
Pull 10-K / 10-Q / 10-K/A filings from EDGAR via edgartools for a target company list.
Saves structured passages (MD&A Item 7, Segment Reporting, Risk Factors Item 1A)
plus XBRL-extracted financial facts to JSONL records the constructor can consume.

Why edgartools: official SEC-friendly client, XBRL-aware, returns parsed sections
rather than raw HTML. Authoritative quote: "edgartools knows about Items."

Usage:
    pip install edgartools
    # Required by SEC: identify yourself with name + email (no auth needed, just polite)
    export EDGAR_IDENTITY="Your Name your.email@school.edu"
    python scripts/pull_edgar.py \
        --targets data/edgar/target_companies.txt \
        --forms 10-K \
        --years 2024 2025 \
        --out data/edgar/edgar_filings.jsonl

Output JSONL record schema:
    {
      "ticker": "AAPL",
      "cik": "0000320193",
      "company_name": "Apple Inc.",
      "form": "10-K",
      "filing_date": "2024-11-01",
      "period_of_report": "2024-09-28",
      "accession_no": "0000320193-24-000123",
      "fiscal_year_end": "0928",
      "sections": {
        "item_1_business": "...",
        "item_1a_risk_factors": "...",
        "item_7_mda": "...",
        "item_8_financial_statements": "..."
      },
      "facts": {                 # selected XBRL facts (period-tagged)
        "Revenues": [...],
        "GrossProfit": [...],
        "OperatingIncomeLoss": [...]
      },
      "url": "https://www.sec.gov/Archives/..."
    }
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("EDGAR_LOCAL_DATA_DIR", str(PROJECT_ROOT / "data" / "edgar" / ".edgar"))
os.environ.setdefault("EDGAR_CACHE_DIR", str(PROJECT_ROOT / "data" / "edgar" / ".edgar_cache"))

try:
    from edgar import Company, set_identity
    from edgar.financials import Financials
except ImportError:
    sys.exit("Missing edgartools. Install with: pip install edgartools")


# Sections we typically want for the constructor seed pool.
# Section keys vary across filing versions, so we try multiple aliases.
SECTION_ALIASES = {
    "item_1_business":           ["Item 1", "Business"],
    "item_1a_risk_factors":      ["Item 1A", "Risk Factors"],
    "item_7_mda":                ["Item 7", "Management's Discussion and Analysis",
                                  "MD&A"],
    "item_7a_market_risk":       ["Item 7A", "Quantitative and Qualitative Disclosures"],
    "item_8_financial_statements": ["Item 8", "Financial Statements and Supplementary Data"],
}

# Core XBRL fact concepts to extract for ground-truth answers
CORE_CONCEPTS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "GrossProfit",
    "OperatingIncomeLoss",
    "NetIncomeLoss",
    "EarningsPerShareDiluted",
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
]


def json_safe(value):
    """Convert pandas/numpy/date-ish values into JSON-safe scalars."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and value != value:
            return None
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def dataframe_records(df, limit: int = 20) -> list[dict]:
    records = []
    for rec in df.head(limit).to_dict(orient="records"):
        records.append({k: json_safe(v) for k, v in rec.items()})
    return records


def read_tickers(path: Path) -> list[str]:
    tickers = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            tickers.append(line.split()[0].upper())
    return tickers


def extract_sections(filing) -> dict:
    """Pull common 10-K items. edgartools >= 4.x exposes filing.obj() with .items.
    Falls back to text() if structured access is unavailable."""
    out = {}
    try:
        tenk = filing.obj()                       # Form-specific parser
        for key, aliases in SECTION_ALIASES.items():
            for alias in aliases:
                try:
                    val = tenk[alias]
                    if val:
                        out[key] = str(val).strip()
                        break
                except Exception:
                    continue
    except Exception as e:
        print(f"    [warn] structured parse failed ({type(e).__name__}): {e}")
    # Fallback: if structured parse yielded nothing, save full text under a sentinel
    if not out:
        try:
            out["_full_text"] = filing.text()[:200_000]   # cap at 200KB
        except Exception as e:
            out["_full_text_error"] = str(e)
    return out


def extract_facts(filing) -> dict:
    """Pull selected XBRL facts. Each fact has (value, period, unit)."""
    facts = {}
    try:
        xbrl = filing.xbrl()
        xbrl_facts = getattr(xbrl, "facts", None)
        if xbrl_facts is None:
            return facts
        for concept in CORE_CONCEPTS:
            try:
                pattern = rf"(^|:){re.escape(concept)}$"
                df = xbrl_facts.get_facts_by_concept(pattern, exact=False)
                if df is not None and not df.empty:
                    facts[concept] = dataframe_records(df)
            except Exception:
                continue
    except Exception as e:
        try:
            fin = Financials.extract(filing)
            facts["_financials_available"] = fin is not None
        except Exception:
            pass
        facts["_error"] = f"{type(e).__name__}: {e}"
    return facts


def pull_one(ticker: str, forms: list[str], years: list[int]) -> list[dict]:
    """Pull all matching filings for one ticker. Returns list of records."""
    records = []
    try:
        co = Company(ticker)
    except Exception as e:
        print(f"  [err] cannot resolve {ticker}: {e}")
        return records

    cik = str(co.cik).zfill(10)
    company_name = co.name

    for form in forms:
        try:
            filings = co.get_filings(form=form)
        except Exception as e:
            print(f"  [err] {ticker} {form} list: {e}")
            continue

        for f in filings:
            try:
                fdate = str(f.filing_date)
                fyear = int(fdate.split("-")[0])
                if fyear not in years:
                    continue
                print(f"  - {ticker} {form} filed {fdate}")
                rec = {
                    "ticker": ticker,
                    "cik": cik,
                    "company_name": company_name,
                    "form": form,
                    "filing_date": fdate,
                    "period_of_report": str(getattr(f, "period_of_report", "") or ""),
                    "accession_no": str(getattr(f, "accession_no", "") or ""),
                    "url": getattr(f, "filing_url", "") or "",
                    "sections": extract_sections(f),
                    "facts": extract_facts(f),
                }
                records.append(rec)
                time.sleep(0.5)   # be polite to SEC
            except Exception as e:
                print(f"  [warn] {ticker} {form} {fdate}: {e}")
                continue
    return records


def main(targets_path: Path, forms: list[str], years: list[int], out_path: Path,
         limit_tickers: int | None):
    identity = os.environ.get("EDGAR_IDENTITY")
    if not identity:
        sys.exit(
            "Set EDGAR_IDENTITY env var with 'Your Name your.email@school.edu' "
            "(SEC requires identification; no API key, just politeness)."
        )
    set_identity(identity)

    tickers = read_tickers(targets_path)
    if limit_tickers is not None:
        tickers = tickers[:limit_tickers]
    print(f"Pulling {len(tickers)} tickers, forms={forms}, years={years}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Load already-pulled tickers so re-runs skip them (append mode)
    done_tickers: set[str] = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    done_tickers.add(json.loads(line.strip()).get("ticker", ""))
                except Exception:
                    pass
    if done_tickers:
        print(f"[resume] skipping {len(done_tickers)} already-pulled tickers")
        tickers = [t for t in tickers if t not in done_tickers]
    n_written = 0
    with out_path.open("a", encoding="utf-8") as f:
        for t in tickers:
            print(f"[{t}]")
            for rec in pull_one(t, forms, years):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"\nWrote {n_written} filings to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--targets", type=Path,
                   default=Path("data/edgar/target_companies.txt"))
    p.add_argument("--forms", nargs="+", default=["10-K"],
                   help="EDGAR forms to pull (e.g. 10-K 10-Q 10-K/A)")
    p.add_argument("--years", nargs="+", type=int, default=[2024, 2025],
                   help="Filing years to keep")
    p.add_argument("--out", type=Path,
                   default=Path("data/edgar/edgar_filings.jsonl"))
    p.add_argument("--limit-tickers", type=int, default=None,
                   help="Only pull the first N tickers from the target file.")
    args = p.parse_args()
    main(args.targets, args.forms, args.years, args.out, args.limit_tickers)
