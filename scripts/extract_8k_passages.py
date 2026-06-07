"""
Extract GAAP vs non-GAAP passage records from EDGAR 8-K earnings releases.

8-K Item 2.02 (Results of Operations) press releases are the richest source
of metric_definition ambiguity: CFOs invariably state both GAAP and non-GAAP
figures side-by-side in the same document, providing guaranteed paired values.

Strategy
--------
1. Pull 8-K filings (Item 2.02) for each target ticker via edgartools.
2. Search the exhibit text for GAAP / non-GAAP reconciliation tables.
3. Extract a GAAP value (intended) and the corresponding non-GAAP value
   (default — what a casual reader might grab from headlines).
4. Emit a passage record with candidate_axes=['metric_definition'].

Usage
-----
    export EDGAR_IDENTITY="Your Name your.email@school.edu"
    python scripts/extract_8k_passages.py \\
        --targets data/edgar/target_companies_b2.txt \\
        --years 2023 2024 2025 \\
        --out data/sources/edgar_8k_passages.jsonl

Merge into the main pool:
    cat data/sources/edgar_8k_passages.jsonl >> data/sources/edgar_passages.jsonl
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

# Patterns to find GAAP vs non-GAAP pairs in press release text
GAAP_LABEL_RE = re.compile(
    r"(?:GAAP|as reported|reported)\s+(?:net income|EPS|earnings per share|"
    r"operating income|revenue|net revenue|gross profit)[^\n]*?"
    r"(\$[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|B|M))?|\d+\.\d+)",
    re.I,
)
NON_GAAP_LABEL_RE = re.compile(
    r"(?:non-GAAP|adjusted|non GAAP)\s+(?:net income|EPS|earnings per share|"
    r"operating income|revenue|net revenue|gross profit)[^\n]*?"
    r"(\$[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|B|M))?|\d+\.\d+)",
    re.I,
)
DOLLAR_RE = re.compile(r"\$\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|B|M)?", re.I)


def _find_gaap_nongaap_pair(text: str) -> tuple[str, str] | None:
    """Return (gaap_value, nongaap_value) strings if a clear pair is found."""
    gaap_m = GAAP_LABEL_RE.search(text)
    ngaap_m = NON_GAAP_LABEL_RE.search(text)
    if gaap_m and ngaap_m:
        gv = gaap_m.group(1).strip()
        nv = ngaap_m.group(1).strip()
        if gv != nv:
            return (gv, nv)

    # Fallback: look for "GAAP ... $X ... non-GAAP ... $Y" in proximity
    gaap_idx = text.lower().find("gaap")
    ngaap_idx = text.lower().find("non-gaap")
    if gaap_idx == -1 or ngaap_idx == -1:
        return None

    window = 500
    gaap_window = text[max(0, gaap_idx - 50): gaap_idx + window]
    ngaap_window = text[max(0, ngaap_idx - 50): ngaap_idx + window]
    gaap_nums = DOLLAR_RE.findall(gaap_window)
    ngaap_nums = DOLLAR_RE.findall(ngaap_window)
    if gaap_nums and ngaap_nums and gaap_nums[0] != ngaap_nums[0]:
        return (gaap_nums[0].strip(), ngaap_nums[0].strip())

    return None


def _extract_8k_text(filing) -> str:
    """Pull the press release exhibit text from an 8-K filing."""
    try:
        # Try to get the earnings release exhibit (typically ex-99.1)
        docs = filing.documents
        if docs is not None:
            for doc in docs:
                desc = str(getattr(doc, "description", "") or "").lower()
                doc_type = str(getattr(doc, "document_type", "") or "").lower()
                if any(kw in desc for kw in ("press release", "earnings", "results")) or \
                   "ex-99" in doc_type or "ex99" in doc_type:
                    try:
                        content = doc.text if hasattr(doc, "text") else str(doc)
                        if len(content) > 300:
                            return content[:10000]
                    except Exception:
                        continue
    except Exception:
        pass
    # Fallback to full filing text
    try:
        return filing.text()[:10000]
    except Exception:
        return ""


def pull_8k_for_ticker(ticker: str, years: list[int]) -> list[dict]:
    passages = []
    try:
        co = Company(ticker)
    except Exception as e:
        print(f"  [err] {ticker}: {e}")
        return passages

    company_name = co.name

    try:
        filings_8k = co.get_filings(form="8-K")
    except Exception as e:
        print(f"  [warn] {ticker} 8-K list: {e}")
        return passages

    found = 0
    for filing in filings_8k:
        if found >= 2:  # cap at 2 passages per ticker to avoid redundancy
            break
        try:
            fdate = str(filing.filing_date)
            if int(fdate.split("-")[0]) not in years:
                continue

            text = _extract_8k_text(filing)
            if len(text) < 300:
                continue

            # Must mention both GAAP and non-GAAP
            tl = text.lower()
            if "non-gaap" not in tl and "adjusted" not in tl:
                continue
            if "gaap" not in tl:
                continue

            pair = _find_gaap_nongaap_pair(text)
            if pair is None:
                continue

            gaap_val, nongaap_val = pair
            print(f"  [8-K] {ticker} filed {fdate}: GAAP={gaap_val} non-GAAP={nongaap_val}")

            passages.append({
                "passage_id":       f"edgar_8k_{ticker}_{fdate[:7].replace('-','')}",
                "source":           "edgar_8k",
                "language":         "en",
                "ticker":           ticker,
                "company":          company_name,
                "period":           fdate[:7],
                "filing_type":      "8-K",
                "filing_date":      fdate,
                "passage_text":     text,
                "candidate_answer": gaap_val,
                "candidate_axes":   ["metric_definition"],
                "_xbrl_hint": {
                    "gaap_val":    gaap_val,
                    "nongaap_val": nongaap_val,
                    "interpretation": "gaap",
                    "default_interp": "non_gaap",
                },
            })
            found += 1
            time.sleep(0.3)

        except Exception as e:
            print(f"  [warn] {ticker} 8-K {fdate}: {e}")
            continue

    return passages


def read_tickers(path: Path) -> list[str]:
    tickers, seen = [], set()
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
    print(f"Scanning {len(tickers)} tickers for 8-K GAAP/non-GAAP pairs, years={years}")

    out.parent.mkdir(parents=True, exist_ok=True)
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
            passages = pull_8k_for_ticker(ticker, years)
            for p in passages:
                if p.get("passage_id") in existing:
                    continue
                f_out.write(json.dumps(p, ensure_ascii=False) + "\n")
                existing.add(p.get("passage_id", ""))
                total += 1

    print(f"\nDone. {total} metric_definition passages → {out}")
    print("Next steps:")
    print(f"  cat {out} >> data/sources/edgar_passages.jsonl")
    print("  python scripts/prepare_passages.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Extract GAAP vs non-GAAP pairs from 8-K earnings releases")
    p.add_argument("--targets", type=Path,
                   default=Path("data/edgar/target_companies_b2.txt"))
    p.add_argument("--years",   nargs="+", type=int, default=[2023, 2024, 2025])
    p.add_argument("--out",     type=Path,
                   default=Path("data/sources/edgar_8k_passages.jsonl"))
    p.add_argument("--limit",   type=int, default=None)
    args = p.parse_args()
    main(args.targets, args.years, args.out, args.limit)
