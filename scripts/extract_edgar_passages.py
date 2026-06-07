"""
Extract candidate passage records from EDGAR XBRL facts for use in construct_instances.py.

Each EDGAR filing has structured XBRL facts (segment revenues, operating income, etc.).
This script generates passage records with verified candidate_answer values derived from
the XBRL data — enabling entity_scope, temporal_scope, and metric_definition instances
that DocFinQA cannot easily provide.

Strategy per filing
--------------------
1. Entity-scope passages: find dimensioned facts (segment breakdowns).
   The "intended" interpretation is a specific segment; "default" is consolidated.
   candidate_answer = consolidated value or largest-segment value.

2. Temporal-scope passages: find facts across two consecutive fiscal years.
   The "intended" interpretation is a specific FY; "default" is the prior year.
   candidate_answer = value for the most recent FY.

3. Metric-definition passages (for companies reporting both GAAP and adjusted metrics):
   Requires passage text mentioning non-GAAP; candidate_answer = GAAP value.

Output: appended to data/sources/edgar_passages.jsonl (separate file for inspection).
Run prepare_passages.py after this to merge into the main pool.

Usage:
    python scripts/extract_edgar_passages.py
    python scripts/extract_edgar_passages.py --out data/sources/edgar_passages.jsonl
"""

import json
import argparse
from pathlib import Path

EDGAR_FILE  = Path("data/edgar/edgar_filings.jsonl")
OUT_DEFAULT = Path("data/sources/edgar_passages.jsonl")

# XBRL concepts we can use as candidate answers (order = preference)
REVENUE_CONCEPTS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
]
INCOME_CONCEPTS = [
    "OperatingIncomeLoss",
    "NetIncomeLoss",
    "GrossProfit",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
]
EPS_CONCEPTS = [
    "EarningsPerShareDiluted",
    "EarningsPerShareBasic",
]
ALL_TARGET_CONCEPTS = REVENUE_CONCEPTS + INCOME_CONCEPTS + EPS_CONCEPTS

# Companies whose fiscal year ends in a month other than December.
# For these, "What was X's revenue?" is genuinely ambiguous between FY and CY.
NON_CALENDAR_FY = {
    "AAPL",  # FY ends late September
    "MSFT",  # FY ends June 30
    "WMT",   # FY ends late January
    "NKE",   # FY ends May 31
    "ORCL",  # FY ends May 31
    "ADBE",  # FY ends late November
    "COST",  # FY ends late August
    "HPE",   # FY ends October
    "FDX",   # FY ends May 31
    "ACN",   # FY ends August 31
    "MU",    # FY ends late August
    "BBY",   # FY ends late January
    "DLTR",  # FY ends late January
    "TJX",   # FY ends late January
    "HD",    # FY ends late January
    "LOW",   # FY ends late January
    "TGT",   # FY ends late January
    "INTU",  # FY ends July
    "WDAY",  # FY ends January
    "ADSK",  # FY ends January
    "DE",    # FY ends October
    "EMR",   # FY ends September
    "PH",    # FY ends June
}


def _fmt(val: float, concept: str) -> str:
    """Format a numeric XBRL value as a human-readable answer string."""
    if any(c in concept for c in ("PerShare", "EPS")):
        return f"${val:.2f}"
    if abs(val) >= 1e9:
        return f"${val/1e9:.2f}B"
    if abs(val) >= 1e6:
        return f"${val/1e6:.1f}M"
    return f"${val:.2f}"


def _infer_axes(ticker: str, text: str, has_segments: bool,
                has_multi_period: bool) -> list[str]:
    axes = set()
    if ticker.upper() in NON_CALENDAR_FY:
        axes.add("temporal_scope")
    if has_segments:
        axes.add("entity_scope")
    if has_multi_period:
        axes.add("temporal_scope")
    text_l = text.lower()
    if any(kw in text_l for kw in ("non-gaap", "adjusted", "organic", "constant currency",
                                    "ebitda", "excluding")):
        axes.add("metric_definition")
    if any(kw in text_l for kw in ("revenue recognition", "impairment", "capitalized",
                                    "related party", "deferred")):
        axes.add("recognition_policy")
    if not axes:
        axes = {"temporal_scope", "entity_scope"}
    return sorted(axes)[:3]


def extract_from_filing(rec: dict, passage_idx: int) -> list[dict]:
    """Return 0-3 passage records from one EDGAR filing."""
    ticker  = rec.get("ticker", "")
    company = rec.get("company_name", "")
    period  = rec.get("period_of_report", "")
    filing_date = rec.get("filing_date", "")
    facts   = rec.get("facts", {})
    sections = rec.get("sections") or {}

    # Prefer MD&A for passage text
    text = (sections.get("item_7_mda")
            or sections.get("item_8_financial_statements")
            or sections.get("_full_text") or "")
    if len(text) < 200:
        return []

    results = []

    for concept_name, fact_list in facts.items():
        # Only look at targeted concepts
        short_name = concept_name.split(":")[-1] if ":" in concept_name else concept_name
        if short_name not in ALL_TARGET_CONCEPTS:
            continue
        if not isinstance(fact_list, list) or not fact_list:
            continue

        # Separate dimensioned (segment) from undimensioned (consolidated)
        dimensioned   = [f for f in fact_list if f.get("is_dimensioned")]
        undimensioned = [f for f in fact_list if not f.get("is_dimensioned")]

        # --- Entity-scope passage: consolidated vs segment ---
        if dimensioned and undimensioned:
            # Take the most recent FY consolidated value
            consol = sorted(undimensioned, key=lambda f: f.get("fiscal_year", 0))
            if not consol:
                continue
            consol_fact = consol[-1]
            consol_val  = consol_fact.get("numeric_value")
            if not consol_val:
                continue

            # Find a named segment with a materially different value (>10% difference)
            segs = [f for f in dimensioned
                    if f.get("fiscal_year") == consol_fact.get("fiscal_year")
                    and f.get("label") and f.get("numeric_value")]
            if not segs:
                continue
            # Pick the largest segment
            seg = max(segs, key=lambda f: abs(f.get("numeric_value", 0)))
            seg_val = seg.get("numeric_value")
            if seg_val is None or abs(seg_val - consol_val) / (abs(consol_val) + 1) < 0.05:
                continue

            label   = short_name.replace("FromContractWithCustomer", "")
            concept_readable = (
                "revenue" if "Revenue" in short_name or "Sales" in short_name
                else "operating income" if "Operating" in short_name
                else "net income" if "NetIncome" in short_name
                else "gross profit" if "GrossProfit" in short_name
                else "EPS"
            )
            axes = _infer_axes(ticker, text, has_segments=True, has_multi_period=False)

            results.append({
                "passage_id":       f"edgar_{ticker}_{passage_idx}_seg",
                "source":           "edgar",
                "language":         "en",
                "ticker":           ticker,
                "company":          company,
                "period":           period,
                "filing_type":      rec.get("form", "10-K"),
                "filing_date":      filing_date,
                "passage_text":     text[:8000],
                "candidate_answer": _fmt(consol_val, short_name),
                "candidate_axes":   axes,
                "_xbrl_hint": {
                    "concept":      short_name,
                    "consolidated": _fmt(consol_val, short_name),
                    "segment":      seg.get("label"),
                    "segment_val":  _fmt(seg_val, short_name),
                    "fiscal_year":  consol_fact.get("fiscal_year"),
                    "concept_type": concept_readable,
                },
            })

        # --- Temporal-scope passage: two consecutive fiscal years ---
        if len(undimensioned) >= 2:
            annual = sorted(
                [f for f in undimensioned
                 if f.get("fiscal_period") == "FY" and f.get("numeric_value") is not None],
                key=lambda f: f.get("fiscal_year", 0)
            )
            if len(annual) < 2:
                continue
            curr, prev = annual[-1], annual[-2]
            if curr.get("fiscal_year") == prev.get("fiscal_year"):
                continue
            curr_val = curr.get("numeric_value")
            prev_val = prev.get("numeric_value")
            if not curr_val or not prev_val or prev_val == 0:
                continue
            growth = (curr_val - prev_val) / abs(prev_val) * 100

            concept_readable = (
                "revenue" if "Revenue" in short_name or "Sales" in short_name
                else "operating income" if "Operating" in short_name
                else "net income" if "NetIncome" in short_name
                else "EPS"
            )
            axes = _infer_axes(ticker, text, has_segments=False, has_multi_period=True)

            results.append({
                "passage_id":       f"edgar_{ticker}_{passage_idx}_yoy",
                "source":           "edgar",
                "language":         "en",
                "ticker":           ticker,
                "company":          company,
                "period":           curr.get("period_end", period),
                "filing_type":      rec.get("form", "10-K"),
                "filing_date":      filing_date,
                "passage_text":     text[:8000],
                "candidate_answer": f"{growth:.1f}%",
                "candidate_axes":   axes,
                "_xbrl_hint": {
                    "concept":       short_name,
                    "curr_fy":       curr.get("fiscal_year"),
                    "curr_val":      _fmt(curr_val, short_name),
                    "prev_fy":       prev.get("fiscal_year"),
                    "prev_val":      _fmt(prev_val, short_name),
                    "yoy_growth":    f"{growth:.1f}%",
                    "concept_type":  concept_readable,
                },
            })

        if len(results) >= 3:
            break   # cap at 3 passages per filing to avoid duplicating context

    # --- Non-calendar FY temporal passage: absolute FY value vs same CY period ---
    # For non-calendar FY companies, "What was X's revenue?" can mean the company's
    # fiscal year (e.g. Apple's FY ending Sept 2024) OR the calendar year 2024.
    # We emit a passage anchored to the FY value as the intended answer.
    if ticker.upper() in NON_CALENDAR_FY and len(results) < 3:
        for concept_name, fact_list in facts.items():
            short_name = concept_name.split(":")[-1] if ":" in concept_name else concept_name
            if short_name not in ALL_TARGET_CONCEPTS:
                continue
            if not isinstance(fact_list, list) or not fact_list:
                continue
            undim = [f for f in fact_list if not f.get("is_dimensioned")]
            fy_rows = [f for f in undim
                       if f.get("fiscal_period") == "FY"
                       and f.get("numeric_value") is not None]
            if not fy_rows:
                continue
            fy_rows.sort(key=lambda f: f.get("fiscal_year", 0))
            latest = fy_rows[-1]
            fy_val = latest.get("numeric_value")
            fy_year = latest.get("fiscal_year")
            period_end = latest.get("period_end", "")
            if not fy_val or not fy_year:
                continue

            # We need a prior-year value to make the passage text informative
            if len(fy_rows) >= 2:
                prior = fy_rows[-2]
                prior_val = prior.get("numeric_value")
                prior_year = prior.get("fiscal_year")
            else:
                prior_val, prior_year = None, None

            concept_readable = (
                "revenue" if "Revenue" in short_name or "Sales" in short_name
                else "operating income" if "Operating" in short_name
                else "net income" if "NetIncome" in short_name
                else "EPS" if "PerShare" in short_name
                else "gross profit"
            )
            axes = ["temporal_scope", "metric_definition"] if any(
                kw in text.lower() for kw in ("non-gaap", "adjusted")
            ) else ["temporal_scope"]

            hint: dict = {
                "concept":        short_name,
                "fy_year":        fy_year,
                "fy_val":         _fmt(fy_val, short_name),
                "period_end":     period_end,
                "fy_end_month":   period_end[5:7] if len(period_end) >= 7 else "",
                "concept_type":   concept_readable,
                "non_calendar_fy": True,
            }
            if prior_val and prior_year:
                hint["prior_fy_year"] = prior_year
                hint["prior_fy_val"] = _fmt(prior_val, short_name)

            results.append({
                "passage_id":       f"edgar_{ticker}_{passage_idx}_fy_temporal",
                "source":           "edgar",
                "language":         "en",
                "ticker":           ticker,
                "company":          company,
                "period":           period_end or period,
                "filing_type":      rec.get("form", "10-K"),
                "filing_date":      filing_date,
                "passage_text":     text[:8000],
                "candidate_answer": _fmt(fy_val, short_name),
                "candidate_axes":   axes,
                "_xbrl_hint":       hint,
            })
            break  # one temporal passage per non-calendar FY filing is enough

    return results[:3]


def main(out: Path, filings: Path | None = None):
    edgar_file = filings if filings is not None else EDGAR_FILE
    out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out.open("w", encoding="utf-8") as f_out:
        with edgar_file.open() as f_in:
            for idx, line in enumerate(f_in):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                passages = extract_from_filing(rec, idx)
                for p in passages:
                    f_out.write(json.dumps(p, ensure_ascii=False) + "\n")
                total += len(passages)

    print(f"Extracted {total} EDGAR passages → {out}")
    axis_counts: dict[str, int] = {}
    with out.open() as f:
        for line in f:
            r = json.loads(line)
            for ax in r.get("candidate_axes", []):
                axis_counts[ax] = axis_counts.get(ax, 0) + 1
    for ax, cnt in sorted(axis_counts.items(), key=lambda x: -x[1]):
        print(f"  {ax:<25} {cnt}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--filings", type=Path, default=None,
                   help="EDGAR filings JSONL (default: data/edgar/edgar_filings.jsonl)")
    args = p.parse_args()
    main(args.out, filings=args.filings)
