"""
Merge all source datasets into a uniform passages.jsonl feed for construct_instances.py.

Sources handled:
  - FinanceBench  data/financebench/
  - FinQA         data/finqa/
  - DocFinQA      data/docfinqa/docfinqa_validation.jsonl
  - EDGAR         data/edgar/edgar_filings.jsonl
  - FinEval       data/fineval/
  - CFLUE         data/cflue/

Output: data/sources/passages.jsonl
Each line: one passage record in the construct_instances.py input schema.

Axis assignment heuristics
---------------------------
We assign candidate_axes based on signals in the passage or metadata:

  temporal_scope    : company has non-calendar FY (AAPL, MSFT, WMT, NKE…)
                      OR passage mentions "fiscal year", "FY", "quarter"
  metric_definition : passage mentions GAAP/non-GAAP, adjusted, organic,
                      constant currency, EBITDA, "excluding"
  entity_scope      : passage mentions segments, subsidiaries, geographies,
                      consolidated, parent, or (for ZH) 集团/子公司/分部
  filing_vintage    : passage or filing_type contains "10-K/A", "amended",
                      "restatement", "restated", 更正公告
  recognition_policy: passage mentions revenue recognition, impairment,
                      capitalization, related party, 关联交易

Usage
-----
    python scripts/prepare_passages.py --out data/sources/passages.jsonl

Add --limit N to cap per source for quick testing.
"""

import argparse
import json
import re
from pathlib import Path

OUT_DEFAULT = Path("data/sources/passages.jsonl")

# Tickers known to have non-calendar fiscal years (temporal_scope always hits)
NON_CALENDAR_FY_TICKERS = {
    "AAPL", "MSFT", "WMT", "NKE", "ORCL", "ADBE", "COST", "HPE",
    "PG", "JNJ", "HD", "TGT", "FDX",
}

# Keyword signals → axis
AXIS_SIGNALS: dict[str, list[str]] = {
    "temporal_scope": [
        "fiscal year", "fy20", "fy 20", "quarter", "ttm",
        "trailing twelve", "year-over-year", "comparable period",
        "财年", "季度", "同比",
    ],
    "metric_definition": [
        "non-gaap", "non gaap", "adjusted", "organic", "constant currency",
        "ebitda", "excluding", "pro forma", "as reported",
        "非公认会计准则", "调整后", "剔除", "有机增长",
    ],
    "entity_scope": [
        "segment", "subsidiary", "geographic", "consolidated", "parent company",
        "division", "business unit", "region",
        "分部", "子公司", "集团", "合并", "母公司", "地区",
    ],
    "filing_vintage": [
        "amendment", "amended", "restatement", "restated", "10-k/a", "8-k/a",
        "更正公告", "更正报告",
    ],
    "recognition_policy": [
        "revenue recognition", "impairment", "capitalized", "deferred revenue",
        "related party", "related-party", "receivable",
        "收入确认", "减值", "资本化", "递延收入", "关联交易", "应收",
    ],
}


def infer_axes(ticker: str, text: str) -> list[str]:
    axes = set()
    text_lower = text.lower()
    if ticker.upper() in NON_CALENDAR_FY_TICKERS:
        axes.add("temporal_scope")
    for axis, signals in AXIS_SIGNALS.items():
        if any(s in text_lower for s in signals):
            axes.add(axis)
    # Require at least one axis; default to temporal + metric if nothing found
    if not axes:
        axes = {"temporal_scope", "metric_definition"}
    # Cap at 3 axes per instance (multi-axis instances are harder; save some for variety)
    sorted_axes = sorted(axes)
    return sorted_axes[:3]


def passage_record(passage_id, source, language, ticker, company,
                   period, filing_type, filing_date, text,
                   candidate_answer="") -> dict:
    axes = infer_axes(ticker, text)
    return {
        "passage_id":       passage_id,
        "source":           source,
        "language":         language,
        "ticker":           ticker,
        "company":          company,
        "period":           period,
        "filing_type":      filing_type,
        "filing_date":      filing_date,
        "passage_text":     text[:8000],   # constructor context window cap
        "candidate_answer": candidate_answer,
        "candidate_axes":   axes,
    }


# ---------------------------------------------------------------------------
# Source-specific loaders
# ---------------------------------------------------------------------------

def load_financebench(limit: int | None) -> list[dict]:
    """FinanceBench open-source 150 QA pairs."""
    records = []
    fb_dir = Path("data/financebench")
    # FinanceBench ships as data/financebench_open_source.jsonl or similar
    candidates = list(fb_dir.glob("**/*.jsonl")) + list(fb_dir.glob("**/*.json"))
    for p in candidates[:1]:   # take first matching file
        with p.open() as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = (rec.get("evidence_text") or rec.get("context")
                        or rec.get("text") or "")
                if not text:
                    continue
                ticker = rec.get("ticker") or rec.get("company_ticker") or ""
                records.append(passage_record(
                    passage_id    = f"financebench_{i:04d}",
                    source        = "financebench",
                    language      = "en",
                    ticker        = ticker,
                    company       = rec.get("company_name") or rec.get("company") or "",
                    period        = rec.get("period_of_report") or "",
                    filing_type   = rec.get("doc_type") or "10-K",
                    filing_date   = rec.get("date") or "",
                    text          = text,
                    candidate_answer = str(rec.get("answer") or ""),
                ))
    return records


def load_finqa(limit: int | None) -> list[dict]:
    """FinQA — use as filing-passage source only (answers are our candidate answers)."""
    records = []
    finqa_dir = Path("data/finqa") / "FinQA" / "dataset"
    for split_file in ["train.json", "dev.json"]:
        p = finqa_dir / split_file
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        for i, rec in enumerate(data):
            if limit and len(records) >= limit:
                break
            pre  = " ".join(rec.get("pre_text",  []))
            post = " ".join(rec.get("post_text", []))
            text = pre + " " + post
            if not text.strip():
                continue
            records.append(passage_record(
                passage_id    = f"finqa_{split_file[:-5]}_{i:05d}",
                source        = "finqa",
                language      = "en",
                ticker        = "",
                company       = rec.get("id", "").split("-")[0] if rec.get("id") else "",
                period        = "",
                filing_type   = "10-K",
                filing_date   = "",
                text          = text,
                candidate_answer = str(rec.get("qa", {}).get("answer", "")),
            ))
    return records


def load_docfinqa(limit: int | None) -> list[dict]:
    """DocFinQA — long-context passages."""
    records = []
    p = Path("data/docfinqa/docfinqa_validation.jsonl")
    if not p.exists():
        return records
    with p.open() as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = rec.get("context") or ""
            if not text:
                continue
            records.append(passage_record(
                passage_id    = f"docfinqa_{i:05d}",
                source        = "docfinqa",
                language      = "en",
                ticker        = rec.get("cik") or "",
                company       = "",
                period        = rec.get("filing_date") or "",
                filing_type   = "10-K",
                filing_date   = rec.get("filing_date") or "",
                text          = text[:8000],
                candidate_answer = str(rec.get("answer") or ""),
            ))
    return records


def load_edgar(limit: int | None) -> list[dict]:
    """EDGAR XBRL-derived passages — contamination-safe, verified candidate answers.

    Prefers data/sources/edgar_passages.jsonl (produced by extract_edgar_passages.py)
    which has candidate_answer populated from XBRL facts.  Falls back to raw filings
    (no candidate_answer) only if the XBRL file is absent.
    """
    records = []

    # Preferred: XBRL-derived passages with verified answers
    xbrl_path = Path("data/sources/edgar_passages.jsonl")
    if xbrl_path.exists():
        with xbrl_path.open() as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    records.append(rec)  # keep _xbrl_hint for constructor to use
                except json.JSONDecodeError:
                    continue
        return records

    # Fallback: raw filings (no candidate_answer — skipped by construct_one guard)
    p = Path("data/edgar/edgar_filings.jsonl")
    if not p.exists():
        return records
    with p.open() as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sections = rec.get("sections") or {}
            text = (sections.get("item_7_mda")
                    or sections.get("item_8_financial_statements")
                    or sections.get("_full_text") or "")
            if not text:
                continue
            records.append(passage_record(
                passage_id    = f"edgar_{rec.get('ticker','')}_{i:04d}",
                source        = "edgar",
                language      = "en",
                ticker        = rec.get("ticker") or "",
                company       = rec.get("company_name") or "",
                period        = rec.get("period_of_report") or "",
                filing_type   = rec.get("form") or "10-K",
                filing_date   = rec.get("filing_date") or "",
                text          = text,
            ))
    return records


def load_cninfo_akshare(limit: int | None) -> list[dict]:
    """Pre-built ZH passages from akshare (pull_cninfo.py output)."""
    records = []
    p = Path("data/sources/zh_passages.jsonl")
    if not p.exists():
        return records
    with p.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                # Already in full passage schema — pass through, dropping _meta
                rec.pop("_meta", None)
                records.append(rec)
            except json.JSONDecodeError:
                continue
    return records


def load_zh_sources(limit: int | None) -> list[dict]:
    """FinEval + CFLUE + BBT-FinCUGE — ZH passages (MCQ benchmarks, mostly empty)."""
    records = []

    # FinEval: look for passage-level data
    fineval_dir = Path("FinEval")
    for p in fineval_dir.glob("**/*.jsonl"):
        with p.open() as f:
            for i, line in enumerate(f):
                if limit and len(records) >= limit:
                    break
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = (rec.get("context") or rec.get("passage")
                        or rec.get("text") or "")
                if not text:
                    continue
                records.append(passage_record(
                    passage_id    = f"fineval_{p.stem}_{i:04d}",
                    source        = "fineval",
                    language      = "zh",
                    ticker        = "",
                    company       = rec.get("company") or "",
                    period        = rec.get("year") or "",
                    filing_type   = "年报",
                    filing_date   = "",
                    text          = text,
                    candidate_answer = str(rec.get("answer") or ""),
                ))

    # CFLUE
    cflue_dir = Path("cflue")
    for p in cflue_dir.glob("**/*.jsonl"):
        with p.open() as f:
            for i, line in enumerate(f):
                if limit and len(records) >= limit:
                    break
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = (rec.get("context") or rec.get("passage") or "")
                if not text:
                    continue
                records.append(passage_record(
                    passage_id    = f"cflue_{p.stem}_{i:04d}",
                    source        = "cflue",
                    language      = "zh",
                    ticker        = "",
                    company       = "",
                    period        = "",
                    filing_type   = "年报",
                    filing_date   = "",
                    text          = text,
                    candidate_answer = str(rec.get("answer") or ""),
                ))
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(out: Path, limit: int | None):
    out.parent.mkdir(parents=True, exist_ok=True)

    loaders = [
        ("FinanceBench",   load_financebench),
        ("FinQA",          load_finqa),
        ("DocFinQA",       load_docfinqa),
        ("EDGAR",          load_edgar),
        ("CNINFO/akshare", load_cninfo_akshare),
        ("ZH sources",     load_zh_sources),
    ]

    total = 0
    with out.open("w", encoding="utf-8") as f:
        for name, loader in loaders:
            recs = loader(limit)
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  {name}: {len(recs)} passages")
            total += len(recs)

    print(f"\nTotal passages written: {total} → {out}")
    lang_en = lang_zh = 0
    with out.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("language") == "zh":
                lang_zh += 1
            else:
                lang_en += 1
    print(f"  EN: {lang_en}  ZH: {lang_zh}  ratio: {lang_en/(lang_en+lang_zh+1e-9):.0%}/{lang_zh/(lang_en+lang_zh+1e-9):.0%}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out",   type=Path, default=OUT_DEFAULT)
    p.add_argument("--limit", type=int,  default=None,
                   help="Cap per source (useful for testing)")
    args = p.parse_args()
    main(args.out, args.limit)
