"""
Pull ZH financial passages from A-share annual reports via akshare.

Generates three passage types covering the 5 ambiguity axes:
  - metric_definition : 净利润 vs 扣非净利润 (GAAP net profit vs adjusted excl. non-recurring)
  - entity_scope      : 归母净利润 vs 净利润 (parent-attributable vs consolidated incl. minority)
  - temporal_scope    : YoY revenue growth (ambiguous: FY vs quarterly period)

Output: data/sources/zh_passages.jsonl (same schema as passages.jsonl entries)

Usage:
    conda run -n finteract python scripts/pull_cninfo.py
    conda run -n finteract python scripts/pull_cninfo.py --limit 30 --out data/sources/zh_passages_test.jsonl
"""

import argparse
import json
import random
import time
from pathlib import Path

try:
    import akshare as ak
    import pandas as pd
except ImportError:
    raise SystemExit("Run: conda run -n finteract pip install akshare pandas")

OUT_DEFAULT = Path("data/sources/zh_passages.jsonl")

# Annual report periods we want (full-year only, skip quarterly for clean passages)
TARGET_PERIODS = {"20241231", "20231231", "20221231"}

# Minimum absolute difference threshold (as fraction) for a passage to be "interesting"
MIN_DIFF_FRAC = 0.03   # 3% difference between metric variants triggers a passage


def _format_yuan(value: float) -> str:
    """Format RMB value as human-readable '亿元' string."""
    yi = value / 1e8
    if abs(yi) >= 10000:
        return f"{yi / 10000:.2f}万亿元"
    if abs(yi) >= 0.1:
        return f"{yi:.2f}亿元"
    return f"{value / 1e4:.0f}万元"


def _get_financial_abstract(ticker: str) -> pd.DataFrame | None:
    try:
        df = ak.stock_financial_abstract(symbol=ticker)
        return df
    except Exception:
        return None


def _extract_metric(df: pd.DataFrame, category: str, metric: str,
                    period: str) -> float | None:
    """Extract a single value from stock_financial_abstract wide-format DataFrame."""
    mask = (df["选项"] == category) & (df["指标"] == metric)
    rows = df[mask]
    if rows.empty or period not in df.columns:
        return None
    val = rows.iloc[0][period]
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _yoy_period(period: str) -> str:
    """Return the prior-year annual period (e.g. '20241231' → '20231231')."""
    year = int(period[:4])
    return f"{year - 1}1231"


def build_metric_def_passage(ticker: str, company: str, df: pd.DataFrame,
                             period: str) -> dict | None:
    """
    metric_definition: 净利润 vs 扣非净利润 ambiguity.
    Ambiguity: "Company's FY net profit" — default = 归母净利润, intended = 扣非净利润.
    """
    parent_net = _extract_metric(df, "常用指标", "归母净利润", period)
    adj_net    = _extract_metric(df, "常用指标", "扣非净利润", period)
    revenue    = _extract_metric(df, "常用指标", "营业总收入", period)

    if parent_net is None or adj_net is None or revenue is None:
        return None
    if abs(parent_net) < 1e6 or abs(adj_net) < 1e6:
        return None
    diff_frac = abs(parent_net - adj_net) / abs(parent_net)
    if diff_frac < MIN_DIFF_FRAC:
        return None

    year = period[:4]
    rev_str        = _format_yuan(revenue)
    parent_str     = _format_yuan(parent_net)
    adj_str        = _format_yuan(adj_net)
    diff_str       = _format_yuan(abs(parent_net - adj_net))
    direction      = "高于" if parent_net > adj_net else "低于"

    passage_text = (
        f"{company}（{ticker}）{year}年年度报告主要财务数据如下：\n"
        f"营业总收入：{rev_str}；\n"
        f"净利润：{_format_yuan(_extract_metric(df, '常用指标', '净利润', period) or parent_net)}；\n"
        f"归属于上市公司股东的净利润（归母净利润）：{parent_str}；\n"
        f"归属于上市公司股东的扣除非经常性损益的净利润（扣非净利润）：{adj_str}。\n"
        f"归母净利润{direction}扣非净利润{diff_str}，差异主要来自非经常性损益项目，"
        f"包括政府补贴、资产处置收益、投资收益及公允价值变动损益等。"
    )

    candidate_answer = adj_str   # intended: 扣非净利润

    return {
        "passage_id":       f"cninfo_metric_{ticker}_{period}",
        "source":           "cninfo_akshare",
        "language":         "zh",
        "ticker":           ticker,
        "company":          company,
        "period":           period,
        "filing_type":      "年报",
        "filing_date":      f"{year}-04-30",
        "passage_text":     passage_text,
        "candidate_answer": candidate_answer,
        "candidate_axes":   ["metric_definition", "entity_scope"],
        "_meta": {
            "axis":        "metric_definition",
            "parent_net":  parent_net,
            "adj_net":     adj_net,
            "revenue":     revenue,
            "diff_frac":   round(diff_frac, 4),
        },
    }


def build_entity_scope_passage(ticker: str, company: str, df: pd.DataFrame,
                                period: str) -> dict | None:
    """
    entity_scope: 归母净利润 vs 净利润 (consolidated incl. minority).
    Ambiguity: "Company's FY net profit" — default = 净利润 (full consolidated),
    intended = 归母净利润 (parent-only).
    """
    total_net  = _extract_metric(df, "常用指标", "净利润", period)
    parent_net = _extract_metric(df, "常用指标", "归母净利润", period)
    revenue    = _extract_metric(df, "常用指标", "营业总收入", period)

    if total_net is None or parent_net is None or revenue is None:
        return None
    if abs(total_net) < 1e6:
        return None
    diff_frac = abs(total_net - parent_net) / abs(total_net)
    if diff_frac < MIN_DIFF_FRAC:
        return None

    year = period[:4]
    minority = total_net - parent_net
    minority_pct = abs(minority) / abs(total_net) * 100

    passage_text = (
        f"{company}（{ticker}）{year}年实现净利润（含少数股东损益）{_format_yuan(total_net)}，"
        f"其中归属于母公司股东的净利润（归母净利润）为{_format_yuan(parent_net)}，"
        f"少数股东损益为{_format_yuan(minority)}，"
        f"占净利润总额约{minority_pct:.1f}%。\n"
        f"公司合并报表范围包含多家控股子公司，少数股东权益占比较高，"
        f"导致合并净利润与归母净利润存在差异。"
        f"营业总收入为{_format_yuan(revenue)}。"
    )

    candidate_answer = _format_yuan(parent_net)   # intended: 归母净利润

    return {
        "passage_id":       f"cninfo_entity_{ticker}_{period}",
        "source":           "cninfo_akshare",
        "language":         "zh",
        "ticker":           ticker,
        "company":          company,
        "period":           period,
        "filing_type":      "年报",
        "filing_date":      f"{year}-04-30",
        "passage_text":     passage_text,
        "candidate_answer": candidate_answer,
        "candidate_axes":   ["entity_scope", "metric_definition"],
        "_meta": {
            "axis":        "entity_scope",
            "total_net":   total_net,
            "parent_net":  parent_net,
            "minority":    minority,
            "diff_frac":   round(diff_frac, 4),
        },
    }


def build_temporal_scope_passage(ticker: str, company: str, df: pd.DataFrame,
                                  period: str) -> dict | None:
    """
    temporal_scope: FY annual vs prior-year comparison.
    Ambiguity: "Company's revenue growth in FY{year}" — default = raw absolute figure,
    intended = YoY growth rate vs same period prior year.
    """
    prior_period = _yoy_period(period)
    revenue      = _extract_metric(df, "常用指标", "营业总收入", period)
    prior_rev    = _extract_metric(df, "常用指标", "营业总收入", prior_period)
    net_profit   = _extract_metric(df, "常用指标", "归母净利润", period)
    prior_profit = _extract_metric(df, "常用指标", "归母净利润", prior_period)

    if revenue is None or prior_rev is None or net_profit is None or prior_profit is None:
        return None
    if prior_rev == 0 or prior_profit == 0:
        return None

    year = period[:4]
    rev_yoy    = (revenue - prior_rev) / abs(prior_rev) * 100
    profit_yoy = (net_profit - prior_profit) / abs(prior_profit) * 100

    rev_dir    = "增长" if rev_yoy >= 0 else "下降"
    profit_dir = "增长" if profit_yoy >= 0 else "下降"

    passage_text = (
        f"{company}（{ticker}）{year}年实现营业总收入{_format_yuan(revenue)}，"
        f"同比{rev_dir}{abs(rev_yoy):.2f}%（{int(year)-1}年为{_format_yuan(prior_rev)}）。\n"
        f"归属于母公司股东的净利润为{_format_yuan(net_profit)}，"
        f"同比{profit_dir}{abs(profit_yoy):.2f}%（{int(year)-1}年为{_format_yuan(prior_profit)}）。\n"
        f"以上数据均来自{year}年年度报告合并报表，报告期为{year}年1月1日至{year}年12月31日。"
        f"公司采用合并报表口径披露，财务数据以人民币元为单位。"
    )

    # candidate_answer = YoY revenue growth rate
    sign = "+" if rev_yoy >= 0 else ""
    candidate_answer = f"{sign}{rev_yoy:.2f}%"

    return {
        "passage_id":       f"cninfo_temporal_{ticker}_{period}",
        "source":           "cninfo_akshare",
        "language":         "zh",
        "ticker":           ticker,
        "company":          company,
        "period":           period,
        "filing_type":      "年报",
        "filing_date":      f"{year}-04-30",
        "passage_text":     passage_text,
        "candidate_answer": candidate_answer,
        "candidate_axes":   ["temporal_scope", "metric_definition"],
        "_meta": {
            "axis":       "temporal_scope",
            "revenue":    revenue,
            "prior_rev":  prior_rev,
            "rev_yoy":    round(rev_yoy, 4),
            "profit_yoy": round(profit_yoy, 4),
        },
    }


def pull_company(ticker: str, company: str,
                 seen_tickers: set) -> list[dict]:
    """Fetch and build all passage types for one company."""
    if ticker in seen_tickers:
        return []

    df = _get_financial_abstract(ticker)
    if df is None:
        print(f"  [skip] {ticker} {company} — could not fetch data")
        return []

    records = []
    for period in sorted(TARGET_PERIODS, reverse=True):
        metric = build_metric_def_passage(ticker, company, df, period)
        entity = build_entity_scope_passage(ticker, company, df, period)
        temporal = build_temporal_scope_passage(ticker, company, df, period)
        for rec in (metric, entity, temporal):
            if rec is not None:
                records.append(rec)

    seen_tickers.add(ticker)
    return records


def main(out: Path, limit: int | None):
    out.parent.mkdir(parents=True, exist_ok=True)

    # Get 上证50 + 沪深300 subset for diversity
    print("Fetching index constituents …")
    tickers: list[tuple[str, str]] = []

    for symbol in ("000016", "000300"):   # 上证50, 沪深300
        try:
            df = ak.index_stock_cons(symbol=symbol)
            for _, row in df.iterrows():
                code = str(row["品种代码"]).zfill(6)
                name = str(row["品种名称"])
                tickers.append((code, name))
        except Exception as e:
            print(f"  Warning: failed to fetch index {symbol}: {e}")

    # Deduplicate, shuffle for diversity
    seen: dict[str, str] = {}
    for code, name in tickers:
        seen[code] = name
    unique = list(seen.items())
    random.seed(42)
    random.shuffle(unique)

    print(f"  {len(unique)} unique tickers across indices")

    records: list[dict] = []
    seen_tickers: set[str] = set()
    n_companies = 0

    for ticker, company in unique:
        if limit and len(records) >= limit:
            break
        recs = pull_company(ticker, company, seen_tickers)
        if recs:
            records.extend(recs)
            n_companies += 1
            print(f"  {ticker} {company}: {len(recs)} passages "
                  f"(axes: {[r['candidate_axes'][0] for r in recs]})")
        time.sleep(0.3)   # polite rate-limit

    # Shuffle and write
    random.shuffle(records)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    axis_counts: dict[str, int] = {}
    for r in records:
        ax = r.get("_meta", {}).get("axis", "unknown")
        axis_counts[ax] = axis_counts.get(ax, 0) + 1

    print(f"\nTotal ZH passages: {len(records)} from {n_companies} companies")
    print(f"Axis distribution: {axis_counts}")
    print(f"Output: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pull ZH passages from A-share data via akshare")
    p.add_argument("--out",   type=Path, default=OUT_DEFAULT)
    p.add_argument("--limit", type=int,  default=None,
                   help="Cap total passages (for testing)")
    args = p.parse_args()
    main(args.out, args.limit)
