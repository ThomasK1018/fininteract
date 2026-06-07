"""
Automated quality scorer for FinInteract pilot instances.
Applies rule-based checks without API calls.

Usage:
    python scripts/score_pilot.py data/constructed/pilot_v2.jsonl
"""

import json
import sys
import re
from pathlib import Path

FINANCIAL_KEYWORDS = {
    "revenue", "sales", "earnings", "income", "profit", "loss", "margin",
    "ebitda", "ebit", "eps", "cash flow", "debt", "equity", "assets",
    "liabilities", "return on", "ratio", "rate", "growth", "yield",
    "dividend", "share", "stock", "price", "market cap", "market value",
    "book value", "operating", "gross", "net", "free cash", "capex",
    "depreciation", "amortization", "interest", "tax", "segment",
    "revenue growth", "profit margin", "leverage", "turnover", "coverage",
    "capacity", "utilization", "outstanding", "diluted", "basic",
    "working capital", "current ratio", "quick ratio", "roe", "roa", "roc",
    "capital", "impairment", "goodwill", "intangible", "backlog",
}

ADMIN_KEYWORDS = {
    "page", "pages", "filing fee", "word count", "exhibit", "signature",
    "submission date", "registrant", "form type", "sec file", "fiscal year end",
    "address", "phone", "fax", "zip code", "state of incorporation",
}

DISAMBIG_TERMS = {
    "fiscal year", "fy20", "fy 20", "calendar year", "non-gaap", "adjusted",
    "consolidated", "segment", "amended", "restated", "original filing",
    "as reported", "organic", "constant currency",
}

VALID_AXES = {
    "temporal_scope", "metric_definition", "entity_scope",
    "filing_vintage", "recognition_policy",
}


def check_instance(inst: dict) -> dict[str, bool | str]:
    q = inst.get("question", "").lower()
    a = str(inst.get("answer", "")).strip()
    c = inst.get("context", "").lower()
    intended = inst.get("intended_interpretation", {})
    default  = inst.get("default_interpretation", {})
    axes     = inst.get("axes", [])

    checks = {}

    # R1: not a yes/no question
    checks["r1_not_yesno"] = not re.match(r"^(did|does|is|are|was|were|has|have|can|will|would|should)\b", q)

    # R2: no disambiguating terms in Q — also catches inline dates and segment names
    q_raw = inst.get("question", "").lower()
    has_disambig_term = any(t in q_raw for t in DISAMBIG_TERMS)
    # A month name or a 4-digit year in Q over-specifies temporal scope
    has_inline_date   = bool(re.search(
        r"\b(january|february|march|april|may|june|july|"
        r"august|september|october|november|december|"
        r"19\d{2}|20\d{2})\b", q_raw))
    # Naming a specific segment in Q over-specifies entity scope
    has_segment_name  = bool(re.search(r"\b\w+ (segment|division|subsidiary|unit)\b", q_raw))
    checks["r2_no_disambig_in_q"] = not (has_disambig_term or has_inline_date or has_segment_name)

    # R4: Q names a company — capitalized EN proper noun OR Chinese characters (ZH instances)
    _STOP = {
        "the", "a", "an", "for", "in", "at", "of", "by", "to", "is", "as",
        "are", "was", "were", "its", "inc", "and", "or", "but", "not", "how",
        "what", "which", "when", "where", "who", "why", "with", "from", "this",
        "that", "has", "have", "had", "does", "did", "do", "be", "been",
    }
    q_raw_r4 = inst.get("question", "")
    # ZH: presence of CJK characters implies the question names a Chinese entity
    has_cjk = bool(re.search(r"[一-鿿]", q_raw_r4))
    if has_cjk:
        has_company = True
    else:
        q_words = q_raw_r4.split()
        def _is_proper(w: str) -> bool:
            core = "".join(c for c in w if c.isalpha() or c == "-")
            if not core or core.lower() in _STOP:
                return False
            return len(core) >= 2 and core[0].isupper() and core[1:].replace("-", "").isalpha()
        has_company = any(_is_proper(w) for w in q_words[1:])
    checks["r4_names_company"] = has_company

    # R5: asks about a financial metric, not admin
    # Chinese financial terms: pass automatically if CJK characters present and
    # common financial metric words (利润/收入/增长/etc.) appear in the question
    _ZH_FINANCIAL = {
        "利润", "收入", "营业", "净利", "毛利", "增长", "亏损", "资产", "负债",
        "现金", "股本", "每股", "市值", "资本", "回报", "产能", "出货",
    }
    has_cjk = bool(re.search(r"[一-鿿]", q))
    if has_cjk:
        checks["r5_financial_metric"] = (
            any(kw in q for kw in _ZH_FINANCIAL)
            and not any(kw in q for kw in ADMIN_KEYWORDS)
        )
    else:
        checks["r5_financial_metric"] = (
            any(kw in q for kw in FINANCIAL_KEYWORDS)
            and not any(kw in q for kw in ADMIN_KEYWORDS)
        )

    # R6: A type consistent with Q (count Q → integer A)
    count_q = any(w in q for w in ("how many", "number of", "count of"))
    pct_a   = "%" in a
    checks["r6_type_consistent"] = not (count_q and pct_a)

    # R7: C does not contain answer value — use word-boundary to avoid false positives
    # on short answers like "1" matching "2018" or "2" matching "2.5B"
    a_stripped = re.sub(r"[$%,\s]", "", a).lower()
    if len(a_stripped) >= 3:
        checks["r7_c_no_answer"] = a_stripped not in c.replace(",", "").replace(" ", "")
    else:
        # Very short answers: require word-boundary match to avoid false positives
        checks["r7_c_no_answer"] = not bool(re.search(rf"\b{re.escape(a_stripped)}\b", c))

    # R8: intended ≠ default (real ambiguity exists)
    checks["r8_real_ambiguity"] = (
        intended.get("entity")  != default.get("entity")  or
        intended.get("period")  != default.get("period")  or
        intended.get("metric")  != default.get("metric")  or
        intended.get("basis")   != default.get("basis")
    )

    # R9: axes are valid and non-empty
    checks["r9_valid_axes"] = bool(axes) and all(a in VALID_AXES for a in axes)

    # R10: answer is non-trivial — not empty or bare punctuation
    checks["r10_nontrivial_answer"] = len(a) >= 1 and a not in {".", ",", "-", "0", ""}

    # R11 (reviewer addition): default_answer present and differs from intended answer
    default_ans = str(inst.get("default_answer", "")).strip()
    checks["r11_default_answer"] = bool(default_ans) and default_ans != a

    # R12 (reviewer addition): both evidence spans present and non-empty
    intended_span = str(inst.get("intended_evidence_span", "")).strip()
    default_span  = str(inst.get("default_evidence_span", "")).strip()
    checks["r12_evidence_spans"] = bool(intended_span) and bool(default_span)

    # R13 (reviewer addition): evidence spans must be meaningfully distinct
    if intended_span and default_span:
        s1 = " ".join(intended_span.lower().split())
        s2 = " ".join(default_span.lower().split())
        common_prefix = sum(1 for c1, c2 in zip(s1, s2) if c1 == c2)
        shorter = min(len(s1), len(s2))
        too_similar = (s1 == s2) or (shorter > 20 and common_prefix / shorter > 0.85)
        checks["r13_distinct_spans"] = not too_similar
    else:
        checks["r13_distinct_spans"] = False

    passed  = sum(1 for v in checks.values() if v is True)
    total   = len(checks)
    score   = passed / total
    failing = [k for k, v in checks.items() if v is False]

    return {
        "instance_id": inst.get("instance_id"),
        "question":    inst.get("question"),
        "answer":      a,
        "axes":        axes,
        "score":       round(score, 2),
        "passed":      passed,
        "total":       total,
        "failing":     failing,
    }


def main(path: str):
    p = Path(path)
    if not p.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    instances = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))

    if not instances:
        print("No instances found.")
        return

    results = [check_instance(i) for i in instances]

    # Summary stats
    scores     = [r["score"] for r in results]
    perfect    = sum(1 for r in results if r["score"] == 1.0)
    mean_score = sum(scores) / len(scores)

    # Axis distribution
    axis_counts: dict[str, int] = {}
    for inst in instances:
        for ax in inst.get("axes", []):
            axis_counts[ax] = axis_counts.get(ax, 0) + 1

    # Failure frequency
    fail_counts: dict[str, int] = {}
    for r in results:
        for f in r["failing"]:
            fail_counts[f] = fail_counts.get(f, 0) + 1

    print(f"\n{'='*60}")
    print(f"Pilot quality report: {p.name}  ({len(instances)} instances)")
    print(f"{'='*60}")
    print(f"Mean rule-pass score : {mean_score:.2%}")
    print(f"Perfect instances    : {perfect}/{len(instances)}")
    print(f"\nAxis distribution:")
    for ax, cnt in sorted(axis_counts.items(), key=lambda x: -x[1]):
        print(f"  {ax:<25} {cnt}")
    print(f"\nFailure frequency (by rule):")
    for rule, cnt in sorted(fail_counts.items(), key=lambda x: -x[1]):
        print(f"  {rule:<30} {cnt}/{len(instances)}")
    print(f"\nPer-instance scores:")
    for r in results:
        mark = "✓" if r["score"] == 1.0 else "✗"
        flag = f"  [{', '.join(r['failing'])}]" if r["failing"] else ""
        print(f"  {mark} {r['instance_id']}  score={r['score']:.0%}  axes={r['axes']}{flag}")
        print(f"      Q: {r['question']}")
        print(f"      A: {r['answer']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <pilot.jsonl>")
        sys.exit(1)
    main(sys.argv[1])
