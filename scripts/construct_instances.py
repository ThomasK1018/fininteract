"""
FinInteract construction pipeline.

Roles
-----
Constructor  : Claude-Opus-4 (or GPT-5) — generates (Q, C, A, metadata) from
               a filing passage and a target ambiguity axis set.
Verifier     : GPT-4o-mini + Claude-Sonnet-4 — 5-round adversarial trials.
               Reject if ≥2 of 10 trials (5 per model) answer correctly
               WITHOUT the disambiguating context C.

Input  : data/sources/passages.jsonl  (one passage per line, see schema below)
Output : data/constructed/instances.jsonl

Usage
-----
    pip install anthropic openai tqdm
    export ANTHROPIC_API_KEY=...
    export OPENAI_API_KEY=...

    # Dry-run on 5 passages
    python scripts/construct_instances.py --source data/sources/passages.jsonl \
        --out data/constructed/instances.jsonl --limit 5 --dry-run

    # Full run targeting 500 accepted instances
    python scripts/construct_instances.py --source data/sources/passages.jsonl \
        --out data/constructed/instances.jsonl --target 500

Passage input schema (one JSON per line)
-----------------------------------------
{
  "passage_id":   "financebench_042",
  "source":       "financebench | finqa | edgar | fineval | cflue",
  "language":     "en" | "zh",
  "ticker":       "AAPL",
  "company":      "Apple Inc.",
  "period":       "FY2024",
  "filing_type":  "10-K",
  "filing_date":  "2024-11-01",
  "passage_text": "...",          # the filing excerpt fed to the constructor
  "candidate_answer": "73.9%",   # known correct answer (for grader verification)
  "candidate_axes":  ["temporal_scope", "entity_scope"]  # axes to exercise
}

Output instance schema
-----------------------
{
  "instance_id":    "fininteract_001",
  "passage_id":     "financebench_042",
  "language":       "en",
  "question":       "...",
  "context":        "...",
  "answer":         "73.9%",
  "intended_interpretation":  {"entity": ..., "period": ..., "metric": ..., "basis": ...},
  "default_interpretation":   {"entity": ..., "period": ..., "metric": ..., "basis": ...},
  "axes":           ["temporal_scope", "entity_scope"],
  "n_axes":         2,
  "h0":             2.0,     # log2 bits of interpretation entropy
  "source":         "financebench",
  "ticker":         "AAPL",
  "company":        "Apple Inc.",
  "filing_type":    "10-K",
  "filing_date":    "2024-11-01",
  "qc": {
    "constructor_attempts":  1,
    "verifier_reject_rate":  0.2,   # fraction of verifier trials that answered correctly
    "accepted":              true
  }
}
"""

import argparse
import json
import os
import re
import sys
import time
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    sys.exit("pip install openai")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from dise import compute_h0, DEFAULT_AXIS_OPTIONS


# ---------------------------------------------------------------------------
# Tickers with non-calendar fiscal years — the ONLY ones where a yoy EDGAR
# temporal_scope passage has genuine FY vs CY ambiguity.
# ---------------------------------------------------------------------------
_NON_CALENDAR_FY_TICKERS = {
    "AAPL", "MSFT", "WMT", "NKE", "ORCL", "ADBE", "COST", "HPE",
    "FDX", "ACN", "MU", "BBY", "DLTR", "TJX", "HD", "LOW", "TGT",
    "INTU", "WDAY", "ADSK", "DE", "EMR", "PH",
}

# ---------------------------------------------------------------------------
# DocFinQA / EDGAR axis pre-scan: skip passages unlikely to yield a valid
# instance, saving API budget before the constructor call.
# ---------------------------------------------------------------------------
_AXIS_SIGNALS: dict[str, re.Pattern] = {
    "temporal_scope": re.compile(
        r"fiscal year|calendar year|Q[1-4]\s*\d{4}|quarter(?:ly)?|"
        r"year[-\s]over[-\s]year|YoY|prior year|same period|"
        r"year ended|twelve months|TTM|full.year vs", re.I),
    "metric_definition": re.compile(
        r"non-GAAP|non\s+GAAP|adjusted|organic|constant.currency|"
        r"as.reported|EBITDA|operating income|pro.forma|excluding|"
        r"reconciliation", re.I),
    "filing_vintage": re.compile(
        r"amend|restat|revision|10-K/A|10-Q/A|as restated|"
        r"originally reported|prior filing|correction|supersed", re.I),
    "recognition_policy": re.compile(
        r"point.in.time|over time|over-time|recogni[sz]ed|recognition|"
        r"gross.*net revenue|net.*gross revenue|capitali[sz]ed|expensed|"
        r"deferred revenue|ASC 606|IFRS 15", re.I),
    "entity_scope": re.compile(
        r"segment|division|consolid|subsidiary|parent company|geographic|"
        r"Class [AB]|wholly.owned|joint venture|non.controlling", re.I),
}


_CURRENCY_RE = re.compile(r'\$\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|B|M|K)?\b', re.I)
_PERCENT_RE  = re.compile(r'\b\d+(?:\.\d+)?\s*%')


def _parse_numeric(s: Any) -> float | None:
    """Parse a financial answer string into a comparable float.

    Handles $-prefixes, commas, %, B/M/K suffixes, and CJK 亿/万 units.
    Returns None if no number can be extracted.
    """
    s = str(s).strip()
    m = re.search(r'-?[\d,]+(?:\.\d+)?', s.replace(",", ""))
    if not m:
        # retry without comma-stripping artifacts
        m = re.search(r'-?[\d.]+', s)
        if not m:
            return None
    val = float(m.group().replace(",", ""))
    sl = s.lower()
    if "亿" in s:
        val *= 1e8
    elif "万" in s:
        val *= 1e4
    elif "b" in sl or "billion" in sl:
        val *= 1e9
    elif ("m" in sl or "million" in sl) and "%" not in s:
        val *= 1e6
    elif "k" in sl and "%" not in s:
        val *= 1e3
    return val


def _answer_gap(answer: Any, default_answer: Any) -> float | None:
    """Absolute numeric gap between two answer strings, or None if not comparable.

    For percentage answers this is the gap in percentage points; for currency/value
    answers it is the gap in the underlying unit.
    """
    a = _parse_numeric(answer)
    d = _parse_numeric(default_answer)
    if a is None or d is None:
        return None
    return abs(a - d)


def _passes_axis_prescan(passage: dict) -> bool:
    """Return False for DocFinQA passages that are unlikely to yield a valid instance.

    EDGAR, CNINFO, edgar_amendment, edgar_8k, and other structured sources always
    return True because their ambiguity is guaranteed by the extraction pipeline.

    For DocFinQA we apply three gates:
    1. Hard axis exclusion: filing_vintage and recognition_policy require paired
       original/restated or gross/net values that DocFinQA passages never contain
       — skip all DocFinQA passages assigned to these axes.
    2. Axis keyword signal must be present in the text.
    3. At least two distinct currency/percentage values must appear — the
       constructor requires both an intended and a default answer.
    """
    source = passage.get("source", "")
    primary_axis = (passage.get("candidate_axes") or [""])[0]

    # --- EDGAR temporal_scope gate ---
    # Calendar-FY EDGAR yoy/fy_temporal passages have no genuine FY/CY ambiguity.
    # Only apply to passage types designed for temporal comparison, not to
    # metric_definition or entity_scope passages where temporal_scope is a
    # secondary axis that got promoted by the priority sort.
    pid = passage.get("passage_id", "")
    if source == "edgar" and primary_axis == "temporal_scope" and (
        pid.endswith("_yoy") or pid.endswith("_fy_temporal")
    ):
        ticker = passage.get("ticker", "").upper()
        hint = passage.get("_xbrl_hint") or {}
        is_non_cal = hint.get("non_calendar_fy") or ticker in _NON_CALENDAR_FY_TICKERS
        if not is_non_cal:
            return False

    # --- EDGAR yoy metric_definition gate ---
    # A yoy (growth-rate) passage forces a "growth rate" question. Adding a
    # metric_definition axis to it produces a "growth rate under basis X vs basis Y"
    # question whose two answers (e.g. GAAP net-income growth vs GAAP EPS growth)
    # collapse to within grader tolerance — a non-discriminating instance that R14
    # would reject post-construction. Skip here to avoid wasting constructor budget.
    # metric_definition is routed to value-based passages (_seg, amendment, 8-K) instead.
    if source == "edgar" and primary_axis == "metric_definition" and pid.endswith("_yoy"):
        return False

    if source != "docfinqa":
        return True

    # --- DocFinQA gates ---
    # Gate 1: axes where DocFinQA almost never works — skip entirely, no API call.
    if primary_axis in ("filing_vintage", "recognition_policy", "temporal_scope"):
        return False

    text = passage.get("passage_text", "")

    # Gate 2: axis-specific keyword
    pattern = _AXIS_SIGNALS.get(primary_axis)
    if pattern and not pattern.search(text):
        return False

    # Gate 3: at least two distinct numeric values (otherwise no default/intended pair)
    amounts = _CURRENCY_RE.findall(text) + _PERCENT_RE.findall(text)
    if len(set(amounts)) < 2:
        return False

    return True


# ---------------------------------------------------------------------------
# Model identifiers — OpenAI only
# ---------------------------------------------------------------------------
CONSTRUCTOR_MODEL   = "gpt-5"         # strong reasoning for generation
VERIFIER_MODEL_A    = "gpt-5-mini"    # fast/cheap adversarial trials
VERIFIER_MODEL_B    = "gpt-5"         # stronger adversarial trials (catches more)
VERIFIER_ROUNDS     = 5               # trials per verifier model (10 total)
REJECT_THRESHOLD    = 2               # reject if ≥ this many pass w/o context

MAX_CONSTRUCTOR_RETRIES = 3

# ---------------------------------------------------------------------------
# Constructor prompt
# ---------------------------------------------------------------------------
CONSTRUCTOR_SYSTEM = """\
You are a financial benchmark constructor building FinInteract, a benchmark that tests
whether LLM agents can recognize ambiguous financial queries and resolve them through
strategic yes/no interaction.

Design principle: "Easy to verify, Ambiguous to resolve."
  - The question Q must be answerable in multiple plausible ways from the passage.
  - The context C must be the minimal set of attributes that uniquely collapses Q to answer A.
  - C must NOT contain the answer value itself — only scope/definitional constraints.
  - A person reading Q alone (without C) must find it genuinely ambiguous.

Hard rules for Q (question):
  1. Q must require a specific value (number, name, or metric) to answer — NEVER a yes/no or
     true/false question. Bad: "Did revenue grow?" Good: "What was the revenue growth rate?"
  2. Q must be built from SHARED attributes only. Do NOT include disambiguating terms in Q.
     NEVER include ANY of these words in Q: fiscal year, FY, calendar year, quarter, TTM,
     consolidated, segment, division, subsidiary, geographic, amended, restated, original,
     adjusted, non-GAAP, organic, constant currency, as-reported, EBITDA, pro forma, or any
     specific date, year, or period (e.g. "in 2024", "as of June 30", "Q3").
     Bad: "What was Apple's fiscal-year revenue?" (fiscal-year disambiguates temporal scope).
     Bad: "What was Apple's revenue as of June 30, 2017?" (date disambiguates temporal scope).
     Bad: "What was Apple's Electrical Products segment revenue?" (segment name disambiguates entity scope).
     Bad: "What was Apple's revenue in 2024?" (year disambiguates temporal scope).
     Bad: "What was Everest Re Group's consolidated earnings?" (consolidated disambiguates entity scope).
     Bad: "What was Marathon Oil's restated revenue?" (restated disambiguates filing_vintage).
     Bad: "What was Company X's adjusted EBITDA?" (adjusted/EBITDA disambiguates metric_definition).
     Good: "What was Apple's revenue?" — The disambiguation belongs entirely in C, not Q.
  3. Use the provided candidate_answer field as A verbatim. Do NOT infer or generate your own
     answer value from the passage text.
  4. Q MUST name the company (legal name or ticker). Never use "the company" or omit the
     entity. Bad: "What was the revenue growth rate?" Good: "What was Citigroup's revenue
     growth rate?"
  5. Q must ask about a SUBSTANTIVE FINANCIAL METRIC: revenue, earnings, profit, margin,
     ratio, EPS, cash flow, debt, return on equity, etc. NEVER ask about administrative
     details like page count, filing date, filing fee, word count, or regulatory boilerplate.

Hard rules for A (answer):
  6. A must be type-consistent with Q. If Q asks "how much" or "what rate/margin/ratio",
     A must be a number with units (e.g., "73.9%", "$4.2B"). If Q asks "how many" (a count),
     A must be an integer, not a percentage.

Hard rule for C (context):
  7. C must NOT contain the answer value. C provides only scope/definitional constraints that
     a user would supply when asked "which interpretation do you mean?"
  8. C must describe the INTENDED interpretation (the one that yields A), not the default one.
     Bad: C says "Class A shares" when A is the Class B share count.
     Good: C says "Class B common shares" to collapse Q to the intended answer.
"""

CONSTRUCTOR_TEMPLATE = """\
## Filing Passage
{passage_text}

## Target
- Company: {company} ({ticker})
- Period: {period}
- Filing type: {filing_type}
- Correct answer: {candidate_answer}
- PRIMARY ambiguity axis (MANDATORY — see axis-specific requirements below): {primary_axis}
- Secondary axes (exercise only if genuinely supported by the passage): {secondary_axes}
{xbrl_section}

## Axis-specific MANDATORY requirement for {primary_axis}

{axis_specific_instruction}

If the passage does NOT support a genuine {primary_axis} ambiguity, output:
{{"error": "no viable {primary_axis} ambiguity in this passage"}}
Do NOT generate an instance using a different axis instead.

## Axis definitions (for reference)
- temporal_scope     → interpretations differ on "period" (FY vs CY, Q3 vs annual, TTM vs point-in-time)
- metric_definition  → interpretations differ on "basis" (GAAP vs non-GAAP, organic vs as-reported, adjusted EBITDA vs operating income)
- entity_scope       → interpretations differ on "entity" (segment vs consolidated, parent vs subsidiary, share class)
- filing_vintage     → interpretations differ on "basis" citing the filing version (original 10-K vs 10-K/A, restated vs as-reported)
- recognition_policy → interpretations differ on "basis" citing the accounting policy (point-in-time vs over-time, gross vs net)

## Task
Produce a (Q, C, A) instance following the default-vs-intended interpretation pattern:
  - intended_interpretation: the specific reading that yields the correct answer
  - default_interpretation: the plausible reading a non-expert would assume
  - default_answer: the CONCRETE VALUE (number, %, name) the question would return under the
    DEFAULT interpretation. This must be a different value from answer. Derive it from the passage.
  - intended_evidence_span: a verbatim quote (≤ 60 words) from the passage that supports answer
    under the intended interpretation.
  - default_evidence_span: a verbatim quote (≤ 60 words) from the passage that supports
    default_answer under the default interpretation.

If the passage does NOT contain enough information to produce BOTH evidence spans and BOTH answers,
output: {{"error": "no viable {primary_axis} ambiguity in this passage"}}

Output ONLY valid JSON (no markdown fences, no commentary):
{{
  "question": "...",
  "context": "...",
  "answer": "...",
  "default_answer": "...",
  "intended_evidence_span": "...",
  "default_evidence_span": "...",
  "intended_interpretation": {{
    "entity": "...",
    "period": "...",
    "metric": "...",
    "basis": "..."
  }},
  "default_interpretation": {{
    "entity": "...",
    "period": "...",
    "metric": "...",
    "basis": "..."
  }},
  "shared_attributes": ["...", "..."],
  "distinctive_attributes": ["...", "..."],
  "axes_exercised": ["..."]
}}

axes_exercised must list ONLY the axes that are ACTUALLY exercised (where intended ≠ default).
axes_exercised values must be EXACT strings: temporal_scope | metric_definition | entity_scope | filing_vintage | recognition_policy
The PRIMARY axis ({primary_axis}) MUST appear in axes_exercised.
"""

# Per-axis specific instructions injected into CONSTRUCTOR_TEMPLATE
_AXIS_REQUIREMENTS = {
    "metric_definition": (
        "METRIC_DEFINITION: The question must be ambiguous between two metric definitions:\n"
        "  - Examples: GAAP net income vs non-GAAP adjusted income; revenue as-reported vs organic;\n"
        "    operating income vs adjusted EBITDA; diluted EPS vs adjusted EPS.\n"
        "  - intended_interpretation.basis = specific metric basis (e.g. 'GAAP net income')\n"
        "  - default_interpretation.basis = alternative reasonable basis (e.g. 'non-GAAP adjusted')\n"
        "  - MUST: axes_exercised includes 'metric_definition'\n"
        "  - DO NOT generate entity_scope or filing_vintage ambiguity for this instance."
    ),
    "recognition_policy": (
        "RECOGNITION_POLICY: The question must be ambiguous about an accounting policy choice:\n"
        "  - Examples: revenue recognized point-in-time vs over-time; gross vs net revenue;\n"
        "    capitalized vs expensed costs; impairment timing; related-party at cost vs market.\n"
        "  - intended_interpretation.basis = specific policy applied\n"
        "  - default_interpretation.basis = alternative policy a non-expert would assume\n"
        "  - MUST: axes_exercised includes 'recognition_policy'\n"
        "  - DO NOT generate entity_scope or filing_vintage ambiguity for this instance."
    ),
    "temporal_scope": (
        "TEMPORAL_SCOPE: The question must be ambiguous about WHICH TIME PERIOD is assumed:\n"
        "  - Examples: fiscal year vs calendar year; Q3 results vs full-year; TTM vs point-in-time;\n"
        "    reported period vs comparable period; YoY calculation start year.\n"
        "  - intended_interpretation.period = specific time period (e.g. 'FY2023 ending Sep 30')\n"
        "  - default_interpretation.period = period a non-expert would assume (e.g. 'calendar year 2023')\n"
        "  - MUST: axes_exercised includes 'temporal_scope'\n"
        "  - DO NOT generate entity_scope ambiguity for this instance."
    ),
    "entity_scope": (
        "ENTITY_SCOPE: The question must be ambiguous about WHICH ENTITY or scope level is meant:\n"
        "  - Examples: consolidated total vs specific segment; parent company vs subsidiary;\n"
        "    Class A vs Class B shares; geographic region vs worldwide.\n"
        "  - intended_interpretation.entity = specific entity (e.g. 'Retail segment only')\n"
        "  - default_interpretation.entity = entity a non-expert would assume (e.g. 'consolidated total')\n"
        "  - MUST: axes_exercised includes 'entity_scope'"
    ),
    "filing_vintage": (
        "FILING_VINTAGE: The question must be ambiguous about WHICH FILING VERSION is used:\n"
        "  - Examples: original 10-K vs amended 10-K/A; as-reported vs restated figures;\n"
        "    first filed vs corrected; preliminary vs final.\n"
        "  - intended_interpretation.basis = specific filing (e.g. 'original 10-K filing')\n"
        "  - default_interpretation.basis = version a non-expert would assume (e.g. 'latest/amended filing')\n"
        "  - MUST: axes_exercised includes 'filing_vintage'"
    ),
}

# ---------------------------------------------------------------------------
# Adversarial verifier prompt
# Verifier answers WITHOUT context C and also states its assumed interpretation.
# A trial counts as a rejection vote ONLY IF both: (a) answer is correct, AND
# (b) the assumed interpretation aligns with the intended interpretation — this
# prevents false rejections from lucky guesses that happen to produce the right
# number under a different interpretation.
# ---------------------------------------------------------------------------
VERIFIER_SYSTEM = """\
You are answering a financial question using your knowledge and reasoning.
Output ONLY valid JSON — no markdown fences, no commentary.
"""

VERIFIER_TEMPLATE = """\
Question: {question}

Answer this question as best you can. Output ONLY valid JSON:
{{
  "answer": "<short answer — value, name, or metric>",
  "assumed_period": "<time period you assumed, e.g. 'FY2023 ending September' or 'calendar year 2023'>",
  "assumed_entity": "<entity/scope you assumed, e.g. 'consolidated' or 'Retail segment'>",
  "assumed_metric": "<metric definition you assumed, e.g. 'GAAP net income' or 'adjusted EPS'>",
  "assumed_basis": "<accounting basis you assumed, e.g. 'GAAP' or 'non-GAAP adjusted'>"
}}
If you are uncertain about a field, write "unspecified" for that field.
"""

# Grader prompt — checks if verifier answer matches the correct answer value
GRADER_SYSTEM = """\
You are a strict financial answer grader. Respond with ONLY 'yes' or 'no'.
"""

GRADER_TEMPLATE = """\
Question: {question}
Correct answer: {correct_answer}
Model answer: {model_answer}

Grading rules:
1. Entity must match (name, ticker, or Chinese/English equivalent).
2. Numeric tolerance: ±1% relative error for rounding.
3. Different fiscal year vs calendar year = 'no' even if numerically close.
4. Different reporting basis (GAAP vs non-GAAP, segment vs consolidated) = 'no'.
5. Currency/unit mismatch = 'no'.

Respond with ONLY 'yes' or 'no'.
"""


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------
def get_client() -> OpenAI:
    oai_key = os.environ.get("OPENAI_API_KEY")
    if not oai_key:
        sys.exit("Set OPENAI_API_KEY")
    return OpenAI(api_key=oai_key, timeout=600.0)  # 10-min hard cap per call


def get_model_client(model: str) -> OpenAI:
    """Provider-aware OpenAI-SDK client for the constructor-ablation study.

    Routing by model id:
      - contains "/"  (e.g. "anthropic/claude-3.7-sonnet", "google/gemini-2.5-pro",
                       "openai/gpt-5", "deepseek/deepseek-chat") -> OpenRouter gateway,
        a single OpenAI-compatible endpoint that fronts Claude, Gemini, DeepSeek, Qwen, etc.
      - "deepseek..." -> DeepSeek native;  "qwen..." -> DashScope;  "glm..." -> Zhipu
      - otherwise -> native OpenAI
    Using OpenRouter for the exotic providers keeps everything on the OpenAI SDK (one client
    type), so Claude/Gemini need no extra SDKs.
    """
    timeout = 600.0
    if "/" in model:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            sys.exit("Set OPENROUTER_API_KEY for namespaced models like "
                     "'anthropic/...', 'google/...'.")
        return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", timeout=timeout)
    ml = model.lower()
    if "deepseek" in ml:
        return OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
                      base_url="https://api.deepseek.com/v1", timeout=timeout)
    if "qwen" in ml:
        return OpenAI(api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
                      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", timeout=timeout)
    if "glm" in ml:
        return OpenAI(api_key=os.environ.get("ZHIPU_API_KEY", ""),
                      base_url="https://open.bigmodel.cn/api/paas/v4", timeout=timeout)
    return get_client()


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------
def _build_xbrl_section(passage: dict) -> str:
    """Build an optional XBRL disambiguation hint for EDGAR passages."""
    hint = passage.get("_xbrl_hint")
    if not hint:
        return ""
    ctype = hint.get("concept_type", "metric")
    lines = [f"\n## XBRL disambiguation hint (for entity/temporal scope construction)"]
    lines.append(f"- Metric type: {ctype}")
    if "consolidated" in hint:
        lines.append(f"- Consolidated value: {hint['consolidated']}")
    if "segment" in hint and "segment_val" in hint:
        lines.append(f"- Key segment: {hint['segment']} = {hint['segment_val']}")
        lines.append(f"  Use this to construct entity_scope ambiguity: Q asks about the "
                     f"company's {ctype}; intended interpretation = {hint['segment']} "
                     f"segment; default interpretation = consolidated total.")
    if "curr_fy" in hint and "prev_fy" in hint:
        lines.append(f"- FY{hint['curr_fy']}: {hint.get('curr_val','')}  "
                     f"FY{hint['prev_fy']}: {hint.get('prev_val','')}  "
                     f"YoY growth: {hint.get('yoy_growth','')}")
        lines.append(f"  Use this to construct temporal_scope ambiguity: Q asks about "
                     f"{ctype} growth; intended = FY{hint['curr_fy']} vs prior year; "
                     f"default = prior year or calendar year.")
    return "\n".join(lines)


def call_constructor(passage: dict, oai: OpenAI,
                     retry: int = 0, model: str | None = None) -> dict | None:
    model = model or CONSTRUCTOR_MODEL
    axes_list = passage.get("candidate_axes", [])
    primary_axis            = axes_list[0] if axes_list else "metric_definition"
    secondary_axes          = ", ".join(axes_list[1:]) if len(axes_list) > 1 else "none"
    axis_specific_instruction = _AXIS_REQUIREMENTS.get(primary_axis,
        f"Exercise {primary_axis} ambiguity as defined above.")
    # Suppress XBRL entity/temporal hints when primary axis is metric_definition or
    # recognition_policy — the hints would push the constructor off-target
    if primary_axis in ("metric_definition", "recognition_policy"):
        xbrl_sec = ""
    else:
        xbrl_sec = _build_xbrl_section(passage)
    lang = passage.get("language", "en")
    lang_note = (
        "\n## Language requirement\nThe passage is in Chinese. "
        "Generate question, context, and all string fields IN CHINESE (普通话). "
        "Rule 4 (name the company): use the Chinese company name, e.g. '中煤能源'.\n"
        if lang == "zh" else ""
    )
    prompt = CONSTRUCTOR_TEMPLATE.format(
        passage_text              = passage["passage_text"][:12_000] + lang_note,
        company                   = passage.get("company", ""),
        ticker                    = passage.get("ticker", ""),
        period                    = passage.get("period", ""),
        filing_type               = passage.get("filing_type", ""),
        candidate_answer          = passage.get("candidate_answer", ""),
        primary_axis              = primary_axis,
        secondary_axes            = secondary_axes,
        axis_specific_instruction = axis_specific_instruction,
        xbrl_section              = xbrl_sec,
    )
    # Native OpenAI gpt-5/gpt-4o support max_completion_tokens + JSON mode; other providers
    # (Claude/Gemini/open-weight via OpenRouter or compatible APIs) use max_tokens and may not
    # support response_format — fall back gracefully.
    is_native_openai = ("/" not in model) and model.startswith(("gpt-", "o1", "o3", "o4"))
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": CONSTRUCTOR_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
    }
    if is_native_openai:
        kwargs["max_completion_tokens"] = 4096
        kwargs["response_format"] = {"type": "json_object"}
    else:
        kwargs["max_tokens"] = 4096
    try:
        resp = oai.chat.completions.create(**kwargs)
        raw = (resp.choices[0].message.content or "").strip()
        # strip markdown fences some non-OpenAI models add despite instructions
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1].lstrip("json").strip() if "```" in raw[3:] else raw.strip("`")
        if not raw:
            raise json.JSONDecodeError("empty response", "", 0)
        # tolerate leading prose before the JSON object
        if not raw.startswith("{"):
            b = raw.find("{")
            if b != -1:
                raw = raw[b:raw.rfind("}") + 1]
        return json.loads(raw)
    except (json.JSONDecodeError, IndexError) as e:
        if retry < MAX_CONSTRUCTOR_RETRIES:
            time.sleep(2)
            return call_constructor(passage, oai, retry + 1, model=model)
        print(f"  [constructor] JSON parse failed after {MAX_CONSTRUCTOR_RETRIES} retries: {e}")
        return None
    except Exception as e:
        if "rate" in str(e).lower() and retry < MAX_CONSTRUCTOR_RETRIES:
            time.sleep(10)
            return call_constructor(passage, oai, retry + 1, model=model)
        print(f"  [constructor] API error: {e}")
        return None


# ---------------------------------------------------------------------------
# Adversarial verifier + grader — OpenAI only
# ---------------------------------------------------------------------------
def _interp_overlap(assumed: dict, intended: dict) -> bool:
    """
    Check if the verifier's assumed interpretation aligns with the intended one.
    Returns True if at least one non-trivial field matches (case-insensitive substring).
    Fields: period, entity, metric, basis.
    'unspecified' fields are skipped.
    """
    fields = [("assumed_period",  "period"),
              ("assumed_entity",  "entity"),
              ("assumed_metric",  "metric"),
              ("assumed_basis",   "basis")]
    for assumed_key, intended_key in fields:
        av = str(assumed.get(assumed_key, "")).lower().strip()
        iv = str(intended.get(intended_key, "")).lower().strip()
        if not av or av == "unspecified" or not iv or iv == "unspecified":
            continue
        # Substring overlap in either direction (≥5 chars to avoid noise)
        if len(av) >= 5 and len(iv) >= 5:
            if av in iv or iv in av:
                return True
        # Word-level overlap: share ≥1 meaningful word (>4 chars)
        av_words = {w for w in av.split() if len(w) > 4}
        iv_words = {w for w in iv.split() if len(w) > 4}
        if av_words & iv_words:
            return True
    return False


def verifier_answer(question: str, oai: OpenAI, model: str) -> dict:
    """
    Call the verifier for one trial. Returns parsed JSON dict with
    'answer' and assumed interpretation fields.
    Falls back to {'answer': raw_text} on parse failure.
    """
    try:
        resp = oai.chat.completions.create(
            model      = model,
            max_completion_tokens = 1024,
            messages   = [
                {"role": "system", "content": VERIFIER_SYSTEM},
                {"role": "user",   "content": VERIFIER_TEMPLATE.format(question=question)},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        return json.loads(raw)
    except Exception:
        return {"answer": ""}


def grade_answer(question: str, correct: str, predicted: str, oai: OpenAI) -> bool:
    """Binary correctness check: does predicted value match the correct answer?"""
    if not predicted or predicted.lower() in ("", "unspecified", "unknown"):
        return False
    resp = oai.chat.completions.create(
        model      = "gpt-4o-mini",
        max_completion_tokens = 4,
        messages   = [
            {"role": "system", "content": GRADER_SYSTEM},
            {"role": "user",   "content": GRADER_TEMPLATE.format(
                question=question, correct_answer=correct, model_answer=predicted,
            )},
        ],
    )
    return resp.choices[0].message.content.strip().lower().startswith("yes")


def _one_verifier_trial(question: str, correct_answer: str,
                         intended_interpretation: dict,
                         oai: OpenAI, model: str) -> bool:
    """
    Single verifier trial. Returns True (rejection vote) ONLY IF:
      (a) the model's answer matches correct_answer, AND
      (b) the model's assumed interpretation aligns with the intended interpretation.
    This prevents false rejections from lucky guesses that happen to produce the
    right number under a different (e.g. default) interpretation.
    """
    trial = verifier_answer(question, oai, model)
    ans   = str(trial.get("answer", "")).strip()
    if not grade_answer(question, correct_answer, ans, oai):
        return False
    return _interp_overlap(trial, intended_interpretation)


def run_adversarial_verifier(question: str, correct_answer: str,
                              intended_interpretation: dict,
                              oai: OpenAI) -> tuple[int, int]:
    """
    Run VERIFIER_ROUNDS trials on each of two OpenAI models WITHOUT context C.
    All 10 trials run concurrently. Rejection vote requires answer correct AND
    assumed interpretation aligns with intended_interpretation.
    Returns (n_votes, n_total).  Instance rejected if n_votes >= REJECT_THRESHOLD.
    """
    n_total = VERIFIER_ROUNDS * 2
    tasks   = (
        [(question, correct_answer, intended_interpretation, oai, VERIFIER_MODEL_A)] * VERIFIER_ROUNDS
        + [(question, correct_answer, intended_interpretation, oai, VERIFIER_MODEL_B)] * VERIFIER_ROUNDS
    )
    n_votes = 0
    with ThreadPoolExecutor(max_workers=n_total) as pool:
        futures = [pool.submit(_one_verifier_trial, *t) for t in tasks]
        for fut in as_completed(futures):
            try:
                if fut.result():
                    n_votes += 1
            except Exception:
                pass
    return n_votes, n_total


# ---------------------------------------------------------------------------
# Main construction loop
# ---------------------------------------------------------------------------
def construct_one(passage: dict, instance_id: int,
                  oai: OpenAI, dry_run: bool = False,
                  constructor_client: OpenAI | None = None,
                  constructor_model: str | None = None) -> dict | None:
    # constructor may use a different model/provider than the (fixed) verifier `oai`.
    # This isolates the constructor as the experimental variable in the ablation study.
    constructor_client = constructor_client or oai
    axes = passage.get("candidate_axes", [])
    h0   = compute_h0(axes)

    # Skip passages without a verified ground-truth answer — constructor would hallucinate
    if not passage.get("candidate_answer"):
        return None

    if dry_run:
        print(f"  [dry-run] would construct from passage {passage.get('passage_id')} "
              f"(axes={axes}, h0={h0:.2f})")
        return None

    # Step 1: constructor (variable model/provider; verifier stays on `oai`)
    constructed = call_constructor(passage, constructor_client, model=constructor_model)
    if not constructed:
        return None

    # Constructor may signal no viable ambiguity for the requested primary axis
    if "error" in constructed:
        print(f"  [constructor] {constructed['error']}")
        return None

    question        = constructed.get("question", "")
    context         = constructed.get("context", "")
    answer          = constructed.get("answer", "") or passage.get("candidate_answer", "")
    default_answer  = str(constructed.get("default_answer", "")).strip()
    intended_span   = constructed.get("intended_evidence_span", "")
    default_span    = constructed.get("default_evidence_span", "")

    if not question or not answer:
        print(f"  [constructor] missing question or answer in output")
        return None

    # Require both answers and both evidence spans — enforces reviewer's "dual-answer evidence"
    if not default_answer or default_answer == str(answer).strip():
        print(f"  [sanity] missing or identical default_answer: {default_answer!r}")
        return None
    if not intended_span or not default_span:
        print(f"  [sanity] missing evidence spans")
        return None

    # Structural sanity checks — catches type mismatches and missing entity before
    # spending verifier budget
    q_lower = question.lower()
    count_question = any(w in q_lower for w in ("how many", "number of", "count of"))
    if count_question and "%" in str(answer):
        print(f"  [sanity] Q-A type mismatch: count question but pct answer")
        return None
    # Implausibly large percentage answers (e.g. 3044%) indicate a wrong passage mapping
    if "%" in str(answer):
        try:
            pct_val = float(str(answer).replace("%", "").replace(",", "").strip())
            if abs(pct_val) > 500:
                print(f"  [sanity] implausible pct answer: {answer}")
                return None
        except ValueError:
            pass

    # Reject Q containing a 4-digit year or a month name — mirrors score_pilot r2 check.
    # Avoids wasting verifier budget on questions that will fail automated QC.
    if re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|"
        r"november|december|19\d{2}|20\d{2})\b", question.lower()
    ):
        print(f"  [sanity] inline date/year in Q: {question[:80]}")
        return None

    # Reject Q containing hard-coded disambiguating terms — mirrors score_pilot r2 check.
    _DISAMBIG_IN_Q = {
        # English
        "fiscal year", "fy20", "fy 20", "calendar year", "non-gaap", "adjusted",
        "consolidated", "segment", "amended", "restated", "original filing",
        "as reported", "organic", "constant currency", "ebitda", "non gaap",
        # Chinese equivalents
        "财年", "自然年", "非公认会计准则", "调整后", "合并报表", "合并口径",
        "分部", "子公司", "更正公告", "追溯重述", "扣非", "有机增长", "同口径",
    }
    q_lower_full = question.lower()
    if any(t in q_lower_full for t in _DISAMBIG_IN_Q):
        print(f"  [sanity] disambig term in Q: {question[:80]}")
        return None

    # EPS / per-share metrics should never have a % answer — unless the Q asks about a
    # RATE/GROWTH/CHANGE, which legitimately produces a % answer.
    _ABS_VALUE_METRICS = ("earnings per share", "per share", "eps", "stock price",
                          "share price", "book value per share", "diluted eps", "basic eps")
    _RATE_WORDS = ("growth", "rate", "margin", "yield", "ratio", "return", "change",
                   "increase", "decrease", "appreciation", "performance", "improvement")
    if ("%" in str(answer)
            and any(m in q_lower_full for m in _ABS_VALUE_METRICS)
            and not any(r in q_lower_full for r in _RATE_WORDS)):
        print(f"  [sanity] per-share metric with % answer: {question[:60]}  A={answer}")
        return None

    # Revenue/income/cash flow questions (without rate/growth/margin/yield words) should
    # have absolute answers, not percentages
    _ABS_METRICS = ("revenue", "net income", "earnings", "cash flow", "operating income",
                    "net sales", "total sales", "profit", "loss")
    if ("%" in str(answer)
            and any(m in q_lower_full for m in _ABS_METRICS)
            and not any(r in q_lower_full for r in _RATE_WORDS)):
        print(f"  [sanity] absolute metric with % answer: {question[:60]}  A={answer}")
        return None

    # Resolve final axes: constructor-reported axes_exercised filtered to valid taxonomy names,
    # falling back to the passage heuristic axes if the field is missing or all invalid
    _VALID_AXES = {"temporal_scope", "metric_definition", "entity_scope",
                   "filing_vintage", "recognition_policy"}
    _raw_axes   = constructed.get("axes_exercised") or axes
    _final_axes = [a for a in _raw_axes if a in _VALID_AXES] or axes

    # R13: evidence spans must be meaningfully distinct (not identical or near-identical).
    # Guards against the constructor quoting the same broad passage segment for both
    # interpretations, which would make the default/intended distinction unverifiable.
    if intended_span and default_span:
        # Strip whitespace and lowercased compare
        s1 = " ".join(intended_span.lower().split())
        s2 = " ".join(default_span.lower().split())
        if s1 == s2:
            print(f"  [sanity R13] evidence spans are identical")
            return None
        # Require spans differ by at least 30% of characters (rough distinctness)
        shorter = min(len(s1), len(s2))
        # Longest common prefix length as a cheap overlap proxy
        common_prefix = 0
        for c1, c2 in zip(s1, s2):
            if c1 == c2:
                common_prefix += 1
            else:
                break
        if shorter > 20 and common_prefix / shorter > 0.85:
            print(f"  [sanity R13] evidence spans too similar (>{85}% prefix overlap)")
            return None

    # R14: intended and default answers must differ by MORE than the grader's tolerance.
    # The grader treats numeric answers within ~1% (or ~1 percentage point for rate answers)
    # as equivalent. If the default answer falls inside that band, a model resolving to the
    # WRONG interpretation is still graded correct — the instance cannot discriminate
    # ambiguity-aware from ambiguity-blind models, so it is rejected.
    _gap = _answer_gap(answer, default_answer)
    if _gap is not None:
        is_pct = "%" in str(answer) and "%" in str(default_answer)
        if is_pct:
            # percentage-point gap must exceed 1.5 points (grader tol. is ~1 point)
            if _gap < 1.5:
                print(f"  [sanity R14] answers within grader tolerance "
                      f"({_gap:.2f}pt): A={answer} default={default_answer}")
                return None
        else:
            # relative gap on absolute values must exceed 3% (grader tol. is ~1%)
            try:
                a_val = _parse_numeric(answer)
                rel = _gap / abs(a_val) if a_val else None
                if rel is not None and rel < 0.03:
                    print(f"  [sanity R14] answers within grader tolerance "
                          f"({rel:.1%} rel): A={answer} default={default_answer}")
                    return None
            except (TypeError, ZeroDivisionError):
                pass

    # Require that the primary (most-needed) axis was actually exercised — saves verifier
    # budget on instances that will be rejected for axis distribution purposes anyway
    _primary_required = axes[0] if axes else None
    if _primary_required and _primary_required not in _final_axes:
        print(f"  [sanity] primary axis {_primary_required!r} not in axes_exercised={_final_axes}")
        return None

    # Step 2: adversarial verifier (pass intended_interpretation so rejection
    # only counts when verifier BOTH answers correctly AND uses the intended
    # interpretation, filtering out lucky guesses under a different reading)
    intended_interp = constructed.get("intended_interpretation", {})
    n_votes, n_total = run_adversarial_verifier(question, answer, intended_interp, oai)
    reject_rate = n_votes / n_total
    accepted    = n_votes < REJECT_THRESHOLD

    instance = {
        "instance_id":  f"fininteract_{instance_id:04d}",
        "passage_id":   passage.get("passage_id", ""),
        "language":     passage.get("language", "en"),
        "source":       passage.get("source", ""),
        "ticker":       passage.get("ticker", ""),
        "company":      passage.get("company", ""),
        "filing_type":  passage.get("filing_type", ""),
        "filing_date":  passage.get("filing_date", ""),
        "question":               question,
        "context":                context,
        "answer":                 answer,
        "default_answer":         default_answer,
        "intended_evidence_span": intended_span,
        "default_evidence_span":  default_span,
        "intended_interpretation": constructed.get("intended_interpretation", {}),
        "default_interpretation":  constructed.get("default_interpretation", {}),
        "shared_attributes":       constructed.get("shared_attributes", []),
        "distinctive_attributes":  constructed.get("distinctive_attributes", []),
        "axes":   _final_axes,
        "n_axes": len(_final_axes),
        "h0":     round(compute_h0(_final_axes), 4),
        "qc": {
            "n_verifier_votes": n_votes,   # trials where answer correct + interp matched
            "n_verifier_total": n_total,
            "verifier_vote_rate": round(reject_rate, 3),
            "accepted": accepted,
        },
    }
    return instance


ALL_AXES = [
    "temporal_scope", "metric_definition", "entity_scope",
    "filing_vintage", "recognition_policy",
]

# Per-axis target share in the accepted pool (for diversity enforcement).
# entity_scope is in 71% of pool passages (DocFinQA mentions segments everywhere),
# filing_vintage in 82% — both would saturate naturally. Metric_def/recognition_policy
# are rarer (24%/19%) so need higher targets to counterbalance the priority boost.
AXIS_TARGET_SHARE = {
    "temporal_scope":      0.25,   # target 25% of primary-axis slots
    "metric_definition":   0.25,   # rare in pool (24%) — boost needed; sum with others = 1.00
    "entity_scope":        0.15,   # over-represented in pool (71%) — cap low
    "filing_vintage":      0.15,   # over-represented in pool (82%) — cap low
    "recognition_policy":  0.20,   # total: 0.25+0.25+0.15+0.15+0.20 = 1.00
}


def _passage_priority(passage: dict, axis_counts: dict[str, int],
                      n_accepted: int) -> float:
    """
    Return a priority score for a passage given current axis counts.
    Higher = more desirable to process next.
    Under-represented axes get a boost; over-represented ones get a real penalty.
    Passages where ALL axes are more than 2x their target are excluded (score=0).
    """
    axes = passage.get("candidate_axes", [])
    if not axes:
        return 0.5

    score = 0.0
    any_under_quota = False
    all_hard_over_cap = True

    for ax in axes:
        current_share = axis_counts.get(ax, 0) / max(n_accepted, 1)
        target = AXIS_TARGET_SHARE.get(ax, 0.2)
        gap = target - current_share

        if current_share < target * 2.0:
            all_hard_over_cap = False

        if gap > 0:
            score += gap / target
            any_under_quota = True
        else:
            # Penalize over-quota axes at 3x the boost rate so they actually lose priority
            score += gap / target * 3.0

    # Hard exclusion: if every axis this passage covers is already at 2x its target,
    # skip it — it can only pile onto already-saturated axes
    if all_hard_over_cap:
        return 0.0

    return max(0.0, score / len(axes))


LANG_TARGET_SHARE = {"en": 0.80, "zh": 0.20}


def main(source: Path, out: Path, target: int, limit: int | None,
         dry_run: bool, skip_rejected: bool, resume: bool = False):
    oai = get_client()
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- Resume: read already-completed work ---
    done_passage_ids: set[str] = set()
    n_accepted = n_rejected = n_failed = 0
    instance_id = 1
    axis_counts: dict[str, int] = {ax: 0 for ax in ALL_AXES}
    lang_counts: dict[str, int] = {}

    if resume and out.exists():
        existing: list[dict] = []
        with out.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        for inst in existing:
            done_passage_ids.add(inst.get("passage_id", ""))
            for ax in inst.get("axes", []):
                if ax in axis_counts:
                    axis_counts[ax] += 1
            lang = inst.get("language", "en")
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        n_accepted = len(existing)
        instance_id = n_accepted + 1
        # Also skip previously rejected passages — no point re-spending verifier budget
        rej_path = out.with_suffix(".rejected.jsonl")
        n_rej_skipped = 0
        if rej_path.exists():
            with rej_path.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rej = json.loads(line)
                            pid = rej.get("passage_id", "")
                            if pid and pid not in done_passage_ids:
                                done_passage_ids.add(pid)
                                n_rej_skipped += 1
                        except json.JSONDecodeError:
                            pass
        print(f"[resume] Loaded {n_accepted} existing instances, "
              f"skipping {len(done_passage_ids)} passage IDs "
              f"({n_rej_skipped} from rejections). "
              f"Axis counts: {axis_counts}  Lang counts: {lang_counts}")
    elif out.exists() and not resume:
        existing_lines = sum(1 for l in out.open() if l.strip())
        if existing_lines > 0:
            print(f"WARNING: {out} already has {existing_lines} instances. "
                  f"Add --resume to continue, or delete the file to start fresh.")

    passages = []
    with source.open() as f:
        for line in f:
            line = line.strip()
            if line:
                passages.append(json.loads(line))

    # Filter already-done passages when resuming
    if done_passage_ids:
        passages = [p for p in passages if p.get("passage_id") not in done_passage_ids]
        print(f"[resume] {len(passages)} passages remaining after filtering done ones.")

    random.shuffle(passages)   # break alphabetical/source bias before sampling

    # Language balance cap: each language is capped at round(target * share), but
    # never below how many we already accepted (we don't discard existing work).
    lang_caps: dict[str, int] = {
        lang: max(round(target * share), lang_counts.get(lang, 0))
        for lang, share in LANG_TARGET_SHARE.items()
    }
    passages_before_lang_filter = len(passages)
    passages = [p for p in passages
                if lang_counts.get(p.get("language", "en"), 0) < lang_caps.get(p.get("language", "en"), target)]
    print(f"[lang-cap] Caps: {lang_caps}  Current: {lang_counts}  "
          f"Passages after filter: {len(passages)}/{passages_before_lang_filter}")

    if limit:
        passages = passages[:limit]

    # Initial sort: bring passages with rare, high-need axes to the front BEFORE
    # the dynamic diversity loop starts. Rarity weights = target_share / pool_frequency
    # so axes that are both rare in the pool AND needed get the highest score.
    # These are computed from the measured pool stats (entity_scope 71%, filing_vintage 82%,
    # metric_definition 24%, recognition_policy 19%, temporal_scope 49%).
    _INITIAL_AXIS_RARITY = {
        "metric_definition":  1.25,  # 0.30 / 0.24
        "recognition_policy": 1.05,  # 0.20 / 0.19
        "temporal_scope":     0.61,  # 0.30 / 0.49
        "entity_scope":       0.21,  # 0.15 / 0.71
        "filing_vintage":     0.18,  # 0.15 / 0.82
    }
    passages.sort(
        key=lambda p: sum(_INITIAL_AXIS_RARITY.get(ax, 0.5)
                          for ax in p.get("candidate_axes", []))
                      + random.random() * 0.05,  # jitter to break ties randomly
        reverse=True,
    )

    # Sort passages by diversity priority (re-sort every 10 passages processed
    # to avoid O(n²) overhead while still adapting dynamically)
    RESORT_EVERY = 10
    passages_processed = 0

    write_mode = "a" if resume else "w"
    with out.open(write_mode, encoding="utf-8") as f_out:
        remaining = list(passages)
        while remaining and not (target and n_accepted >= target):
            # Re-rank by under-represented axes periodically
            if passages_processed % RESORT_EVERY == 0 and n_accepted > 0:
                remaining.sort(
                    key=lambda p: _passage_priority(p, axis_counts, n_accepted),
                    reverse=True,
                )

            passage = remaining.pop(0)
            passages_processed += 1

            # NOTE: intra-passage axis sort removed — it caused multi-axis passages to be
            # tried for a promoted secondary axis (e.g. temporal_scope on a segment passage)
            # and then discarded when that promoted axis failed construction. The inter-passage
            # priority sort (above) is sufficient for axis diversity enforcement.

            # Fast pre-scan: skip DocFinQA passages that lack axis keyword signals
            if not _passes_axis_prescan(passage):
                n_failed += 1
                continue

            iterator_label = f"pass={passages_processed} acc={n_accepted}"
            if tqdm is None:
                print(f"  [{iterator_label}] {passage.get('passage_id', '')} "
                      f"primary={passage.get('candidate_axes', ['?'])[0]}")

            instance = construct_one(passage, instance_id, oai, dry_run)
            if instance is None:
                n_failed += 1
                continue

            if instance["qc"]["accepted"]:
                n_accepted += 1
                instance_id += 1
                f_out.write(json.dumps(instance, ensure_ascii=False) + "\n")
                f_out.flush()
                for ax in instance.get("axes", []):
                    if ax in axis_counts:
                        axis_counts[ax] += 1
                lang = instance.get("language", "en")
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
            else:
                n_rejected += 1
                if not skip_rejected:
                    rej_path = out.with_suffix(".rejected.jsonl")
                    with rej_path.open("a", encoding="utf-8") as f_rej:
                        f_rej.write(json.dumps(instance, ensure_ascii=False) + "\n")

    print(f"\nDone. accepted={n_accepted}  rejected={n_rejected}  failed={n_failed}")
    print(f"Verifier rejection rate: {n_rejected/(n_accepted+n_rejected+1e-9):.1%}")
    print(f"Axis distribution: { {ax: axis_counts[ax] for ax in ALL_AXES} }")
    print(f"Language distribution: {lang_counts}")
    print(f"Output: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source",  type=Path,
                   default=Path("data/sources/passages.jsonl"),
                   help="Normalized passage pool (one JSON per line)")
    p.add_argument("--out",     type=Path,
                   default=Path("data/constructed/instances.jsonl"))
    p.add_argument("--target",  type=int, default=500,
                   help="Stop after this many accepted instances")
    p.add_argument("--limit",   type=int, default=None,
                   help="Cap passages read (useful for dry-runs)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would happen without calling APIs")
    p.add_argument("--skip-rejected", action="store_true",
                   help="Don't write rejected instances to .rejected.jsonl")
    p.add_argument("--resume", action="store_true",
                   help="Resume from existing output file — reads done passage IDs, "
                        "appends new instances, skips already-processed passages")
    args = p.parse_args()
    main(args.source, args.out, args.target, args.limit,
         args.dry_run, args.skip_rejected, args.resume)
