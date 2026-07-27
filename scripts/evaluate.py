"""
FinInteract model evaluation.

Evaluates agents under three modes:
  1. answer-only      : no retrieval, no interaction (parametric recall baseline)
  2. answer+search    : ReAct with search action only; passage returned as oracle retrieval
  3. answer+search+interact : full ReAct with search + interact actions

Agent actions (emitted as JSON in the model's response):
  {"action": "search",   "query": "..."}
  {"action": "interact", "question": "yes/no question"}
  {"action": "answer",   "response": "..."}

User simulator (for interact action):
  GPT-5 at temperature=1.0, given disambiguating context C.
  Responds: "Yes", "No", or "I don't know."

Grader:
  gpt-4o-mini binary (yes/no), with financial-domain tolerance:
    ±1% numeric, entity/ticker equivalence, currency normalization.
    Fiscal-year mismatch is treated as wrong.

Models supported via OpenAI-compatible API:
  OpenAI   : gpt-5, gpt-5-mini, gpt-4o
  DeepSeek : deepseek-chat (pass DEEPSEEK_API_KEY + --deepseek-base-url)
  Qwen     : qwen-plus (pass DASHSCOPE_API_KEY + --qwen-base-url)
  GLM      : glm-4 (pass ZHIPU_API_KEY + --zhipu-base-url)
  Note: Claude evaluation requires Anthropic API (not implemented here).

Usage
-----
    export OPENAI_API_KEY=...

    # Evaluate GPT-5 on all three modes, 50 instances
    python scripts/evaluate.py \\
        --instances data/constructed/instances.jsonl \\
        --models gpt-5 \\
        --modes answer-only answer+search answer+search+interact \\
        --limit 50 \\
        --out data/results/eval_gpt5.jsonl

    # Evaluate multiple models
    python scripts/evaluate.py \\
        --instances data/constructed/instances.jsonl \\
        --models gpt-5 gpt-5-mini gpt-4o \\
        --modes answer+search+interact \\
        --out data/results/eval_all.jsonl

    # Forced-interaction ablation (require N asks before answering)
    python scripts/evaluate.py \\
        --instances data/constructed/instances.jsonl \\
        --models gpt-5 \\
        --modes answer+search+interact \\
        --forced-interact 4 \\
        --out data/results/eval_forced4.jsonl
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    sys.exit("pip install openai")

sys.path.insert(0, str(Path(__file__).parent))
from dise import compute_dise, aggregate_dise, DEFAULT_AXIS_OPTIONS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
USER_SIM_MODEL  = "gpt-5"    # temperature=1.0 — mirrors real user uncertainty
GRADER_MODEL    = "gpt-4o-mini"
AGENT_EXTRA_BODY: dict | None = None  # set from --agent-thinking; applied to AGENT calls only
JUDGE_CLIENT: "OpenAI | None" = None  # if set (OpenRouter), simulator+grader+axis-judge route here
TOKEN_USAGE: dict = {}         # model -> {"prompt":int,"completion":int,"calls":int}; for cost accounting


class CreditExhausted(Exception):
    """Raised when a provider rejects a call for lack of credit/quota, so the run
    halts cleanly instead of writing empty (all-wrong) rows that resume would skip."""
MAX_TURNS       = 12          # hard cap on ReAct loop turns
MAX_INTERACT    = 6           # max interact actions per instance
TEMPERATURE_SIM = 1.0

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
AGENT_SYSTEM_ANSWER_ONLY = """\
You are a financial analyst answering questions about company filings.
Given a financial question, output your final answer directly.

Output format — reply with ONLY valid JSON:
{"action": "answer", "response": "<your answer>"}
"""

AGENT_SYSTEM_SEARCH = """\
You are a financial analyst agent. You have access to a search tool.
Given a financial question, you may search for relevant filing information before answering.

Available actions (emit ONLY valid JSON, one action per turn):
  {"action": "search", "query": "<search query>"}
  {"action": "answer", "response": "<final answer>"}

Strategy:
- If the question is clear, answer directly.
- If you need supporting data, search first.
- After receiving search results, answer with the specific value requested.
"""

AGENT_SYSTEM_INTERACT = """\
You are a financial analyst agent. You have access to search and a user interaction tool.
Given a financial question, you may search for data AND ask the user clarifying yes/no questions.

Available actions (emit ONLY valid JSON, one action per turn):
  {"action": "search",   "query": "<search query>"}
  {"action": "interact", "question": "<a yes/no question to the user>"}
  {"action": "answer",   "response": "<final answer>"}

Strategy:
- Recognize when a question is ambiguous (e.g., unclear time period, metric definition, entity scope).
- Use interact to resolve ambiguity BEFORE answering — ask a precise yes/no question targeting one axis.
- Do NOT ask multiple questions at once; one interact action per turn.
- Interact questions MUST be answerable with yes, no, or I don't know.
- Once ambiguity is resolved, answer with the specific value.
"""

AGENT_SYSTEM_FORCED = """\
You are a financial analyst agent. You have access to search and a user interaction tool.
REQUIREMENT: You MUST ask at least {n_forced} clarifying questions via interact before you may answer.

Available actions (emit ONLY valid JSON, one action per turn):
  {"action": "search",   "query": "<search query>"}
  {"action": "interact", "question": "<a yes/no question to the user>"}
  {"action": "answer",   "response": "<final answer — only allowed after {n_forced} interact actions>"}

Ask targeted yes/no questions that resolve genuine ambiguity (time period, metric definition, entity scope, etc.).
"""

AGENT_SYSTEM_ALWAYS_ASK = """\
You are a financial analyst agent. You have access to search and a user interaction tool.
REQUIREMENT: You MUST ask exactly one clarifying question before answering, regardless of whether you think it is necessary.

Available actions (emit ONLY valid JSON, one action per turn):
  {"action": "search",   "query": "<search query>"}
  {"action": "interact", "question": "<a yes/no question to the user>"}
  {"action": "answer",   "response": "<final answer — only after one interact action>"}

First ask a yes/no clarifying question, then answer.
"""

AGENT_SYSTEM_AXIS_ORACLE = """\
You are a financial analyst agent. You have access to search and a user interaction tool.
The question is known to be ambiguous on the following dimension: {axis}.

Use the interact action to ask a targeted yes/no question about this dimension before answering.

Available actions (emit ONLY valid JSON, one action per turn):
  {"action": "search",   "query": "<search query>"}
  {"action": "interact", "question": "<a yes/no question about {axis}>"}
  {"action": "answer",   "response": "<final answer>"}
"""

AGENT_SYSTEM_ENUMERATE = """\
You are a financial analyst agent. You have access to search.
For ambiguous questions, enumerate the most plausible interpretations and provide an answer for each.

Available actions (emit ONLY valid JSON, one action per turn):
  {"action": "search",   "query": "<search query>"}
  {"action": "answer",   "response": "<your answer — list interpretations if ambiguous>"}

If the question is ambiguous, structure your answer as:
  "Under interpretation A (e.g. GAAP): <value>. Under interpretation B (e.g. non-GAAP): <value>."
"""

# ---------------------------------------------------------------------------
# Axis-Aware Clarification ReAct
# ---------------------------------------------------------------------------
# Implements the three-component method baseline from the paper:
#   Component 1 — Ambiguity pre-check: detect which fields are underspecified
#   Component 2 — Axis-conditioned clarification: generate targeted yes/no questions
#   Component 3 — Interpretation-state-gated answering: only answer when state resolved
# ---------------------------------------------------------------------------

INTERPRETATION_STATE_SCHEMA = """\
{
  "entity":   "<company / consolidation level, or 'unknown'>",
  "period":   "<fiscal year / quarter / TTM, or 'unknown'>",
  "metric":   "<GAAP / non-GAAP / organic / segment metric name, or 'unknown'>",
  "basis":    "<reported / adjusted / constant-currency, or 'unknown'>",
  "vintage":  "<original / amended / restated, or 'unknown'>"
}"""

AGENT_SYSTEM_AXIS_AWARE = """\
You are an expert financial analyst agent with access to search and user interaction.

## Four-step Axis-Aware Clarification Protocol

**Step 1 — Ambiguity Pre-check.**
Before doing anything else, decide which of the five financial ambiguity axes are underspecified
in the question. Mark each field in your interpretation state as "unknown" if ambiguous.

Five axes:
  temporal_scope    — is the reporting period (fiscal vs calendar year, quarter vs annual) clear?
  metric_definition — is the accounting basis (GAAP vs non-GAAP, organic, EBITDA variant) clear?
  entity_scope      — is the reporting scope (consolidated vs segment, parent vs subsidiary) clear?
  filing_vintage    — is the filing version (original vs amended/restated) clear?
  recognition_policy — is the accounting treatment (revenue timing, gross vs net) clear?

**Step 2 — Initialize Interpretation State.**
Emit a state action listing each field as known or "unknown":
  {"action": "state", "state": {"entity": "...", "period": "...", "metric": "...", "basis": "...", "vintage": "..."}}

**Step 3 — Axis-Conditioned Clarification.**
For each "unknown" field, ask ONE targeted yes/no question per turn.
Use the axis templates below — never ask a generic "can you clarify?":
  temporal_scope    → "Are you asking about the fiscal year results (not the calendar year)?"
  metric_definition → "Should I use the GAAP figure rather than the adjusted or non-GAAP figure?"
  entity_scope      → "Are you asking about the consolidated company-wide figure rather than a specific segment?"
  filing_vintage    → "Should I use the most recently filed version rather than an earlier or amended filing?"
  recognition_policy → "Should I use the accounting treatment as reported in the filing?"

**Step 4 — State-Gated Answering.**
You may ONLY emit an answer action once all required fields are no longer "unknown"
OR once the user has explicitly stated their intent. Do NOT answer while required fields
are still unresolved.

Available actions (emit ONLY valid JSON, one action per turn):
  {"action": "state",   "state": {<interpretation state>}}
  {"action": "search",  "query": "<search query>"}
  {"action": "interact","question": "<targeted yes/no question about one specific axis>"}
  {"action": "answer",  "response": "<final answer — only when state is resolved>"}
"""

AGENT_SYSTEM_GENERIC_STRUCTURED = """\
You are an expert financial analyst agent with access to search and user interaction.

## Structured Clarification Protocol

**Step 1 — Ambiguity Pre-check.**
Before doing anything else, identify every assumption the question leaves unspecified (for example
which entity, which period, which metric definition, or which accounting treatment is meant). List
what you would need to know to answer a single, unambiguous value.

**Step 2 — Initialize Interpretation State.**
Emit a state action listing each needed field as known or "unknown":
  {"action": "state", "state": {"<field name>": "known value or unknown", ...}}

**Step 3 — Targeted Clarification.**
For each "unknown" field, ask ONE targeted yes/no question per turn, the one that most reduces the
ambiguity. Ask about the specific missing detail. Never ask a generic "can you clarify?".

**Step 4 — State-Gated Answering.**
You may ONLY emit an answer action once no required field is still "unknown" OR the user has stated
their intent. Do NOT answer while required fields are unresolved.

Available actions (emit ONLY valid JSON, one action per turn):
  {"action": "state",   "state": {<interpretation state>}}
  {"action": "search",  "query": "<search query>"}
  {"action": "interact","question": "<targeted yes/no question about one missing detail>"}
  {"action": "answer",  "response": "<final answer — only when state is resolved>"}
"""

USER_SIM_SYSTEM = """\
You are a user who has submitted a financial question. You know the specific interpretation you intended.
When asked a yes/no clarifying question, respond with:
  - "Yes" if the question matches your intended interpretation
  - "No" if it does not
  - "I don't know" if you are genuinely uncertain

Respond with ONLY one of: "Yes", "No", or "I don't know". No other text.
"""

GRADER_SYSTEM = """\
You are a financial answer grader. Given a question, the correct answer, and a model's answer,
output "Yes" if the model answered correctly, "No" otherwise.

Grading rules:
- Numeric tolerance: ±1% of the correct value is acceptable (e.g., $42.1B vs $42.0B is correct).
- Entity/ticker equivalence: "AAPL" == "Apple Inc." == "Apple".
- Currency normalization: "$42B" == "$42,000M" == "42 billion dollars".
- Fiscal-year mismatch: if the model cites a different fiscal year than correct, grade as No.
- For percentage answers: 73.9% and 73.9 are the same; 73.9% and 74% are within tolerance.
- Output ONLY "Yes" or "No".
"""

GRADER_TEMPLATE = """\
Question: {question}
Correct answer: {correct_answer}
Model answer: {model_answer}
"""

SEARCH_RESULT_TEMPLATE = """\
[Search results for: "{query}"]
{content}
[End of search results]
"""

INTERACT_TEMPLATE = """\
[User response to: "{question}"]
{response}
"""

AXIS_HIT_SYSTEM = """\
Identify EVERY ambiguity axis a clarifying question is trying to resolve. Judge by what the
question asks the user to CHOOSE, not by words merely mentioned in passing.

  temporal_scope     — the point is which time period (FY vs CY, quarter vs year, TTM, as-of date).
                       Do NOT pick this merely because a year/quarter is named; only if choosing
                       the period is a purpose of the question.
  metric_definition  — which metric / accounting basis (GAAP vs non-GAAP, net vs attributable-to-parent,
                       level vs growth rate, adjusted, organic)
  entity_scope       — which entity/segment/subsidiary/share-class/parent-vs-consolidated. A question
                       naming or confirming a specific company or segment targets this.
  filing_vintage     — which filing version (original vs amended, restated, preliminary)
  recognition_policy — recognition treatment (revenue timing, over-time vs point-in-time, gross vs net,
                       capitalization)
  generic            — vague clarification request with no specific axis ("Can you clarify?")

Output ALL applicable axis labels, comma-separated (most central first). Labels only, no explanation.
The true axes are NOT provided — classify the question on its own merits.
"""

AXIS_HIT_TEMPLATE = """\
Clarifying question: {question}
"""

ORACLE_SIM_SYSTEM = """\
You are classifying a yes/no clarifying question from a financial agent.

Given:
1. The clarifying question
2. The intended interpretation (what the user means)
3. The default interpretation (what a non-expert would assume)

Classify whether the question's target matches the intended interpretation.
Output ONLY valid JSON: {"axis": "<axis name>", "target_value": "<what the question asks about>", "matches_intended": true/false}
"""

ORACLE_SIM_TEMPLATE = """\
Clarifying question: {question}

Intended interpretation:
  period: {intended_period}
  entity: {intended_entity}
  metric: {intended_metric}
  basis:  {intended_basis}

Default interpretation:
  period: {default_period}
  entity: {default_entity}
  metric: {default_metric}
  basis:  {default_basis}
"""

# Template questions per axis — used by template-oracle baseline
TEMPLATE_QUESTIONS = {
    "temporal_scope":     "Are you asking about the fiscal year (not the calendar year)?",
    "metric_definition":  "Should I use the GAAP figure rather than the adjusted or non-GAAP figure?",
    "entity_scope":       "Are you asking about the consolidated company-wide figure rather than a specific segment or subsidiary?",
    "filing_vintage":     "Should I use the most recently filed version rather than an earlier or amended filing?",
    "recognition_policy": "Should I use the revenue recognition policy as reported in the filing rather than an alternative treatment?",
}


# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------
def judge_client() -> OpenAI:
    """Client for the simulator/grader/axis-judge. Routes through OpenRouter when
    JUDGE_CLIENT is set (e.g. when the OpenAI account is unavailable), else OpenAI."""
    return JUDGE_CLIENT if JUDGE_CLIENT is not None else OpenAI()


def make_client(model_name: str, extra_kwargs: dict) -> OpenAI:
    """Return an OpenAI client configured for the given model family."""
    base_url = extra_kwargs.get("base_url")
    api_key  = extra_kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def chat(client: OpenAI, model: str, messages: list[dict],
         temperature: float = 0.0, max_tokens: int = 512,
         extra_body: dict | None = None) -> str:
    # gpt-5 / o-series reasoning models reject temperature != 1 and use max_completion_tokens;
    # other / non-OpenAI models use temperature + max_tokens.
    ml = model.lower()
    core = ml.split("/")[-1]                 # strip any OpenRouter provider prefix (openai/, x-ai/, ...)
    looks_reasoning = core.startswith(("gpt-5", "o1", "o3", "o4")) or "gpt-5" in core
    is_openai_direct = "/" not in model
    # A small max_tokens cap gets fully consumed by internal reasoning -> empty content.
    # OpenAI-direct reasoning models want max_completion_tokens; OpenRouter (any model with
    # a provider "/" prefix) always wants max_tokens, so reasoning models routed through
    # OpenRouter (openai/gpt-5, x-ai/grok-4.5 with reasoning extra_body, ...) need the same
    # headroom applied to max_tokens. In both cases temperature is omitted (defaults to 1).
    is_router_reasoning = bool(extra_body) and "reasoning" in extra_body
    kwargs: dict = {"model": model, "messages": messages}
    if is_openai_direct and looks_reasoning:
        kwargs["max_completion_tokens"] = max_tokens + 4096   # temperature omitted (defaults to 1)
    elif (not is_openai_direct and looks_reasoning) or is_router_reasoning:
        kwargs["max_tokens"] = max_tokens + 4096              # temperature omitted
    else:
        kwargs["temperature"] = temperature
        kwargs["max_tokens"] = max_tokens
    if extra_body:                      # e.g. {"chat_template_kwargs": {"enable_thinking": False}}
        kwargs["extra_body"] = extra_body
    try:
        resp = client.chat.completions.create(**kwargs)
        u = getattr(resp, "usage", None)
        if u is not None:
            d = TOKEN_USAGE.setdefault(model, {"prompt": 0, "completion": 0, "calls": 0})
            d["prompt"]     += getattr(u, "prompt_tokens", 0) or 0
            d["completion"] += getattr(u, "completion_tokens", 0) or 0
            d["calls"]      += 1
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        msg = str(e).lower()
        # OpenRouter signals no funds with HTTP 402; OpenAI with 429 insufficient_quota.
        if ("402" in msg or "insufficient_quota" in msg
                or "insufficient credit" in msg or "insufficient_credits" in msg
                or "requires more credits" in msg or "negative credit" in msg):
            raise CreditExhausted(str(e))
        print(f"  [chat error] {model}: {e}", file=sys.stderr)
        return ""


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------
def grade(question: str, correct: str, predicted: str, oai: OpenAI) -> bool:
    if not predicted:
        return False
    verdict = chat(
        oai, GRADER_MODEL,
        messages=[
            {"role": "system", "content": GRADER_SYSTEM},
            {"role": "user",   "content": GRADER_TEMPLATE.format(
                question=question, correct_answer=correct, model_answer=predicted
            )},
        ],
        max_tokens=4,
    )
    return verdict.lower().startswith("yes")


# ---------------------------------------------------------------------------
# User simulators (oracle / llm / noisy)
# ---------------------------------------------------------------------------
def _oracle_user_answer(question: str, context: str,
                         intended: dict, default: dict,
                         oai: OpenAI | None = None) -> str:
    """
    Structured oracle simulator: uses GPT-4o-mini to classify the question's
    target axis and value, then deterministically compares against the intended
    interpretation. Handles semantic equivalences (e.g. 'non-GAAP' = 'adjusted').
    Falls back to keyword matching if the LLM call fails.
    """
    if oai is None:
        oai = judge_client()

    prompt = ORACLE_SIM_TEMPLATE.format(
        question=question,
        intended_period=intended.get("period", "unspecified"),
        intended_entity=intended.get("entity", "unspecified"),
        intended_metric=intended.get("metric", "unspecified"),
        intended_basis=intended.get("basis", "unspecified"),
        default_period=default.get("period", "unspecified"),
        default_entity=default.get("entity", "unspecified"),
        default_metric=default.get("metric", "unspecified"),
        default_basis=default.get("basis", "unspecified"),
    )
    try:
        resp = oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": ORACLE_SIM_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_completion_tokens=80,
        )
        raw = (resp.choices[0].message.content or "").strip()
        obj = json.loads(raw)
        matches = obj.get("matches_intended", None)
        if matches is True:
            return "Yes"
        if matches is False:
            return "No"
    except Exception:
        pass

    # Keyword-matching fallback
    q_lower = question.lower()
    for v in intended.values():
        if isinstance(v, str) and len(v) > 4 and v.lower() in q_lower:
            return "Yes"
    for v in default.values():
        if isinstance(v, str) and len(v) > 4 and v.lower() in q_lower:
            return "No"
    return "I don't know"


def _llm_user_answer(question: str, context: str, oai: OpenAI) -> str:
    """GPT-5 simulator at temperature=1.0 (noisy but knowledgeable)."""
    resp = chat(
        oai, USER_SIM_MODEL,
        messages=[
            {"role": "system", "content": USER_SIM_SYSTEM},
            {"role": "user",   "content": (
                f"My intended interpretation is described by: {context}\n\n"
                f"Clarifying question: {question}"
            )},
        ],
        temperature=TEMPERATURE_SIM,
        max_tokens=10,
    )
    resp_lower = resp.lower()
    if resp_lower.startswith("yes"):
        return "Yes"
    if resp_lower.startswith("no"):
        return "No"
    return "I don't know"


def _noisy_user_answer(question: str, context: str, oai: OpenAI,
                        noise_rate: float = 0.15) -> str:
    """LLM simulator with 15% random answer corruption."""
    import random
    base = _llm_user_answer(question, context, oai)
    if random.random() < noise_rate:
        others = [r for r in ("Yes", "No", "I don't know") if r != base]
        return random.choice(others)
    return base


def simulate_user(question: str, context: str, oai: OpenAI,
                  mode: str = "llm",
                  intended: dict | None = None,
                  default: dict | None = None) -> str:
    """Dispatch to the requested simulator type."""
    if mode == "oracle":
        return _oracle_user_answer(
            question, context,
            intended or {}, default or {}, oai
        )
    if mode == "noisy":
        return _noisy_user_answer(question, context, oai)
    if mode == "freeform":
        return _freeform_user_answer(question, context, oai)
    return _llm_user_answer(question, context, oai)


FREEFORM_SIM_SYSTEM = """\
You are the user who submitted a financial question. Your intended interpretation is described below.
Answer the agent's clarifying question truthfully and concisely from that intended interpretation
(entity scope, period, metric, reporting basis). If the question is yes/no, answer Yes, No, or I don't know.
If it is open-ended (for example "which fiscal year?" or "which segment?"), give the specific detail.
Never state the final numeric answer, since the agent must compute that itself. Keep your answer under 15 words.
"""


def _freeform_user_answer(question: str, context: str, oai: OpenAI) -> str:
    """Free-form simulator: answers open-ended clarifying questions from the intended
    interpretation C without revealing the numeric answer. Robustness check for the reviewer
    concern that a yes/no-only simulator manufactures the elicitation bottleneck."""
    resp = chat(
        oai, USER_SIM_MODEL,
        messages=[
            {"role": "system", "content": FREEFORM_SIM_SYSTEM},
            {"role": "user",   "content": (
                f"My intended interpretation is described by: {context}\n\n"
                f"Agent's clarifying question: {question}\n\nYour answer:"
            )},
        ],
        temperature=TEMPERATURE_SIM,
        max_tokens=40,
    )
    return (resp or "I don't know").strip()


# ---------------------------------------------------------------------------
# AxisHit metric
# ---------------------------------------------------------------------------
ALL_AXIS_LABELS = {"temporal_scope", "metric_definition", "entity_scope",
                   "filing_vintage", "recognition_policy", "generic", "none"}


def classify_axis_hit(interact_question: str, true_axes: list[str],
                      oai: OpenAI) -> dict:
    """
    Classify a clarifying question.
    Returns dict with keys: axis_pred, is_hit, is_generic, is_wrong_axis.

    is_hit:        question targets one of the true ambiguity axes
    is_generic:    vague ask with no specific axis target
    is_wrong_axis: question targets a specific axis but not the correct one
    """
    resp = chat(
        oai, GRADER_MODEL,
        messages=[
            {"role": "system", "content": AXIS_HIT_SYSTEM},
            {"role": "user",   "content": AXIS_HIT_TEMPLATE.format(
                question=interact_question)},
        ],
        max_tokens=30,
    )
    # Multi-label parse: a clarifying question is often compound (e.g. names an entity AND a
    # period). Collect every axis the classifier emits; is_hit if any overlaps the true axes.
    # Validated against 5 human annotators (n=80): single-label had a temporal-scope bias and
    # agreed with humans only 0.53; this multi-label form agrees 0.75, matching human-human
    # agreement (0.73). The true axes are NOT shown to the classifier (no leakage).
    text = (resp or "").strip().lower().replace("-", "_")
    AXES_ONLY  = {"temporal_scope", "metric_definition", "entity_scope",
                  "filing_vintage", "recognition_policy"}
    pred_axes  = {ax for ax in AXES_ONLY if ax in text}
    is_generic = ("generic" in text) and not pred_axes

    true_set   = set(true_axes)
    is_hit     = bool(pred_axes & true_set)
    is_wrong   = bool(pred_axes) and not is_hit
    axis_pred  = ",".join(sorted(pred_axes)) if pred_axes else ("generic" if is_generic else "none")

    return {
        "question":      interact_question[:120],
        "axis_pred":     axis_pred,
        "is_hit":        is_hit,
        "is_generic":    is_generic,
        "is_wrong_axis": is_wrong,
    }


# ---------------------------------------------------------------------------
# Oracle search (returns passage as search result; simulates successful retrieval)
# ---------------------------------------------------------------------------
def oracle_search(query: str, passage_text: str) -> str:
    """Simulate a web search by returning the relevant filing passage."""
    return passage_text[:4000]   # trim to avoid token explosion


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------
def parse_action(text: str) -> dict | None:
    """Extract the JSON action from the agent's response."""
    # Try direct JSON parse first
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "action" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in text (model may add commentary)
    match = re.search(r'\{[^{}]*"action"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict) and "action" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# ReAct agent loop
# ---------------------------------------------------------------------------
def elicit_confidence(question: str, answer: str, agent_client: OpenAI,
                      model_name: str) -> int | None:
    """Ask the model how confident it is in its own answer (0-100).

    Post-hoc verbalized confidence (Cole et al. EMNLP 2023 style) — enables the
    calibration analysis without altering the per-mode prompts. Returns None on failure.
    """
    try:
        msg = [
            {"role": "system", "content":
             "You assess confidence in a financial answer. Output ONLY an integer 0-100: "
             "the probability (percent) that the answer is correct. No other text."},
            {"role": "user", "content":
             f"Question: {question}\nProposed answer: {answer}\n"
             f"Confidence (0-100) that this answer is correct:"},
        ]
        out = chat(agent_client, model_name, msg, temperature=0.0, max_tokens=8)
        m = re.search(r"\d{1,3}", out or "")
        if not m:
            return None
        return max(0, min(100, int(m.group())))
    except Exception:
        return None


def run_agent(instance: dict, mode: str, agent_client: OpenAI,
              sim_client: OpenAI, model_name: str,
              forced_n: int = 0,
              user_sim_mode: str = "llm",
              elicit_conf: bool = False) -> dict:
    """
    Run one instance through the ReAct loop.

    Args:
        user_sim_mode: "oracle" | "llm" | "noisy"  (see simulate_user)

    Returns a result dict with:
      correct, final_answer, n_turns, n_asks, n_searches, trajectory,
      axis_hits (per-interact classification), axis_hit_rate
    """
    question     = instance["question"]
    context      = instance["context"]
    passage_text = instance.get("passage_text", "")
    correct_ans  = instance["answer"]
    default_ans  = instance.get("default_answer", "")
    true_axes    = instance.get("axes", [])
    intended_int = instance.get("intended_interpretation", {})
    default_int  = instance.get("default_interpretation", {})
    axis_hits    = []   # init before any early-returning branch (e.g. template-oracle)

    # Template-oracle baseline: inject one fixed human-written question per primary axis,
    # then let the model answer. Bypasses the ReAct loop entirely.
    if mode == "template-oracle":
        primary_axis   = true_axes[0] if true_axes else "metric_definition"
        template_q     = TEMPLATE_QUESTIONS.get(primary_axis, "Can you clarify your intended interpretation?")
        sim_response   = simulate_user(template_q, context, sim_client,
                                       mode=user_sim_mode,
                                       intended=intended_int, default=default_int)
        axis_info = classify_axis_hit(template_q, true_axes, sim_client)
        axis_hits.append(axis_info)
        # Now ask the model to answer given the template clarification. Inject the oracle
        # passage so the model can GROUND the answer -- otherwise this path (which bypasses
        # the ReAct search loop) gives a perfect on-axis question but no evidence, conflating
        # question quality with retrieval. With evidence, template-oracle is the ceiling of
        # question quality (perfect on-axis clarification + grounding), comparable to axis-oracle.
        ev = passage_text[:4000]
        ans_resp = chat(agent_client, model_name, [
            {"role": "system", "content": AGENT_SYSTEM_ANSWER_ONLY},
            {"role": "user",   "content": (f"Relevant evidence: {ev}\n\n{question}" if ev else question)},
            {"role": "assistant", "content": json.dumps({"action": "interact", "question": template_q})},
            {"role": "user",   "content": f"[User: {sim_response}]"},
            {"role": "user",   "content": "Now provide your final answer."},
        ], max_tokens=256)
        action_obj   = parse_action(ans_resp)
        final_answer = (action_obj.get("response", ans_resp) if action_obj else ans_resp)
        correct = grade(question, correct_ans, final_answer, judge_client())
        return {
            "instance_id": instance.get("instance_id", instance.get("id")),
            "model": model_name, "mode": mode, "user_sim": user_sim_mode,
            "forced_n": 0, "correct": correct, "final_answer": final_answer,
            "correct_answer": correct_ans, "n_turns": 2, "n_asks": 1,
            "n_searches": 0, "interacted": True, "axis_hits": axis_hits,
            "axis_hit_rate": 1.0 if axis_hits[0]["is_hit"] else 0.0,
            "axes": true_axes, "n_axes": instance.get("n_axes", len(true_axes)),
            "h0": instance.get("h0", 0.0), "language": instance.get("language", "en"),
            "source": instance.get("source", ""), "trajectory": [],
        }

    # Choose system prompt
    if mode == "answer-only":
        system = AGENT_SYSTEM_ANSWER_ONLY
    elif mode == "answer+search":
        system = AGENT_SYSTEM_SEARCH
    elif mode == "enumerate":
        system = AGENT_SYSTEM_ENUMERATE
    elif mode == "always-ask":
        system = AGENT_SYSTEM_ALWAYS_ASK
    elif mode == "axis-oracle":
        primary_axis = true_axes[0] if true_axes else "metric_definition"
        # .replace (not .format): these templates contain literal JSON braces
        # ({"action": ...}) that .format would misread as replacement fields.
        system = AGENT_SYSTEM_AXIS_ORACLE.replace("{axis}", primary_axis)
    elif mode == "axis-aware":
        system = AGENT_SYSTEM_AXIS_AWARE
    elif mode == "generic-structured":
        system = AGENT_SYSTEM_GENERIC_STRUCTURED
    elif mode == "interp-oracle":
        # Decompose the +interact gap: hand the agent the RESOLVED interpretation C
        # (which reading is intended) but NOT the answer-bearing evidence spans and
        # no search. Drop from full context-oracle (C + spans) to here = the RECALL
        # cost; drop from here to +interact = the ELICITATION cost.
        system = AGENT_SYSTEM_ANSWER_ONLY
    elif mode == "context-oracle":
        # Full context-oracle: hand the agent BOTH the resolved interpretation C AND the
        # oracle evidence (gold passage). No elicitation, no retrieval gap -> the ceiling.
        # Forms a 2x2 with {answer-only, answer+search, interp-oracle}:
        #   C absent/present  x  evidence absent(answer-only)/oracle(passage).
        # Row delta (add evidence) = grounding value; column delta (add C) = elicitation value.
        system = AGENT_SYSTEM_ANSWER_ONLY
    elif forced_n > 0:
        system = AGENT_SYSTEM_FORCED.replace("{n_forced}", str(forced_n))
    else:
        system = AGENT_SYSTEM_INTERACT

    user_content = question
    if mode == "interp-oracle":
        user_content = (f"Intended interpretation: {context}\n\n"
                        f"Question: {question}")
    elif mode == "context-oracle":
        user_content = (f"Intended interpretation: {context}\n\n"
                        f"Relevant evidence: {passage_text[:4000]}\n\n"
                        f"Question: {question}")
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_content},
    ]

    trajectory   = []
    axis_hits    = []    # per-interact: {"question": ..., "axis_pred": ..., "hit": ...}
    n_asks       = 0
    n_searches   = 0
    final_answer = ""
    correct      = False

    for turn in range(MAX_TURNS):
        response = chat(agent_client, model_name, messages, temperature=0.0,
                        max_tokens=1024, extra_body=AGENT_EXTRA_BODY)
        if not response:
            break

        action_obj = parse_action(response)
        if action_obj is None:
            # Model produced unstructured text — treat as final answer
            final_answer = response
            trajectory.append({"turn": turn, "action": "answer_unstructured",
                                "content": response[:200]})
            break

        action = action_obj.get("action", "")
        trajectory.append({"turn": turn, "action": action,
                            "content": str(action_obj)[:200]})

        if action == "answer":
            final_answer = action_obj.get("response", "")
            break

        elif action == "search":
            if mode == "answer-only":
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user",
                                 "content": "Search is not available. Please answer directly."})
            else:
                query   = action_obj.get("query", question)
                content = oracle_search(query, passage_text)
                result_text = SEARCH_RESULT_TEMPLATE.format(
                    query=query, content=content)
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": result_text})
                n_searches += 1

        elif action == "interact":
            if mode in ("answer-only", "answer+search"):
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user",
                                 "content": "User interaction is not available. Please answer."})
            elif n_asks >= MAX_INTERACT:
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user",
                                 "content": "Maximum questions reached. Please answer now."})
            else:
                q = action_obj.get("question", "")

                # AxisHit classification
                axis_info = classify_axis_hit(q, true_axes, sim_client)
                axis_hits.append(axis_info)

                # User simulation
                sim = simulate_user(q, context, sim_client,
                                    mode=user_sim_mode,
                                    intended=intended_int,
                                    default=default_int)
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user",
                                 "content": INTERACT_TEMPLATE.format(
                                     question=q, response=sim)})
                n_asks += 1

        elif action == "state":
            # Axis-aware mode: model emits interpretation state — acknowledge and continue
            messages.append({"role": "assistant", "content": response})
            state_fields = action_obj.get("state", {})
            unknowns = [k for k, v in state_fields.items()
                        if str(v).lower() in ("unknown", "", "?")]
            if unknowns:
                messages.append({"role": "user",
                                 "content": f"Interpretation state noted. "
                                            f"Unresolved fields: {', '.join(unknowns)}. "
                                            f"Please ask a targeted clarification for the most critical field."})
            else:
                messages.append({"role": "user",
                                 "content": "Interpretation state is fully resolved. Please answer."})

        else:
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user",
                             "content": 'Unknown action. Use {"action": "answer", "response": "..."}'})

    grader_client = judge_client()  # OpenAI, or OpenRouter when JUDGE_CLIENT is set
    # Enforce forced-n: if model answered before asking forced_n times, mark wrong
    if forced_n > 0 and n_asks < forced_n:
        correct = False
    else:
        correct = grade(question, correct_ans, final_answer, grader_client)

    # Default-capture: when WRONG, did the model land on the default (naive) interpretation?
    # A high default-capture rate is direct evidence of ambiguity blindness rather than
    # random error. One extra grader call per wrong instance.
    default_captured = False
    if (not correct) and default_ans and final_answer:
        default_captured = grade(question, default_ans, final_answer, grader_client)

    confidence = None
    if elicit_conf and final_answer:
        confidence = elicit_confidence(question, final_answer, agent_client, model_name)

    n_hits    = sum(1 for h in axis_hits if h.get("is_hit"))
    hit_rate  = n_hits / len(axis_hits) if axis_hits else None
    first_action = trajectory[0]["action"] if trajectory else "none"

    return {
        "instance_id":    instance.get("instance_id", instance.get("id")),
        "model":          model_name,
        "mode":           mode,
        "user_sim":       user_sim_mode,
        "forced_n":       forced_n,
        "correct":        correct,
        "final_answer":   final_answer,
        "correct_answer": correct_ans,
        "default_answer": default_ans,
        "default_captured": default_captured,
        "confidence":     confidence,
        "first_action":   first_action,
        "n_turns":        len(trajectory),
        "n_asks":         n_asks,
        "n_searches":     n_searches,
        "interacted":     n_asks > 0,
        "axis_hits":      axis_hits,
        "axis_hit_rate":  hit_rate,
        "axes":           true_axes,
        "n_axes":         instance.get("n_axes", len(true_axes)),
        "h0":             instance.get("h0", 0.0),
        "language":       instance.get("language", "en"),
        "source":         instance.get("source", ""),
        "trajectory":     trajectory,
    }


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------
def compute_metrics(results: list[dict]) -> dict:
    """Compute accuracy, DisE+, IR, AxisHit, and round averages."""
    import statistics

    if not results:
        return {}

    n = len(results)
    accuracy    = sum(r["correct"] for r in results) / n
    ir_rate     = sum(r["interacted"] for r in results) / n
    avg_turns   = statistics.mean(r["n_turns"] for r in results)
    avg_asks    = statistics.mean(r["n_asks"] for r in results)

    # DisE per instance
    dise_results = []
    for r in results:
        dr = compute_dise(
            axes_hit = r["axes"],
            n_asks   = r["n_asks"],
            correct  = r["correct"],
        )
        dise_results.append(dr)
    dise_agg = aggregate_dise(dise_results)

    # Per-axis accuracy
    by_axis: dict[str, list[bool]] = {}
    for r in results:
        for ax in r.get("axes", []):
            by_axis.setdefault(ax, []).append(r["correct"])
    per_axis_acc = {ax: round(sum(vs)/len(vs), 4) for ax, vs in by_axis.items()}

    # Per-language accuracy
    by_lang: dict[str, list[bool]] = {}
    for r in results:
        lang = r.get("language", "en")
        by_lang.setdefault(lang, []).append(r["correct"])
    per_lang_acc = {lang: round(sum(vs)/len(vs), 4) for lang, vs in by_lang.items()}

    # AxisHit fine-grained aggregation
    all_interact_qs = []
    for r in results:
        all_interact_qs.extend(r.get("axis_hits", []))

    n_qs = len(all_interact_qs)
    axis_hit_rate  = round(sum(h["is_hit"]        for h in all_interact_qs) / n_qs, 4) if n_qs else None
    generic_rate   = round(sum(h["is_generic"]    for h in all_interact_qs) / n_qs, 4) if n_qs else None
    wrong_ax_rate  = round(sum(h["is_wrong_axis"] for h in all_interact_qs) / n_qs, 4) if n_qs else None

    # AxisHit@1: first clarification question is on-axis
    first_hits = []
    for r in results:
        hits = r.get("axis_hits", [])
        if hits:
            first_hits.append(hits[0]["is_hit"])
    axis_hit_at1 = round(sum(first_hits) / len(first_hits), 4) if first_hits else None

    # AnyAxisHit: at least one clarification is on-axis
    any_hits = []
    for r in results:
        hits = r.get("axis_hits", [])
        if hits:
            any_hits.append(any(h["is_hit"] for h in hits))
    any_axis_hit = round(sum(any_hits) / len(any_hits), 4) if any_hits else None

    # ECE (5-bin expected calibration error) from verbalized confidence, when present
    # (results produced with --elicit-confidence). None when no confidences recorded.
    conf_pairs = [(r["confidence"] / 100.0, bool(r["correct"]))
                  for r in results if r.get("confidence") is not None]
    ece = None
    if conf_pairs:
        n_conf = len(conf_pairs)
        bins: list[list] = [[] for _ in range(5)]     # [0,.2),[.2,.4),...,[.8,1]
        for c, ok in conf_pairs:
            bins[min(int(c * 5), 4)].append((c, ok))
        ece = 0.0
        for b in bins:
            if b:
                avg_conf = sum(c for c, _ in b) / len(b)
                acc_bin  = sum(ok for _, ok in b) / len(b)
                ece += (len(b) / n_conf) * abs(avg_conf - acc_bin)
        ece = round(ece, 4)

    return {
        "n":              n,
        "accuracy":       round(accuracy, 4),
        "ir_rate":        round(ir_rate, 4),
        "avg_turns":      round(avg_turns, 2),
        "avg_asks":       round(avg_asks, 2),
        "ece":            ece,               # None unless --elicit-confidence was used
        "n_confidence":   len(conf_pairs),
        # AxisHit breakdown
        "axis_hit_rate":    axis_hit_rate,   # fraction of all interact questions on-axis
        "axis_hit_at1":     axis_hit_at1,    # first question is on-axis
        "any_axis_hit":     any_axis_hit,    # at least one question on-axis
        "wrong_axis_rate":  wrong_ax_rate,   # question targets wrong specific axis
        "generic_ask_rate": generic_rate,    # vague clarification with no axis
        "per_axis_accuracy":    per_axis_acc,
        "per_lang_accuracy":    per_lang_acc,
        **dise_agg,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Evaluate FinInteract agents")
    p.add_argument("--instances",  required=True,
                   help="Path to constructed instances JSONL")
    p.add_argument("--models",     nargs="+", default=["gpt-5-mini"],
                   help="Model name(s) to evaluate")
    p.add_argument("--modes",      nargs="+",
                   default=["answer-only", "answer+search", "answer+search+interact"],
                   choices=["answer-only", "answer+search", "answer+search+interact",
                            "always-ask", "axis-oracle", "template-oracle", "enumerate",
                            "axis-aware", "generic-structured", "interp-oracle", "context-oracle"])
    p.add_argument("--limit",      type=int, default=None,
                   help="Max instances to evaluate (for pilot runs)")
    p.add_argument("--out",        default="data/results/eval_results.jsonl",
                   help="Output JSONL path for per-instance results")
    p.add_argument("--summary",    default="data/results/eval_summary.json",
                   help="Output JSON path for aggregate metrics")
    p.add_argument("--forced-interact", type=int, default=0, dest="forced_n",
                   help="Forced-interaction ablation: require N asks before answering")
    p.add_argument("--user-sim", default="llm",
                   choices=["oracle", "llm", "noisy", "freeform"],
                   help="User simulator: oracle (rule-based), llm (GPT-5 yes/no), noisy (GPT-5 + 15%% random), "
                        "freeform (GPT-5 answers open-ended questions from C without leaking the answer)")
    p.add_argument("--max-interact", type=int, default=None,
                   help="Override max interact actions per instance (default 6); caps "
                        "clarification rounds for faster ablation runs (applies to all conditions).")
    p.add_argument("--passage-file", default=None,
                   help="Path to passages.jsonl (for oracle retrieval lookup)")
    p.add_argument("--elicit-confidence", action="store_true",
                   help="Ask each model for verbalized confidence (0-100) after answering, "
                        "enabling the calibration analysis (one extra call per instance).")
    # Per-model API configuration
    p.add_argument("--deepseek-base-url", default="https://api.deepseek.com/v1")
    p.add_argument("--qwen-base-url",     default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    p.add_argument("--zhipu-base-url",    default="https://open.bigmodel.cn/api/paas/v4/")
    p.add_argument("--agent-base-url", default=os.environ.get("AGENT_BASE_URL"),
                   help="OpenAI-compatible endpoint for the AGENT under test "
                        "(e.g. a local vLLM server http://localhost:8000/v1). "
                        "The user-simulator and grader stay on the OpenAI API.")
    p.add_argument("--agent-api-key", default=os.environ.get("AGENT_API_KEY", "EMPTY"),
                   help="API key for --agent-base-url (vLLM ignores it; default EMPTY).")
    p.add_argument("--agent-thinking", choices=["on", "off"], default=None,
                   help="Toggle hybrid-reasoning models' thinking mode on the local agent "
                        "via chat_template_kwargs.enable_thinking (applies to agent calls only).")
    p.add_argument("--openrouter-config", default=None,
                   help="Path to a JSON file with {\"base_url\", \"api_key\"} for OpenRouter. "
                        "Routes the AGENT-under-test through OpenRouter (model IDs like "
                        "anthropic/claude-sonnet-5). By default the simulator+grader also route "
                        "through OpenRouter using the identical OpenAI models (openai/gpt-5, "
                        "openai/gpt-4o-mini) so the same key covers the whole run; pass "
                        "--judge-on-openai to keep them on the OpenAI API instead. Keeps the API "
                        "key out of the command line / process args.")
    p.add_argument("--agent-reasoning", choices=["on", "off"], default="on",
                   help="For OpenRouter agents, enable provider reasoning via "
                        "extra_body={\"reasoning\": {\"enabled\": True}} (default on).")
    p.add_argument("--judge-on-openai", action="store_true",
                   help="Keep the simulator+grader on the OpenAI API even when "
                        "--openrouter-config is set (requires OpenAI quota).")
    p.add_argument("--resume", action="store_true",
                   help="Append to --out and skip (model, mode, instance) tuples already "
                        "present, so a run stopped by exhausted credit can be continued after "
                        "a top-up without re-spending on completed work.")
    args = p.parse_args()
    # UTF-8 console: instances/answers contain CJK; the default Windows cp1252 stdout
    # raises UnicodeEncodeError on the ✓/✗ glyphs and Chinese text.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    global AGENT_EXTRA_BODY, JUDGE_CLIENT, USER_SIM_MODEL, GRADER_MODEL
    if args.openrouter_config:
        cfg = json.loads(Path(args.openrouter_config).read_text(encoding="utf-8"))
        args.agent_base_url = cfg["base_url"]
        args.agent_api_key  = cfg["api_key"]
        if args.agent_reasoning == "on":
            AGENT_EXTRA_BODY = {"reasoning": {"enabled": True}}
        if not args.judge_on_openai:
            # Route simulator+grader+axis-judge through OpenRouter using the *same* underlying
            # OpenAI models, so methodology matches existing rows and one key covers everything.
            JUDGE_CLIENT   = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
            USER_SIM_MODEL = "openai/gpt-5"
            GRADER_MODEL   = "openai/gpt-4o-mini"
        judge_where = "OpenAI API" if args.judge_on_openai else "OpenRouter (openai/gpt-5, openai/gpt-4o-mini)"
        print(f"Routing AGENT through OpenRouter ({args.agent_base_url}); "
              f"simulator+grader via {judge_where}. reasoning={args.agent_reasoning}")
    if args.agent_thinking:
        AGENT_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": args.agent_thinking == "on"}}

    # Load instances
    instances_path = Path(args.instances)
    if not instances_path.exists():
        sys.exit(f"Instances file not found: {instances_path}")
    instances = []
    with instances_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))
    if args.limit:
        instances = instances[:args.limit]
    if args.max_interact is not None:
        globals()["MAX_INTERACT"] = args.max_interact
        print(f"MAX_INTERACT overridden to {args.max_interact}")
    print(f"Loaded {len(instances)} instances")

    # Load passage texts for oracle retrieval (keyed by passage_id)
    passage_texts: dict[str, str] = {}
    if args.passage_file:
        with open(args.passage_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    p_obj = json.loads(line)
                    passage_texts[p_obj["passage_id"]] = p_obj.get("passage_text", "")

    # Inject passage_text into instances
    n_fallback = 0
    for inst in instances:
        if "passage_text" not in inst and inst.get("passage_id") in passage_texts:
            inst["passage_text"] = passage_texts[inst["passage_id"]]
        elif "passage_text" not in inst:
            inst["passage_text"] = inst.get("context", "")[:500]
            n_fallback += 1
    if n_fallback and any(m in ("answer+search", "answer+search+interact") for m in args.modes):
        print(f"\n  *** WARNING: {n_fallback}/{len(instances)} instances have NO retrieval "
              f"passage; search will return the disambiguator C (NO answer values). "
              f"+search/+interact accuracy will be invalid. Pass --passage-file "
              f"data/sources/passages.jsonl. ***\n", file=sys.stderr)

    # Build clients
    oai = judge_client()   # OpenAI, or OpenRouter when JUDGE_CLIENT is set
    def get_client(model_name: str) -> OpenAI:
        # Local/self-hosted agent (vLLM, SGLang, TGI) — any OpenAI-compatible URL.
        # Routes the AGENT only; simulator/grader use OpenAI() directly above.
        if args.agent_base_url:
            return OpenAI(api_key=args.agent_api_key, base_url=args.agent_base_url)
        if "deepseek" in model_name.lower():
            return OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
                base_url=args.deepseek_base_url,
            )
        if "qwen" in model_name.lower():
            return OpenAI(
                api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
                base_url=args.qwen_base_url,
            )
        if "glm" in model_name.lower():
            return OpenAI(
                api_key=os.environ.get("ZHIPU_API_KEY", ""),
                base_url=args.zhipu_base_url,
            )
        return oai

    # Prepare output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    # Resume support: skip (model, mode, instance) tuples already on disk so a run
    # interrupted by exhausted credit can be continued after a top-up without
    # re-spending. Prior results and token usage are reloaded so the final summary
    # and cost sidecar stay complete.
    done: set = set()
    open_mode = "w"
    if args.resume and out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                done.add((r.get("model"), r.get("mode"), r.get("instance_id")))
                all_results.append(r)
        open_mode = "a"
        usage_path = Path(args.out).with_suffix(".usage.json")
        if usage_path.exists():
            try:
                TOKEN_USAGE.update(json.loads(usage_path.read_text(encoding="utf-8")))
            except Exception:
                pass
        print(f"Resume: {len(done)} results already on disk; skipping those.")

    # Collect all (model, mode) combos to run
    combos = [(m, mo) for m in args.models for mo in args.modes]
    print(f"Running {len(combos)} model×mode combos × {len(instances)} instances "
          f"= {len(combos)*len(instances)} evaluations")

    with out_path.open(open_mode, encoding="utf-8") as f_out:
        for model_name, mode in combos:
            client = get_client(model_name)
            label  = f"{model_name}/{mode}"
            if args.forced_n > 0:
                label += f"/forced{args.forced_n}"
            print(f"\n=== {label} ===")

            model_results = []
            for i, inst in enumerate(instances):
                inst_id = inst.get("instance_id", inst.get("id"))
                if (model_name, mode, inst_id) in done:
                    model_results.append(next(
                        (r for r in all_results
                         if (r.get("model"), r.get("mode"), r.get("instance_id"))
                         == (model_name, mode, inst_id)), {}))
                    continue
                print(f"  [{i+1}/{len(instances)}] {inst_id} "
                      f"axes={inst.get('axes',[])} ", end="", flush=True)

                try:
                    result = run_agent(
                        instance     = inst,
                        mode         = mode,
                        agent_client = client,
                        sim_client   = oai,
                        model_name   = model_name,
                        forced_n     = args.forced_n if mode == "answer+search+interact" else 0,
                        user_sim_mode = args.user_sim,
                        elicit_conf  = args.elicit_confidence,
                    )
                except CreditExhausted as e:
                    try:
                        Path(args.out).with_suffix(".usage.json").write_text(
                            json.dumps(TOKEN_USAGE, indent=2), encoding="utf-8")
                    except Exception:
                        pass
                    print(f"\n*** CREDIT EXHAUSTED: {e}\n*** Halting cleanly. "
                          f"Top up and re-run with the SAME command + --resume to continue "
                          f"(this instance was NOT recorded).", file=sys.stderr)
                    raise SystemExit(3)
                except Exception as e:
                    print(f"ERROR: {e}", file=sys.stderr)
                    result = {
                        "instance_id": inst.get("instance_id"),
                        "model": model_name, "mode": mode,
                        "correct": False, "final_answer": "",
                        "n_turns": 0, "n_asks": 0, "n_searches": 0,
                        "interacted": False, "error": str(e),
                        "axes": inst.get("axes", []),
                        "h0": inst.get("h0", 0.0),
                        "language": inst.get("language", "en"),
                    }

                print(f"{'✓' if result['correct'] else '✗'} "
                      f"asks={result.get('n_asks',0)}", flush=True)
                all_results.append(result)
                model_results.append(result)
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()

            metrics = compute_metrics(model_results)
            print(f"  Summary: acc={metrics['accuracy']:.1%}  "
                  f"ir={metrics['ir_rate']:.1%}  "
                  f"dise+={metrics.get('mean_dise_plus')}  "
                  f"turns={metrics['avg_turns']:.1f}")
            # Persist cumulative token usage after every combo so an interrupted run
            # still leaves exact cost data on disk (keyed by model: agent + openai/gpt-5
            # simulator + openai/gpt-4o-mini grader are tracked separately).
            try:
                Path(args.out).with_suffix(".usage.json").write_text(
                    json.dumps(TOKEN_USAGE, indent=2), encoding="utf-8")
            except Exception:
                pass

    # Aggregate summary per model×mode
    summary: dict[str, dict] = {}
    for model_name in args.models:
        for mode in args.modes:
            key = f"{model_name}/{mode}"
            if args.forced_n > 0:
                key += f"/forced{args.forced_n}"
            if args.user_sim != "llm":
                key += f"/{args.user_sim}"
            subset = [r for r in all_results
                      if r["model"] == model_name and r["mode"] == mode]
            summary[key] = compute_metrics(subset)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary written to {summary_path}")
    print(f"Results written to {out_path}")

    # Print final table
    print("\n=== RESULTS TABLE ===")
    print(f"{'Model/Mode':<44} {'Acc':>6} {'IR':>6} {'DisE+':>7} {'AHit':>6} {'AH@1':>6} {'Turns':>6}")
    print("-" * 82)
    for key, m in summary.items():
        dise_str = f"{m['mean_dise_plus_all']:.3f}" if m.get("mean_dise_plus_all") is not None else "  N/A"
        ah_str   = f"{m['axis_hit_rate']:.2f}"  if m.get("axis_hit_rate")  is not None else " N/A"
        ah1_str  = f"{m['axis_hit_at1']:.2f}"   if m.get("axis_hit_at1")   is not None else " N/A"
        print(f"{key:<44} {m['accuracy']:>5.1%} {m['ir_rate']:>5.1%} "
              f"{dise_str:>7} {ah_str:>6} {ah1_str:>6} {m['avg_turns']:>5.1f}")


if __name__ == "__main__":
    main()
