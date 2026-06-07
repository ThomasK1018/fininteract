"""
Paraphrase augmentation for FinInteract — TRAINING DATA ONLY.

Generates K paraphrases of each instance's QUESTION while keeping the answer, context,
axes, and interpretations identical. Used to expand the *training* split for SFT/GRPO so
the model learns to recognize ambiguity regardless of wording.

DO NOT run this on the test/benchmark split — paraphrasing the eval set inflates the count
without adding information and risks train/test leakage. This script refuses to run unless
you pass --i-understand-train-only.

Each paraphrase:
  - keeps the SAME answer / default_answer / context / axes / interpretations / h0
  - changes only `question` (surface form)
  - is re-checked against QC rules R2 (no disambiguating terms / dates), R4 (names company),
    R5 (substantive metric) — paraphrases that leak the axis or drop the entity are discarded
  - carries `paraphrase_of: <parent_instance_id>` and the parent's passage_id, so a
    downstream split keeps all paraphrases of one instance on the SAME side of train/test

Usage:
  export OPENAI_API_KEY=...
  python scripts/augment_paraphrase.py --in fininteract_grpo_kit/data/train.jsonl \\
      --out fininteract_grpo_kit/data/train_augmented.jsonl --k 3 --i-understand-train-only
"""

import argparse
import json
import re
import sys
from pathlib import Path
from openai import OpenAI

# QC: disambiguating terms that must NOT appear in a question (mirrors construct_instances R2)
_DISAMBIG = {
    "fiscal year", "fy20", "fy 20", "calendar year", "non-gaap", "non gaap", "adjusted",
    "consolidated", "segment", "amended", "restated", "original filing", "as reported",
    "organic", "constant currency", "ebitda", "point in time", "over time", "gaap",
    "财年", "自然年", "非公认会计准则", "调整后", "合并报表", "合并口径", "分部",
    "子公司", "更正公告", "追溯重述", "扣非", "有机增长", "同口径",
}
_DATE_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|"
    r"november|december|19\d{2}|20\d{2}|q[1-4])\b", re.I)

# Metric-preservation guard: a paraphrase must stay within the SAME metric synonym set as
# the original. This catches silent metric drift (e.g. ZH 净利润 'net profit' -> 净收入
# 'net revenue' or 利润总额 'total profit', which are DIFFERENT metrics with different values).
_METRIC_SETS = [
    {"净利润", "净利", "净利润额", "利润净额", "净盈利", "归母净利润", "归属于母公司股东的净利润"},
    {"营业收入", "营收", "营业总收入", "总营收", "收入"},          # revenue (ZH)
    {"扣非净利润", "扣除非经常性损益的净利润", "扣非后净利润"},      # non-recurring-excluded
    {"operating income", "operating profit", "income from operations"},
    {"net income", "net earnings", "net profit"},
    {"revenue", "revenues", "net sales", "total revenue", "sales", "net revenue"},
    {"gross profit", "gross margin"},
    {"earnings per share", "eps", "diluted eps", "basic eps"},
    {"growth rate", "growth", "增长率", "增速"},
]


def _metric_set(text: str) -> set | None:
    """Return the metric synonym set the text's metric belongs to, or None."""
    tl = text.lower()
    best = None
    for s in _METRIC_SETS:
        for term in s:
            if term in tl or term in text:   # text for CJK, tl for EN
                # prefer the longest match to disambiguate 净利润 vs 扣非净利润
                if best is None or len(term) > best[1]:
                    best = (s, len(term))
    return best[0] if best else None

PARAPHRASE_SYSTEM = """\
You rewrite financial questions into alternative phrasings for data augmentation.
Rules:
  1. Preserve the EXACT meaning, the SAME company, and the SAME metric.
  2. Keep the metric term essentially verbatim. Do NOT swap it for a related-but-different
     metric. In particular for Chinese: 净利润 (net profit) must stay 净利润 — never change it
     to 净收入 (revenue), 利润总额 (total profit), or 营收. Only vary the surrounding wording.
  3. Keep it a question that asks for a specific value (never yes/no, never an instruction).
  4. NEVER add any disambiguating detail: no fiscal year, calendar year, quarter, segment,
     consolidated, GAAP/non-GAAP, adjusted, restated, point-in-time, over-time, dates, or years.
     The question must remain exactly as ambiguous as the original.
  5. Always keep the company name explicit.
Return ONLY a JSON array of {k} distinct rephrasings, e.g. ["...", "...", "..."]."""


def qc_ok(question: str, company: str, orig_question: str = "") -> bool:
    ql = question.lower()
    if any(t in ql for t in _DISAMBIG):
        return False
    if _DATE_RE.search(ql):
        return False
    # Metric preservation: paraphrase must stay in the original's metric synonym set.
    if orig_question:
        orig_set = _metric_set(orig_question)
        if orig_set is not None and _metric_set(question) is not orig_set:
            return False
    # must name the company: share a token with the company name or contain CJK
    has_cjk = any("一" <= c <= "鿿" for c in question)
    comp_tokens = {w.lower().strip(".,") for w in company.split() if len(w) > 2}
    named = has_cjk or any(tok in ql for tok in comp_tokens) or company.lower()[:5] in ql
    if not named:
        return False
    # must be interrogative / request a value
    if "?" not in question and not has_cjk:
        return False
    return True


def paraphrase(client: OpenAI, question: str, company: str, k: int) -> list[str]:
    sys_prompt = PARAPHRASE_SYSTEM.replace("{k}", str(k))
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": f"Company: {company}\nQuestion: {question}"}],
            temperature=1.0)
        txt = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\[.*\]", txt, re.S)
        cands = json.loads(m.group()) if m else []
    except Exception as e:
        print(f"    [warn] paraphrase failed: {e}")
        return []
    # QC + dedup against original
    out = []
    for c in cands:
        c = str(c).strip()
        if c and c != question and qc_ok(c, company, orig_question=question) and c not in out:
            out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=3, help="paraphrases per instance")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--i-understand-train-only", action="store_true",
                    help="Required acknowledgement that this is TRAINING data only.")
    ap.add_argument("--consistency-probe", action="store_true",
                    help="Generate a paraphrase-CONSISTENCY probe from the test set. Output "
                         "groups are tagged is_probe/group_id and are a DIAGNOSTIC over the "
                         "test instances — NOT added to the benchmark count.")
    args = ap.parse_args()

    if not (args.i_understand_train_only or args.consistency_probe):
        sys.exit("Refusing to run without --i-understand-train-only "
                 "(paraphrasing the test set for TRAINING is not allowed) or "
                 "--consistency-probe (diagnostic use).")

    client = OpenAI()
    rows = [json.loads(l) for l in args.inp.open() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    probe = args.consistency_probe
    out_rows = []
    n_added = 0
    for idx, inst in enumerate(rows):
        gid = inst["instance_id"]
        # In probe mode the original is also a group member (variant 0).
        orig = {**inst, "group_id": gid, **({"is_probe": True} if probe else {})}
        out_rows.append(orig)
        paras = paraphrase(client, inst["question"], inst.get("company", ""), args.k)
        for j, pq in enumerate(paras):
            aug = {**inst,
                   "instance_id": f"{gid}_p{j+1}",
                   "question": pq,
                   "paraphrase_of": gid,
                   "group_id": gid,
                   **({"is_probe": True} if probe else {})}
            out_rows.append(aug)
            n_added += 1
        print(f"[{idx+1}/{len(rows)}] {gid}: +{len(paras)} variants")

    with args.out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    kind = "CONSISTENCY PROBE (diagnostic, not benchmark count)" if probe else "TRAINING data"
    print(f"\nWrote {len(out_rows)} rows ({len(rows)} originals + {n_added} paraphrases) "
          f"-> {args.out}")
    print(f"NOTE: this file is {kind}. Group all variants by `group_id`/`paraphrase_of`.")


if __name__ == "__main__":
    main()
