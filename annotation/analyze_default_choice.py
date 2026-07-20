"""Exp 4a (span-match variant): which reading did blind annotators pick?

Annotators were told to copy-paste their chosen value; in practice they pasted whole
evidence spans. So instead of value-grading the free text, we match each pasted answer
against the TWO evidence spans that were shuffled into the passage (default vs intended)
by token overlap, recovering a clean forced-choice: default reading vs intended reading.

Reports, over DISTINCT annotators (dedup identical files): % chose default span, % intended,
% ambiguous; per-item majority; and genuine inter-annotator agreement on the binary choice.
No API calls.

Usage:
  python annotation/analyze_default_choice.py --sheets <en/zh annotator xlsx ...> \
      --instances data/final/fininteract_v1.jsonl
"""
import argparse, json, re, collections
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from score_baseline import _rows   # xlsx/csv reader


def toks(s):
    return set(re.findall(r"[0-9]+\.?[0-9]*|[a-z]{3,}", (s or "").lower()))


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_spans(path):
    m = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        m[d["instance_id"]] = (d.get("default_evidence_span", ""),
                               d.get("intended_evidence_span", ""))
    return m


def main(a):
    spans = load_spans(a.instances)
    # collect per (annotator, item) pasted answer; dedup identical (annotator_id, item, text)
    seen = set()
    per_item = collections.defaultdict(dict)   # iid -> {annotator: choice}
    tally = collections.Counter()
    for p in a.sheets:
        for r in _rows(p):
            iid = (r.get("instance_id") or "").strip()
            ans = (r.get("your_gold_answer") or "").strip()
            aid = (r.get("annotator_id") or Path(p).stem).strip()
            if not iid or not ans or iid not in spans:
                continue
            dsp, isp = spans[iid]
            td, ti, ta = toks(dsp), toks(isp), toks(ans)
            jd, ji = jaccard(ta, td), jaccard(ta, ti)
            if max(jd, ji) < 0.05 or abs(jd - ji) < 1e-9:
                choice = "ambiguous"
            else:
                choice = "default" if jd > ji else "intended"
            key = (aid, iid)
            if key in seen:           # dedup a re-submitted identical annotator file
                continue
            seen.add(key)
            per_item[iid][aid] = choice
            tally[choice] += 1

    total = sum(tally.values())
    annots = sorted({a for v in per_item.values() for a in v})
    print(f"distinct annotators: {len(annots)} {annots}")
    print(f"items: {len(per_item)} | commitments: {total}\n")
    print("Blind reading choice (span-match):")
    for c in ("default", "intended", "ambiguous"):
        print(f"  {c:10s} {tally[c]:4d}  ({100*tally[c]/total:5.1f}%)")

    # per-item majority + agreement
    unanimous = 0
    maj_default = 0
    agree_frac = []
    for iid, votes in per_item.items():
        vals = list(votes.values())
        cnt = collections.Counter(vals)
        top, topn = cnt.most_common(1)[0]
        agree_frac.append(topn / len(vals))
        if topn == len(vals):
            unanimous += 1
        if top == "default":
            maj_default += 1
    n_items = len(per_item)
    print(f"\nPer-item majority: default={maj_default}/{n_items} "
          f"({100*maj_default/n_items:.1f}%)")
    print(f"Unanimous items: {unanimous}/{n_items} ({100*unanimous/n_items:.1f}%)")
    print(f"Mean inter-annotator agreement (majority frac): {sum(agree_frac)/len(agree_frac):.3f}")
    print("\nReading: high 'default' share + high agreement => blind annotators independently "
          "converge on the DEFAULT reading => single-gold benchmark would encode it.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sheets", nargs="+", required=True)
    p.add_argument("--instances", default="data/final/fininteract_v1.jsonl")
    main(p.parse_args())
