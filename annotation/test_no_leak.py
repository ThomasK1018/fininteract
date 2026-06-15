"""Verify the human-baseline annotator sheets withhold the disambiguator.

The annotator legitimately SEES both candidate values (the retrieved passage is
non-leaky: both spans, neither marked correct). The leak we must prevent is the
*resolver*: the context C and the intended/default interpretation that say WHICH
value is intended. This checks C does not appear in any annotator-facing sheet.
"""
import csv, sys, glob

key = {r["instance_id"]: r for r in
       csv.DictReader(open("annotation/sheets/baseline/baseline_answerkey.csv", encoding="utf-8"))}
leaks = 0
for path in glob.glob("annotation/sheets/baseline/baseline_questions_*.csv"):
    for r in csv.DictReader(open(path, encoding="utf-8")):
        C = (key[r["instance_id"]].get("context_C") or "").strip()
        # the annotator sees question + retrieved_passage only
        seen = " ".join(v for k, v in r.items()
                        if k in ("company", "question", "retrieved_passage"))
        if C and len(C) > 20 and C in seen:
            print(f"LEAK: resolving context C appears in annotator sheet for {r['instance_id']}")
            leaks += 1
        # both values present in the passage (so neither is uniquely 'the answer')
        gold = (key[r["instance_id"]].get("intended_answer") or "").strip()
        dflt = (key[r["instance_id"]].get("default_answer") or "").strip()
        passage = r.get("retrieved_passage", "")
        if gold and dflt and (gold in passage) != (dflt in passage):
            print(f"WARN: only one of (gold,default) in passage for {r['instance_id']} "
                  f"-- one value uniquely readable")
print(f"leak check: {leaks} context-C leaks found")
sys.exit(1 if leaks else 0)
