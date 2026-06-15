"""Verify the human-baseline annotator sheets leak no gold answers."""
import csv, json, sys, glob

key = {r["instance_id"]: r for r in
       csv.DictReader(open("annotation/sheets/baseline/baseline_answerkey.csv", encoding="utf-8"))}
leaks = 0
for path in glob.glob("annotation/sheets/baseline/baseline_questions_*.csv"):
    for r in csv.DictReader(open(path, encoding="utf-8")):
        gold = key[r["instance_id"]]["intended_answer"].strip()
        blob = " ".join(v for k, v in r.items() if k in ("company", "question"))
        if gold and gold in blob:
            print(f"LEAK: gold {gold!r} appears in annotator sheet for {r['instance_id']}")
            leaks += 1
print(f"leak check: {leaks} leaks found")
sys.exit(1 if leaks else 0)
