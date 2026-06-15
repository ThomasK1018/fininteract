# Annotation Kit

Two human-annotation tasks, with ready-to-run sheet generators and scorers. Both
back claims the paper currently makes without data (the §4.5 validation protocol
and the empty "Human (50-inst)" table row).

## Task 1 — Instance Validation (inter-annotator agreement)
Confirms the constructed instances are genuinely ambiguous and correctly labeled.
Two finance-literate annotators independently label the same sample on H1-H8;
we report Cohen's kappa and the pre-adjudication rejection rate.

```bash
# 1. generate the blank sheet (stratified by language x axis)
python annotation/make_validation_sheet.py --n 60 --out annotation/sheets/validation_blank.csv
cp annotation/sheets/validation_blank.csv annotation/sheets/validation_annotatorA.csv
cp annotation/sheets/validation_blank.csv annotation/sheets/validation_annotatorB.csv
# 2. each annotator fills H1-H6 (Yes/No), H7 ('ok' or corrected axis), H8 (free text), INDEPENDENTLY
# 3. score
python annotation/score_validation.py \
    --a annotation/sheets/validation_annotatorA.csv \
    --b annotation/sheets/validation_annotatorB.csv
```
H1-H8: H1 Q ambiguous without C? · H2 default plausible? · H3 C uniquely fixes
intended? · H4 C avoids stating the answer? · H5 A correct under intended? ·
H6 A_d correct under default? · H7 primary-axis label correct? · H8 what yes/no
question would you ask? Acceptance = H1-H6 all Yes after adjudication.

## Task 2 — Human Baseline (performance ceiling)
Finance-literate humans attempt a stratified 50-instance subset in two conditions:
**no-interaction** (answer with the filing available, no asking) and **interaction**
(may ask ONE yes/no question, answered by the experimenter from C). Reuses the
project grader so human numbers match the model table.

```bash
# 1. generate sheets. Annotators get baseline_questions_<lang>.csv (question ONLY).
#    The experimenter keeps baseline_answerkey.csv (gold answers + C) -- NEVER shown.
python annotation/make_baseline_sheet.py --n 50 --outdir annotation/sheets/baseline
# 2. run the session: annotator answers no-ask; then may ask one yes/no question,
#    experimenter replies Yes/No/IDK from C; annotator commits interact_answer.
# 3. grade (needs OPENAI_API_KEY; or hand-mark *_correct columns and use --self-graded)
python annotation/score_baseline.py \
    --sheets annotation/sheets/baseline/baseline_questions_en.csv \
             annotation/sheets/baseline/baseline_questions_zh.csv \
    --answerkey annotation/sheets/baseline/baseline_answerkey.csv
```

## Annotator requirements
Finance/accounting training (CFA candidate, finance degree, or buy-side/audit
experience); **the Chinese (ZH) split needs a fluent Chinese reader.** Record each
annotator's background and report it (as FinanceBench/FinanceQA do).

## Leak safety
The Task-2 annotator sheet contains the **question only** -- no gold answer, no C.
The answer key is a separate file the experimenter holds. `make_baseline_sheet.py`
enforces this split; a unit check (`test_no_leak.py`) verifies no gold answer
appears in any annotator-facing sheet.

## Outputs to send back
`validation_stats.json` (kappa + rejection rate) and `baseline_stats.json`
(human accuracy / default-capture / interaction rate / human AxisHit). I fold
these into §4.5 and the main table's Human row.

> `annotation/sheets/` is gitignored (holds gold answers); commit only the
> generator/scorer scripts and the final aggregate JSON stats.
