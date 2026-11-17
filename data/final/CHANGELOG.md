# FinInteract release changelog

## v1.1
- **Quarantined `fininteract_0172`** (DEERE & CO, `recognition_policy`). Flagged in
  the human-validation study as the sole consensus rejection (both annotators,
  item H5). Adjudicated as axis over-reach: the pairing splits ASC 606
  point-in-time ($56.09B) vs.\ over-time ($5.16B) revenue recognition, but the
  natural lay default to "what was Deere's revenue?" is the *total* (~$61.25B),
  not a timing sub-component, violating the plausible-default and
  intended-answer-is-"the-metric" design rules. See
  `annotation/adjudication_0172.md`.
- The instance is moved to `fininteract_v1.quarantine.jsonl` and removed from the
  public release (`fininteract_v1.1.jsonl`, N=172).

### Effect on reported results
All experiments in the paper are computed on the **frozen v1.0 evaluation
snapshot** (`fininteract_v1.jsonl`, N=173), which retains `fininteract_0172`.
Removing an item changes the evaluation denominator and may also change the numerator, depending on the run. Results on v1.1 must therefore be recomputed from the matching 172 records; v1.0 percentages must not be relabeled as v1.1 results. The paper retains the frozen v1.0 results for reproducibility.

## v1.0
- Initial frozen release: 173 bilingual instances (53 EN / 120 ZH), 5-axis
  financial ambiguity taxonomy, paired default-vs-intended interpretations.
