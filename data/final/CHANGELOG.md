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
The instance is unsolved by every model in every interactive mode (answer-only,
+search, +interact, forced-interact); it is answerable only in the context-oracle
mode, where the resolving context is supplied outright. Removing it would shift
the oracle ceiling by <0.05pp (gpt-4o 93.06→93.02; gpt-5/gpt-5-mini 95.38→95.35,
both still "93–95%") and leave every other reported number unchanged. No finding
depends on it.

## v1.0
- Initial frozen release: 173 bilingual instances (53 EN / 120 ZH), 5-axis
  financial ambiguity taxonomy, paired default-vs-intended interpretations.
