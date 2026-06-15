# Adjudication: fininteract_0172 (DEERE & CO, recognition_policy)

**Flagged by:** both annotators, item H5 (is the intended answer correct under the
intended interpretation?). The only consensus rejection in the 60-instance
validation sample (1.7%).

**Instance.** Q: "What was DEERE & CO's revenue?" — paired across the ASC 606
revenue-recognition *timing* disaggregation:
- intended = revenue recognized **at a point in time** = $56.09B
- default  = revenue recognized **over time** = $5.16B

**Verdict: EXCLUDE.** The instance violates two design principles:
1. **Implausible default (H2).** $56.09B + $5.16B = $61.25B = Deere's *total* FY2023
   net sales and revenues. A non-expert asked "what was Deere's revenue?" answers
   the total (~$61.25B), not the $5.16B over-time sub-component. The paired default
   must be the natural lay reading; here it is a disaggregation bucket.
2. **Intended answer is not "the revenue" (H5).** $56.09B is likewise a timing
   sub-component, not "revenue" in the ordinary sense — which is why both
   annotators marked H5 = No.

Root cause: axis over-reach. `recognition_policy` is a sparse axis (n=9); this
instance forced the axis onto data whose real ambiguity (total vs. sub-component)
is not the labeled axis. The adversarial verifier passed it (0/10 blind-solve)
because no model guesses a sub-component — masking the plausibility flaw humans
caught.

**Impact of exclusion:** none on reported results. fininteract_0172 was solved by
no model in any mode (already in the unsolved-125 bucket), so all accuracy and
finding numbers are materially unchanged. Action: move to
`data/final/fininteract_v1.quarantine.jsonl` and bump the release to v1.1.
