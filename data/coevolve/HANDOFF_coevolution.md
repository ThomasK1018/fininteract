# FinInteract Co-Evolution — Handoff

Onboarding doc for the next agent (GPT-5.6) picking up the **co-evolution / axis-guided
training** line of the FinInteract paper. Read this first, then the two findings files it points
to. Written 2026-07-11.

---

## 0. TL;DR (what to know in 60 seconds)

- **Paper spine** = *the single-gold illusion*: QA benchmarks grade one gold answer, silently
  crediting the *default* reading of an ambiguous query. FinInteract (bilingual, 173 instances,
  5-axis ambiguity taxonomy, default vs intended interpretation) makes it measurable. Failure is
  **elicitation, not knowledge** (models answer 93–95% given the resolved interpretation but
  resolve ~20% in free interaction). See `paper/main.tex` abstract + §1–6.
- **This line of work** feeds §7.6/§7.7 ("When can the benchmark refresh itself?" +
  "What gates co-evolvability?"). Question: can axis-guided SFT + a self-refreshing data loop
  teach a model to ask the right clarifying question, and what gates it?
- **Current answer (v2, on a *correct* instrument):** yes for salient axes, and the **data
  requirement scales inversely with salience** — entity (salience 0.92) learns from one 30-item
  slice; metric (0.48) needs to accumulate two; recognition (0.04) is weak/unreliable. Over-
  elicitation is fixable at inference. Two guarded rounds peak; a validation guard stops overfit.
- **⚠️ v1 had a silent infra bug** (server didn't load LoRA adapters) that produced a false
  "multi-round overfits → human transfer = 0". **Retracted.** With the fix, v1's own adapters
  transfer at entity 0.78–0.90. **Always verify adapter loading before trusting an eval.**

---

## 1. Where the results live (branches + files)

All work is on feature branches of `github.com/ThomasK1018/fininteract` (push via feature
branches only — direct-to-main is blocked by a classifier). Merged into `main`: `coevolve/*`
round0/entity/metric, `probe/per-axis`, `coevolve/when-to-ask`, `coevolve/multiround`.

| branch | what | status |
|---|---|---|
| `coevolve/round0` `coevolve/entity` `coevolve/metric` | first co-evolution probes per axis (round-0 PoC, entity split-GO, metric full-GO) | merged |
| `probe/per-axis` | per-axis linear decodability; **"learnable⟺decodable" REFUTED** (recognition most decodable, least learnable) | merged |
| `coevolve/when-to-ask` | over-elicitation gate: train-time FAILS, **inference gate recovers entity** | merged |
| `coevolve/multiround` | v1 multi-round + Elo. Frontier-yield abundance ceiling **stands**; Elo degenerate; overfit claim **RETRACTED** (see banner in its findings) | merged |
| **`coevolve/multiround-v2`** | **the current state.** Non-degenerate AxisHit-Elo, entity/metric learn+transfer, recognition stalls, full v1 retraction table, seed robustness | **pushed, HEAD of this work** |

**Read these two, in order:**
1. `data/coevolve/mr_v2/FINDINGS_multiround_elo_v2.md` — the authoritative current results.
2. `data/coevolve/mr/FINDINGS_multiround_elo.md` — v1 (has a retraction banner at top; the
   frontier-yield abundance result is still valid, the human-transfer numbers are not).

---

## 2. Results that hold (cite these)

**Instrument (v2).** WIN = **AxisHit@1** (did the first clarifying question target the gold
axis), NOT interactive resolution. AxisHit is leak-proof and not recall-walled, so the Elo matrix
is non-degenerate. (v1 used resolution → all-zero, because of the recall wall AND because the
battle set was the failure-defined frontier — a circular definition. Don't repeat that.)

**Step-0 separation** (base vs single-round vs multi-round, entity val/human AxisHit):
base 0.05/0.0 · single `sft_entity` 0.55/0.675 · multi `S1` 0.65/0.875 · multi `S3` 0.675/0.80.
Clean, monotone → the game separates checkpoints.

**Main runs** (K=3, val 40 frozen, overfit guard on):
- **entity**: val 0.68→**0.78**, human 0.63→**0.93**, solver Elo 1000→~1550–1696. Learns AND
  transfers. Verdict "SOLVER-RUNAWAY" (solver outpaces the corpus-bounded miner — not a clean
  two-sided "healthy" race; that's a structural limit of grounded generation).
- **metric**: single 30-slice = **0.0** (STALL); **accumulate 2 slices → val 0.975 / human
  0.75** (solver Elo 1000→3066); 3rd slice overfits, guard rolls back. **Accumulation is
  required for metric** (opposite of entity, where w1≈w3).
- **recognition**: flat ~0.05–0.067 val even with a fair 26-instance train set → **stall**, but
  *honest nuance*: v1's S1 re-scored to 0.444 on the n=9 human probe → recognition is
  **weakly/unreliably learnable on a scarce, noisy axis**, not a clean "resists all training."

**Peak-at-2-rounds** (robust across seeds 1234 & 7): val peaks at **S2 (round 1)**; round-2 S3
does not improve and the guard rejects it. Claim is "two guarded rounds peak," NOT
"more-rounds-better."

**v1 retraction table** (`data/coevolve/mr_v2/v1_retraction_table.json`, same adapters
re-evaluated with the fixed server): entity v1 0/0/0 → **0.875/0.90/0.775**; metric-S2 → 0.75;
recognition-S1 → 0.444.

**Salience → data-need gradient (the headline synthesis):** entity (0.92) learns from 1 slice,
metric (0.48) needs 2 accumulated, recognition (0.04) doesn't reliably learn. Decodability does
NOT predict this (all axes decodable); **salience does.**

---

## 3. Environment & infrastructure

- **Box**: `/ceph/workspace/xinyu/fininteract_task` (the repo). 8× A100-40GB. **Use GPUs the
  user frees** (they run their own jobs — historically fade/awq/qtip and run_qwen SD on some
  GPUs; ask / check `nvidia-smi` and stay off busy ones). Recent sessions used 0–3 or 4–7.
- **Two venvs (do NOT merge):**
  - `/home/xinyu/fininteract_venv/bin/python` — transformers 5.x. Serving, evaluate.py,
    construct, gen-demos, probes, plotting. Needs bnb≥0.46 for 4-bit (so NOT used for SFT).
  - `/home/xinyu/fininteract_train_venv/bin/python` — transformers 4.51.3 + bnb 0.45.3. **QLoRA
    SFT only.**
- **Serving**: `experiments/gpu_eval/hf_openai_server.py` — a custom OpenAI-compatible HF server
  (vLLM is unusable here: driver 535/CUDA mismatch + no qwen3_5_moe). Env:
  `HF_MODEL_ID`, `HF_ADAPTER` (LoRA dir — **must be set to eval an adapter**), `SERVED_NAME`
  (use `qwen3-4b`), `PORT`, `MAX_NEW`, `CUDA_VISIBLE_DEVICES`. Base model
  `Qwen/Qwen3-4B-Instruct-2507`.
- **SFT**: `/tmp/sft_gate.py` — QLoRA nf4 + grad-checkpointing, **grad_accum=1** (crucial: the
  default 8 makes tiny datasets do ~3 steps → adapter ≡ base), 6 epochs, lr 2e-4, r=16. NOTE:
  it lives in `/tmp` (ephemeral). If gone, it's a sed-copy of `experiments/gpu_eval/
  c2_sft_train_gc.py` with grad_accum=1; recreate it.
- **OpenAI** (constructor gpt-5-mini, user-sim, grader, axis-judge): key in
  `/home/xinyu/.fininteract_env` (`export OPENAI_API_KEY=...`). Scripts source it and
  `unset OPENAI_BASE_URL` so OpenAI() hits the real API, not the local server.
- **evaluate.py** routes the *agent* to `--agent-base-url` (local server) while sim/grader/
  axis-judge stay on OpenAI. Key output fields per instance: `correct`, `n_asks`,
  `axis_hits[i].{axis_pred,is_hit}`, `axis_hit_rate`. `--passage-file data/sources/passages.jsonl`
  is REQUIRED for de-leaked resolution.
- **Push**: `git push "https://x-access-token:${PAT}@github.com/ThomasK1018/fininteract.git"
  HEAD:refs/heads/<branch>`. Use the repo's own identity (`thomask1018`) + trailer
  `Co-Authored-By: Claude ...`. (Ask the user for the PAT; don't hardcode it in committed files.)

---

## 4. ⚠️ Gotchas that cost real time (read before running anything)

1. **The adapter-loading server bug (the big one).** The committed `hf_openai_server.py` at one
   point had NO `HF_ADAPTER`/PEFT code, so *every* checkpoint served base → all AxisHit ≈0.05,
   which masquerades as "instrument unsuitable." The patch is NOW committed (grep for
   `HF_ADAPTER` in the server = 1). **Always confirm `loaded LoRA adapter: ...` appears in the
   server log before trusting an eval.** A `git reset --hard` will silently drop it again if it's
   ever uncommitted.
2. **`git stash -u` grabs `outputs/`.** `outputs/coevolve/` is NOT gitignored, so `git stash -u`
   (used during branch switches) sweeps all LoRA adapters into the stash. Recover with
   `git checkout stash@{N}^3 -- outputs/coevolve/...` (the `^3` is the untracked tree). Adapters
   are ~47 MB each; do NOT commit them (regenerable from demos) — `git reset -- outputs/` before
   committing.
3. **Self-matching kill → exit 144.** `pkill -f "mr_coevolve"` / a python `/proc` scan whose
   command line *contains* the pattern string kills the calling shell (SIGTERM=143/144). Kill by
   explicit PID, or match a string not present in your own command.
4. **`set -e` in Bash tool.** A `kill` of an already-dead PID returns non-zero and aborts the
   whole compound command. Append `|| true` to best-effort kills.
5. **The recall wall.** answer-only accuracy ≈ 0 for every model (nothing is answerable from
   parametric memory). So (a) resolution-based tournament wins are ~always 0 → use AxisHit, and
   (b) a "can the model already answer?" gate must key on *correct-with-search-no-ask*, not
   answer-only-correct.
6. **Circular battle sets.** v1 defined battle = the held-out *frontier* (= instances the solver
   failed by construction) → nobody resolves them → 0/N everywhere. v2 fixes it with a **fixed
   frozen val set + miner battles** (S_g's worst-targeted items), and computes the win matrix by
   **lookup from `val_S{g}_forminer.jsonl`** (each solver's per-instance AxisHit), NOT by
   re-serving battle files (an earlier v2 bug wrote eval-output rows lacking `question` → the
   tournament errored `KeyError:'question'` → 0/20).
7. **Passage yield is source-dependent.** In `data/sources/passages.jsonl`, entity ambiguity
   constructs well from `edgar`/`cninfo` but ~not at all from `docfinqa`; metric needs
   `cninfo_metric_*`. Slicing passages in file order front-loads the low-yield source. The v2
   pipeline sidesteps this by reusing pre-built instance pools (`data/coevolve/mr/_pools/
   <axis>.jsonl`: entity 119, metric 93, recognition 41).
8. **Don't over-poll background jobs.** The interactive AxisHit evals are ~7 min each (agent
   local + sim + grader + axis-judge). A full v2 run is ~2–3 h. Launch detached (`setsid nohup`),
   give a one-line status, and check back on state.json — don't `nvidia-smi`-spin.

---

## 5. Scripts & how to run

```bash
EVAL=/home/xinyu/fininteract_venv/bin/python
# --- v2 co-evolution (the current pipeline) ---
$EVAL scripts/mr_coevolve_v2.py --axis entity --cuda 0 --port 8010 \
    --k 3 --slice-size 30 --val-size 40 --battle-size 20 --train-window 1 [--seed 7]
#   train-window 1 = single-round strength (each S_i trained on ONE slice from base)
#   train-window 3 = capped-cumulative (S_i on last 3 slices) -> tests accumulation
#   output dir namespaced by tag: <axis>[_w<W>][_v<val>][_s<seed>] (avoids clobber)
#   writes state.json (per-round val/test AxisHit + promote), elo_matches.json, and
#   data/results/coevolve_elo_v2_<tag>.{json,png}

# --- Step-0 de-risk (no training): does the AxisHit game separate existing checkpoints? ---
$EVAL scripts/mrv2_step0_derisk.py --axis entity --cuda 0 --port 8010 --val-size 40

# --- v1 retraction: re-eval old mr adapters with the FIXED server ---
$EVAL scripts/mrv2_v1_retraction_eval.py --axes entity metric recognition --cuda 0 --port 8020

# --- figures ---
$EVAL scripts/plot_multiround_v2.py           # val/test AxisHit trajectory (headline)
$EVAL scripts/coevolve_elo.py --matches <elo_matches.json> --out ... --fig ...  # Elo (BT fit)
```
Supporting tools: `scripts/construct_fast.py` (gpt-5-mini instance constructor +
three-role verifier), `scripts/gen_axis_guided_demos.py` (teacher demos for SFT),
`scripts/coevolve_elo.py` (Bradley-Terry Elo, metric-agnostic; `--synthetic` self-tests).

---

## 6. Data & artifact layout (under `data/coevolve/`)

- `mr/_pools/<axis_id>.jsonl` — reusable instance pools (solver-independent; pool SIZE = the
  abundance ceiling). `<axis_id>` ∈ entity_scope / metric_definition / recognition_policy /
  temporal_scope.
- `mr_v2/<tag>/` — per-run: `val_fixed.jsonl` (frozen held-out val, first `val_size` of the
  seeded pool shuffle), `train_slice_r*.jsonl`, `demos_r*.jsonl`, `val_S*.jsonl` (per-round val
  AxisHit), `val_S*_forminer.jsonl` (used to build the tournament by lookup), `test_probe_S*.jsonl`
  (human probe, report-only), `state.json`, `elo_matches.json`.
- Frozen **human probes** (held out, TEST-only, never trained/selected on):
  `data/coevolve/entity/probe_human_entity.jsonl` (40),
  `data/coevolve/metric/probe_human_metric.jsonl` (40),
  `data/coevolve/round0/probe_human_recognition.jsonl` (9).
- Adapters: `outputs/coevolve/mr_v2/<tag>/S{1,2,3}` (NOT committed — regenerable from demos).
- **Frozen benchmark — NEVER edit:** `data/final/fininteract_v1.jsonl` (N=173).

---

## 7. Open questions / suggested next steps

1. **Fold v2 into the paper.** §7.6/§7.7 need: the retraction (v1's "overfit/no-transfer" line),
   the salience→data-need gradient (entity 1 slice / metric 2 / recognition weak), the
   non-degenerate AxisHit-Elo figure, and softening the abstract's "recognition resists targeted
   training" to "weakly/unreliably learnable." Offered but not yet done — confirm with the user.
2. **The Elo is "solver-runaway," not a two-sided "healthy" race.** Grounded (corpus-bounded)
   generation can't manufacture adversarially-harder targeting items, so the miner can't keep
   pace. If a genuine two-sided arms race is wanted, the generator needs to *adapt difficulty*,
   which passage-grounded construction structurally can't. Worth stating as a limitation, not
   papering over.
3. **Recognition is under-powered** (n=9 human probe, 41-instance pool). Its 0.444/0.0/0.0 is
   noisy. If recognition's status matters to a claim, build a bigger recognition probe first.
4. **GRPO leg.** §7.5 mentions axis-guided SFT *and GRPO*; the co-evolution work here is SFT-only.
   The GRPO ladder (SFT→KTO→GRPO) exists in the FinInteract GRPO kit but wasn't wired into the
   multi-round loop.
5. **When-to-ask at scale.** The inference gate recovered entity; metric only kept +20 of +45.
   A confidence-calibrated gate (not base-uncertainty) is the open lever.

---

*Provenance: this line was executed by Claude (Opus 4.8) across several sessions
(2026-07-09 → 07-11). All numbers above are reproducible from the committed scripts + pools;
the only non-committed artifacts are the LoRA adapters (regenerable) and the OpenAI/PAT secrets.*
