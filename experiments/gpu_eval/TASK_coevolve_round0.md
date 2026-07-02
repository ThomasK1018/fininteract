# TASK: Co-evolution round-0 (GATED go/no-go for FinInteract-Evolve)

**Read `experiments/coevolve/DESIGN.md` first.** This runs ONE co-evolution round
on the weakest axis and decides whether a multi-round method is worth building.
Target axis **W = `recognition_policy`** (AxisHit@1 = 0; only 9 human items).

**Go/no-go:** ΔAxisHit@1 on the **9 frozen human `recognition_policy` items** ≥ +15pt,
de-leaked accuracy up, `entity_scope`/`metric_definition` AxisHit not degraded.
Report the verdict honestly — a NO-GO is a valid, publishable result.

## Preconditions
```bash
export OPENAI_API_KEY=sk-...          # Constructor(optional), simulator, grader
git pull                              # this brief + DESIGN.md + fixed eval_model.sh
mkdir -p data/coevolve/round0 outputs/coevolve
```
Serve the base policy locally (agent + judges point here):
```bash
./experiments/gpu_eval/serve.sh qwen3-4b Qwen/Qwen3-4B-Instruct 1 >/tmp/vllm_4b.log 2>&1 &
until curl -sf http://localhost:8000/v1/models >/dev/null; do sleep 10; done
```

## Step 0 — Round-0 baseline on the FROZEN human probe
Slice the 9 human `recognition_policy` items into a probe file (never trained on):
```bash
python3 - <<'PY'
import json
rows=[json.loads(l) for l in open("data/final/fininteract_v1.jsonl")]
probe=[r for r in rows if "recognition_policy" in (r.get("axes") or [r.get("axis")])]
json.dump(probe, open("data/coevolve/round0/probe_human_recognition.jsonl","w"))
open("data/coevolve/round0/probe_human_recognition.jsonl","w").writelines(
    json.dumps(r,ensure_ascii=False)+"\n" for r in probe)
print("probe n =", len(probe))
PY
# Leak-proof AxisHit + DE-LEAKED accuracy (retrieval, NOT oracle span):
python scripts/evaluate.py --instances data/coevolve/round0/probe_human_recognition.jsonl \
  --models qwen3-4b --modes answer+search+interact \
  --passage-file data/sources/passages.jsonl \
  --agent-base-url http://localhost:8000/v1 \
  --out data/coevolve/round0/base_probe.jsonl \
  --summary data/coevolve/round0/base_probe.summary.json
```
Record base AxisHit@1 and accuracy on the probe. Also record base AxisHit on
`entity_scope`/`metric_definition` (from the existing round-1 eval or a quick
axis-sliced run) as the forgetting reference.

## Step 1 — DATA ARM: generate + verify frontier instances on axis W
```bash
# Generate; if the constructor has no axis flag, over-generate and filter by axis.
python scripts/construct_instances.py --source data/sources/passages.jsonl \
  --out data/coevolve/round0/gen_all.jsonl --target 400
python3 - <<'PY'
import json
gen=[json.loads(l) for l in open("data/coevolve/round0/gen_all.jsonl")]
W=[r for r in gen if "recognition_policy" in (r.get("axes") or [r.get("axis")])]
open("data/coevolve/round0/gen_W.jsonl","w").writelines(
    json.dumps(r,ensure_ascii=False)+"\n" for r in W)
print("axis-W generated:", len(W))
PY
```
**Frontier filter (the Judge arm)** — keep only instances that are
(a) solvable WITH context and (b) the base model FAILS WITHOUT context. (a) is the
Verifier's existing gate; run (b) as an answer-only pass and drop instances the
base already gets right:
```bash
python scripts/evaluate.py --instances data/coevolve/round0/gen_W.jsonl \
  --models qwen3-4b --modes answer-only \
  --agent-base-url http://localhost:8000/v1 \
  --out data/coevolve/round0/genW_baseonly.jsonl \
  --summary data/coevolve/round0/genW_baseonly.summary.json
# keep instances where base answer-only is WRONG  ->  frontier_W
python3 - <<'PY'
import json
res={json.loads(l)["instance_id"]:json.loads(l) for l in open("data/coevolve/round0/genW_baseonly.jsonl")}
gen=[json.loads(l) for l in open("data/coevolve/round0/gen_W.jsonl")]
frontier=[r for r in gen if not res.get(r["instance_id"],{}).get("correct",False)]
# split 80/20 train/heldout
k=int(len(frontier)*0.8)
open("data/coevolve/round0/train_W.jsonl","w").writelines(json.dumps(r,ensure_ascii=False)+"\n" for r in frontier[:k])
open("data/coevolve/round0/heldout_W.jsonl","w").writelines(json.dumps(r,ensure_ascii=False)+"\n" for r in frontier[k:])
print("frontier:",len(frontier)," train:",k," heldout:",len(frontier)-k)
PY
```
> Sanity: if the frontier is < ~40, raise `--target`. Report the yield
> (generated → axis-W → frontier) — it is itself a finding about how hard it is to
> auto-generate ambiguity the model can't already resolve.

## Step 2 — MODEL ARM: axis-guided SFT (+GRPO) on train_W ONLY
Reuse the §7.4 recipe. Build axis-guided SFT demos (privately reveal the gold axis
to the teacher; hint NOT stored in the trajectory), then:
```bash
accelerate launch experiments/gpu_eval/c2_sft_train_gc.py \
  --model Qwen/Qwen3-4B-Instruct \
  --data data/coevolve/round0/sft_train_W.jsonl \
  --output outputs/coevolve/sft_recognition
# (optional) continue with the KTO/GRPO stages exactly as in the tab:grpo ladder.
```
Serve the trained adapter/model on :8000 (swap the vLLM model), same as Step 0.

## Step 3 — RE-MEASURE (generalization + forgetting)
```bash
for SET in probe_human_recognition heldout_W; do
  python scripts/evaluate.py --instances data/coevolve/round0/$SET.jsonl \
    --models qwen3-4b-coevolved --modes answer+search+interact \
    --passage-file data/sources/passages.jsonl \
    --agent-base-url http://localhost:8000/v1 \
    --out data/coevolve/round0/r1_$SET.jsonl \
    --summary data/coevolve/round0/r1_$SET.summary.json
done
# forgetting check: AxisHit on entity_scope + metric_definition slices, trained model
```

## Deliverables — write `data/coevolve/round0/FINDINGS_coevolve_round0.md`
A single table:

| set | metric | Round-0 (base) | Round-1 (co-evolved) | Δ |
|---|---|--:|--:|--:|
| frozen human recognition (n=9) | AxisHit@1 | … | … | … |
| frozen human recognition (n=9) | de-leaked acc | … | … | … |
| heldout_W (generated) | AxisHit@1 | … | … | … |
| entity_scope (forgetting) | AxisHit@1 | … | … | … |
| metric_definition (forgetting) | AxisHit@1 | … | … | … |

Plus: generation yield (generated → axis-W → frontier), and the **explicit
GO / NO-GO verdict** against the decision rule. Commit to a branch
(`coevolve/round0`) and push. **Do not** merge generated instances into
`data/final/` — they are a separate `data/coevolve/round0/` artifact.

## Guardrails
- Never edit `data/final/*.jsonl`; the 9 human items are a held-out probe only.
- De-leaked accuracy MUST use `--passage-file` (retrieval), never oracle span.
- Simulator + grader stay on the OpenAI API; policy + axis judge are local.
- Report the honest verdict, NO-GO included.
