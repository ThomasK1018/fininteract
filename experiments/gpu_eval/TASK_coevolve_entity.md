# TASK: Co-evolution POSITIVE CONTROL on a scale-solvable axis (entity_scope)

**Read `experiments/coevolve/DESIGN.md` first.** This is the companion to the
round-0 NO-GO (`TASK_coevolve_round0.md`, `data/coevolve/round0/`). That round
targeted `recognition_policy` — the *scale-invariant* blind spot — and failed to
install the target skill. This round targets **entity_scope**, a *scale-solvable*
axis (zero-shot AxisHit@1 rises 4B .12 → 32B .89; and in round-0 entity improved
0.09→0.79 as a mere side effect of training on recognition data).

**Purpose (positive control).** Show the loop *closes* when the skill is learnable:
if entity co-evolution GENERALIZES to held-out human items, it proves the pipeline
works and isolates the recognition NO-GO as an axis property, not a broken pipeline.
Expected: **GO**. Pairs with Finding 12 as "one dissociation, two levers — scale
and training both fix entity/metric, neither fixes recognition."

**Go/no-go:** ΔAxisHit@1 on held-out **human** entity_scope items ≥ +15pt, de-leaked
accuracy up, `recognition_policy`/`metric_definition` AxisHit not degraded.

## Preconditions
```bash
export OPENAI_API_KEY=sk-...          # simulator + grader + axis-judge (light)
git pull                              # this brief + build/eval fixes
mkdir -p data/coevolve/entity outputs/coevolve
./experiments/gpu_eval/serve.sh qwen3-4b Qwen/Qwen3-4B-Instruct 1 >/tmp/vllm_4b.log 2>&1 &
until curl -sf http://localhost:8000/v1/models >/dev/null; do sleep 10; done
```

## Step 0 — Round-0 baseline on a FROZEN human entity probe
entity_scope has **94 human instances** — hold out a real human probe (never trained
on), unlike recognition (only 9).
```bash
python3 - <<'PY'
import json, random
rows=[json.loads(l) for l in open("data/final/fininteract_v1.jsonl")]
ent=[r for r in rows if "entity_scope" in (r.get("axes") or [r.get("axis")])]
random.Random(0).shuffle(ent)
probe=ent[:40]                                   # frozen human held-out probe
json_out=open("data/coevolve/entity/probe_human_entity.jsonl","w")
json_out.writelines(json.dumps(r,ensure_ascii=False)+"\n" for r in probe)
print("entity human total:",len(ent)," probe(frozen):",len(probe))
PY
python scripts/evaluate.py --instances data/coevolve/entity/probe_human_entity.jsonl \
  --models qwen3-4b --modes answer+search+interact \
  --passage-file data/sources/passages.jsonl \
  --agent-base-url http://localhost:8000/v1 \
  --out data/coevolve/entity/base_probe.jsonl \
  --summary data/coevolve/entity/base_probe.summary.json
```
Record base AxisHit@1 + de-leaked accuracy on the probe, and base AxisHit on
`recognition_policy`/`metric_definition` (forgetting reference).

## Step 1 — DATA ARM: generate + frontier-filter entity instances
Entity is source-abundant (~905/1,847 passages have it primary — no scarcity),
so generation is easy; use the fast constructor.
```bash
python scripts/construct_fast.py --source data/sources/passages.jsonl \
  --out data/coevolve/entity/gen_all.jsonl --target 300   # entity dominates output
python3 - <<'PY'
import json
gen=[json.loads(l) for l in open("data/coevolve/entity/gen_all.jsonl")]
W=[r for r in gen if "entity_scope" in (r.get("axes") or [r.get("axis")])]
open("data/coevolve/entity/gen_W.jsonl","w").writelines(json.dumps(r,ensure_ascii=False)+"\n" for r in W)
print("axis-W generated:",len(W))
PY
# frontier = base fails answer-only
python scripts/evaluate.py --instances data/coevolve/entity/gen_W.jsonl \
  --models qwen3-4b --modes answer-only --agent-base-url http://localhost:8000/v1 \
  --out data/coevolve/entity/genW_baseonly.jsonl \
  --summary data/coevolve/entity/genW_baseonly.summary.json
python3 - <<'PY'
import json
res={json.loads(l)["instance_id"]:json.loads(l) for l in open("data/coevolve/entity/genW_baseonly.jsonl")}
gen=[json.loads(l) for l in open("data/coevolve/entity/gen_W.jsonl")]
frontier=[r for r in gen if not res.get(r["instance_id"],{}).get("correct",False)]
k=int(len(frontier)*0.8)
open("data/coevolve/entity/train_W.jsonl","w").writelines(json.dumps(r,ensure_ascii=False)+"\n" for r in frontier[:k])
open("data/coevolve/entity/heldout_W.jsonl","w").writelines(json.dumps(r,ensure_ascii=False)+"\n" for r in frontier[k:])
print("frontier:",len(frontier)," train:",k," heldout:",len(frontier)-k)
PY
```
> Target ≥80 frontier train instances (entity is abundant; raise `--target` if short).
> Report the yield — the contrast with recognition's 36-instance ceiling is itself a result.

## Step 2 — MODEL ARM: axis-guided SFT (+GRPO) on train_W ONLY
Reuse the §7.4 recipe (privately reveal the gold axis to the SFT teacher; hint NOT
stored in the trajectory). Match round-0's *properly-trained* schedule (grad_accum 1,
~20 epochs / enough steps to converge on the set).
```bash
accelerate launch experiments/gpu_eval/c2_sft_train_gc.py \
  --model Qwen/Qwen3-4B-Instruct \
  --data data/coevolve/entity/sft_train_W.jsonl \
  --output outputs/coevolve/sft_entity
# serve the trained model on :8000 (swap the vLLM model), as in Step 0.
```

## Step 3 — RE-MEASURE (generalization + forgetting)
```bash
for SET in probe_human_entity heldout_W; do
  python scripts/evaluate.py --instances data/coevolve/entity/$SET.jsonl \
    --models qwen3-4b-entity --modes answer+search+interact \
    --passage-file data/sources/passages.jsonl \
    --agent-base-url http://localhost:8000/v1 \
    --out data/coevolve/entity/r1_$SET.jsonl \
    --summary data/coevolve/entity/r1_$SET.summary.json
done
# forgetting: AxisHit on recognition_policy + metric_definition slices, trained model
```

## Deliverables — write `data/coevolve/entity/FINDINGS_coevolve_entity.md`

| set | metric | Round-0 (base) | Round-1 (co-evolved) | Δ |
|---|---|--:|--:|--:|
| frozen human entity (n=40) | AxisHit@1 | … | … | … |
| frozen human entity (n=40) | de-leaked acc | … | … | … |
| heldout_W (generated) | AxisHit@1 | … | … | … |
| recognition_policy (forgetting) | AxisHit@1 | … | … | … |
| metric_definition (forgetting) | AxisHit@1 | … | … | … |

Plus generation yield and the **explicit GO / NO-GO verdict**. Commit to branch
`coevolve/entity`, push. **Never** edit `data/final/*.jsonl`; generated instances
stay in `data/coevolve/entity/`.

## Guardrails
- The 40 human entity items are a held-out probe — never in any training split.
- De-leaked accuracy MUST use `--passage-file` (retrieval), never oracle span.
- Simulator/grader on OpenAI; policy + axis judge local. Report the honest verdict.
- If evaluate.py prints the no-passage warning, STOP.
