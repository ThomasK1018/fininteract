# TASK: Co-evolution DISCRIMINATING TEST on metric_definition (abundant, weakly learnable)

**Read `experiments/coevolve/DESIGN.md` + `data/results/FINDINGS_learnability.md` first.**
This is the third co-evolution round, and the one the learnability framework makes a
**falsifiable prediction** about. Two rounds so far:
- `entity_scope` (abundant + learnable, salience 0.92) → **GO** (ΔAxisHit +0.62 on held-out human).
- `recognition_policy` (scarce + not-learnable, salience 0.04) → **NO-GO** (+0.00).

**metric_definition** is the discriminator: **abundant** (498 primary passages, 108 human
instances) but only **weakly learnable** (zero-shot AxisHit 4B .05 → 32B .26, Δ+0.21 vs
entity's +0.77). 

**PREDICTION (state before running, then report whether it held):**
metric co-evolution is a **PARTIAL GO** — ΔAxisHit@1 on held-out human metric items
**clearly above recognition's +0.00 but well short of entity's +0.62** (expect roughly
+0.2 to +0.4). If metric behaves like entity → *abundance* is the binding constraint. If
like recognition → *learnability* binds. Either way it's a clean result.

**Go/no-go bar (secondary):** ΔAxisHit ≥ +15pt on frozen human metric; watch de-leaked
accuracy for the same over-elicitation cost entity showed (IR→100%).

## Protocol — identical to the entity round (`TASK_coevolve_entity.md`), axis = metric_definition
Reuse every step; only the axis and paths change.
```bash
export OPENAI_API_KEY=sk-...
git pull
mkdir -p data/coevolve/metric outputs/coevolve
# serve base policy (same as entity round; HF server if vLLM unavailable on the box)
./experiments/gpu_eval/serve.sh qwen3-4b Qwen/Qwen3-4B-Instruct 1 >/tmp/vllm_4b.log 2>&1 &
until curl -sf http://localhost:8000/v1/models >/dev/null; do sleep 10; done
```

**Step 0 — frozen human metric probe (n=40 held out of the 108).**
```bash
python3 - <<'PY'
import json, random
rows=[json.loads(l) for l in open("data/final/fininteract_v1.jsonl")]
m=[r for r in rows if "metric_definition" in (r.get("axes") or [r.get("axis")])]
random.Random(0).shuffle(m)
open("data/coevolve/metric/probe_human_metric.jsonl","w").writelines(
    json.dumps(r,ensure_ascii=False)+"\n" for r in m[:40])
print("metric human total:",len(m)," probe:",40)
PY
python scripts/evaluate.py --instances data/coevolve/metric/probe_human_metric.jsonl \
  --models qwen3-4b --modes answer+search+interact \
  --passage-file data/sources/passages.jsonl --agent-base-url http://localhost:8000/v1 \
  --out data/coevolve/metric/base_probe.jsonl --summary data/coevolve/metric/base_probe.summary.json
```

**Step 1 — DATA ARM.** Generate + frontier-filter metric instances (abundant, so
generation is easy):
```bash
python scripts/construct_fast.py --source data/sources/passages.jsonl \
  --out data/coevolve/metric/gen_all.jsonl --target 300
# filter to metric_definition, then keep frontier (base fails answer-only), split 80/20
#   -> data/coevolve/metric/{gen_W,train_W,heldout_W}.jsonl   (mirror entity Step 1 exactly)
```

**Step 2 — MODEL ARM.** Build axis-guided SFT demos EXACTLY as in the entity/round-0
runs (privately reveal the gold axis=metric_definition to the teacher; hint NOT stored in
the trajectory; emit `{messages, instance_id, reward}` → `data/coevolve/metric/demos/sft.jsonl`),
then:
```bash
accelerate launch experiments/gpu_eval/c2_sft_train_gc.py \
  --model Qwen/Qwen3-4B-Instruct --data data/coevolve/metric/demos/sft.jsonl \
  --output outputs/coevolve/sft_metric        # grad_accum 1, ~6 epochs, as entity
# serve trained model on :8000
```

**Step 3 — RE-MEASURE** on `probe_human_metric` + `heldout_W`; check forgetting on
`entity_scope`/`recognition_policy`. (Mirror entity Step 3.)

## Deliverables — `data/coevolve/metric/FINDINGS_coevolve_metric.md`
Same table as entity, plus an explicit line: **did the PARTIAL-GO prediction hold?**
(compare ΔAxisHit to entity +0.62 and recognition +0.00), and note the de-leaked accuracy
change (over-elicitation check). Commit to branch `coevolve/metric`, push.

## Guardrails
- 40 human metric items = held-out probe, never trained on. Never edit `data/final/*`.
- De-leaked accuracy MUST use `--passage-file`. Sim/grader on OpenAI, policy local.
- Report the honest verdict AND whether the framework's prediction held.
