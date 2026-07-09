"""Gated demos = ask-on-axis demos (base-WRONG frontier) + no-ask (search+answer) demos
for items the base already answers correctly WITHOUT asking (n_asks==0, WITH retrieval).
FinInteract has a recall wall (answer-only=0), so 'can already answer' = correct-with-search-
no-ask, not answer-only-correct."""
import json, sys, os

tag = sys.argv[1]
D = "/ceph/workspace/xinyu/fininteract_task/data/coevolve"
SYS = open("/tmp/gate_system.txt").read()
def act(o): return json.dumps(o, ensure_ascii=False, separators=(",", ":"))

# passage_id -> passage_text
ptext = {}
for l in open("/ceph/workspace/xinyu/fininteract_task/data/sources/passages.jsonl"):
    l = l.strip()
    if l:
        o = json.loads(l); ptext[o["passage_id"]] = o.get("passage_text", "")

ask = [json.loads(l) for l in open(f"/tmp/{tag}_demos.jsonl")]           # base-wrong ask demos
nonprobe = {json.loads(l)["instance_id"]: json.loads(l)
            for l in open(f"{D}/when_to_ask_{tag}_nonprobe.jsonl")}
base = [json.loads(l) for l in open(f"{D}/when_to_ask_{tag}_baseeval.jsonl")]

no_ask = []
for b in base:
    if b.get("correct") and b.get("n_asks", 1) == 0:      # base answers correctly WITHOUT asking
        inst = nonprobe[b["instance_id"]]
        ans = str(inst.get("answer", "")).strip()
        psg = ptext.get(inst.get("passage_id"), inst.get("context", ""))[:2500]
        if not ans:
            continue
        no_ask.append({
            "messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": inst["question"]},
                {"role": "assistant", "content": act({"action": "search", "query": inst["question"]})},
                {"role": "user", "content": f"[search result] {psg}"},
                {"role": "assistant", "content": act({"action": "answer", "value": ans})},
            ],
            "instance_id": b["instance_id"], "reward": 1.0,
        })

gated = ask + no_ask
out = f"{D}/{tag}/demos/sft_gated.jsonl"
os.makedirs(os.path.dirname(out), exist_ok=True)
open(out, "w").writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in gated)
n_base_correct = sum(b.get("correct", False) for b in base)
n_noask = sum(b.get("correct") and b.get("n_asks", 1) == 0 for b in base)
print(f"{tag}: base-correct(w/search)={n_base_correct}/{len(base)}, of which no-ask={n_noask} "
      f"-> ask demos={len(ask)} + no-ask demos={len(no_ask)} = {len(gated)} "
      f"({100*len(no_ask)/max(len(gated),1):.0f}% no-ask) -> {out}")
