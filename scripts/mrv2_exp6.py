#!/usr/bin/env python3
"""Exp 6 (reviewer response) -- SFT baseline comparison. Does AXIS-GUIDANCE specifically teach
axis-targeting, or does any clarification-SFT do as well? Train the same backbone on demos that
differ ONLY in which clarifying question is taught (guided / generic / random-axis / always-ask,
all on the SAME 56-instance company-disjoint train set), plus a no-train base. Evaluate AxisHit@1
on the frozen human probe (company-disjoint from train). AxisHit is pre-retrieval, so retrieval
leakage does not affect this comparison. RL arm (KTO/GRPO) is BLOCKED (trainers not in repo).
"""
import argparse, json, os, subprocess, time, urllib.request
from pathlib import Path

REPO = Path("/ceph/workspace/xinyu/fininteract_task")
PY = "/home/xinyu/fininteract_venv/bin/python"; PY_TRAIN = "/home/xinyu/fininteract_train_venv/bin/python"
BASE = "Qwen/Qwen3-4B-Instruct-2507"; SERVED = "qwen3-4b"
PASSAGES = REPO / "data/sources/passages.jsonl"
PROBE = REPO / "data/coevolve/entity/probe_human_entity.jsonl"

def log(m): print(f"[exp6 {time.strftime('%H:%M:%S')}] {m}", flush=True)
def read_jsonl(p): return [json.loads(l) for l in open(p) if l.strip()]
def base_env():
    e=dict(os.environ)
    for line in open("/home/xinyu/.fininteract_env"):
        if line.startswith("export "): k,_,v=line[7:].strip().partition("="); e[k]=v
    e.pop("OPENAI_BASE_URL", None); return e

class Server:
    def __init__(s,cuda,port): s.cuda,s.port,s.proc=cuda,port,None
    def start(s,adapter):
        s.stop(); e=base_env()
        e.update({"CUDA_VISIBLE_DEVICES":s.cuda,"HF_MODEL_ID":BASE,"SERVED_NAME":SERVED,
                  "PORT":str(s.port),"MAX_NEW":"512","HF_ADAPTER":str(adapter) if adapter else ""})
        lf=open(REPO/f"data/coevolve/baselines/srv_{s.port}.log","w")
        s.proc=subprocess.Popen([PY,str(REPO/"experiments/gpu_eval/hf_openai_server.py")],env=e,stdout=lf,stderr=subprocess.STDOUT)
        for _ in range(150):
            time.sleep(2)
            if s.proc.poll() is not None: raise RuntimeError("server died")
            try: urllib.request.urlopen(f"http://localhost:{s.port}/v1/models",timeout=3).read(); return
            except Exception: pass
        raise RuntimeError("server timeout")
    def stop(s):
        if s.proc and s.proc.poll() is None:
            s.proc.terminate()
            try: s.proc.wait(timeout=30)
            except Exception: s.proc.kill()
        s.proc=None

def run(cmd,env,logp):
    if subprocess.run(cmd,env=env,stdout=open(logp,"w"),stderr=subprocess.STDOUT).returncode!=0:
        raise RuntimeError(f"failed: see {logp}")

def eval_probe(server,adapter,out,env):
    if not (Path(out).exists() and read_jsonl(out)):
        server.start(adapter); u=f"http://localhost:{server.port}/v1"
        run([PY,str(REPO/"scripts/evaluate.py"),"--instances",str(PROBE),"--models",SERVED,
             "--modes","answer+search+interact","--passage-file",str(PASSAGES),
             "--agent-base-url",u,"--out",str(out),"--summary","/dev/null"],env,str(out)+".log")
    rows=read_jsonl(out)
    asked=[r for r in rows if r.get("axis_hits")]
    ah1=sum(1 for r in rows if r.get("axis_hits") and r["axis_hits"][0].get("is_hit"))/len(rows)
    acc=sum(1 for r in rows if r.get("correct"))/len(rows)
    ir=len(asked)/len(rows)
    return {"axishit1": round(ah1,3), "resolve_acc": round(acc,3), "ir": round(ir,3), "n": len(rows)}

def main(a):
    root=REPO/"data/coevolve/baselines"; env=base_env(); srv=Server(a.cuda,a.port)
    conds=[("base", None)] + [(m, root/f"entity_{m}.jsonl") for m in a.modes]
    results={}
    try:
        for name, demos in conds:
            adapter=None
            if demos is not None:
                adapter=REPO/"outputs/coevolve/baselines"/f"entity_{name}"
                if not (adapter/"adapter_model.safetensors").exists():
                    srv.stop(); te=base_env(); te["CUDA_VISIBLE_DEVICES"]=a.cuda
                    run([PY_TRAIN,"/tmp/sft_gate.py","--model",BASE,"--data",str(demos),"--output",str(adapter)],
                        te, str(root/f"sft_{name}.log"))
            r=eval_probe(srv, adapter, root/f"eval_{name}.jsonl", env)
            results[name]=r; log(f"{name}: AxisHit@1={r['axishit1']} IR={r['ir']} acc={r['resolve_acc']}")
        srv.stop()
        out=REPO/"data/results/exp6_sft_baselines_entity.json"
        json.dump({"axis":"entity_scope","train_n":56,"train":"company-disjoint","test":"human_probe_n40",
                   "note":"AxisHit@1 is pre-retrieval; RL(KTO/GRPO) arm blocked (trainers not in repo)",
                   "results":results}, open(out,"w"), indent=2)
        print("\n===== EXP 6 SFT BASELINES (entity, company-disjoint train -> human probe) =====")
        for n,r in results.items(): print(f"  {n:12s} AxisHit@1={r['axishit1']:.3f}  IR={r['ir']:.3f}  acc={r['resolve_acc']:.3f}")
        print(f"-> {out}")
    finally: srv.stop()

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--cuda",required=True); p.add_argument("--port",type=int,default=8040)
    p.add_argument("--modes",nargs="+",default=["guided","generic","random-axis","always-ask"])
    main(p.parse_args())
