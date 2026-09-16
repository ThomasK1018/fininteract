#!/usr/bin/env python3
"""Exp 5 (reviewer response) -- MATCHED SAMPLE SIZE control. Does entity/metric out-learn
recognition because they are more SALIENT, or just because they had more training data? Control:
train every axis on the SAME tiny N (recognition's available sizes 9 and 26), >=5 seeds, and
report mean+-sd AxisHit@1 on the frozen human probe. If entity/metric still beat recognition at
matched N, the driver is salience, not data quantity.

Per (axis, N, seed): sample N from the generated pool (seeded), axis-guided demos -> QLoRA SFT
-> serve -> AxisHit@1 on the human probe. No val/promotion (single-shot); human probe is the
held-out test, never in the pool. Reuses the same server/SFT/demo tools as mr_coevolve_v2.
"""
import argparse, json, os, subprocess, time, urllib.request, random, statistics
from pathlib import Path

REPO = Path("/ceph/workspace/xinyu/fininteract_task")
PY = "/home/xinyu/fininteract_venv/bin/python"
PY_TRAIN = "/home/xinyu/fininteract_train_venv/bin/python"
BASE = "Qwen/Qwen3-4B-Instruct-2507"; SERVED = "qwen3-4b"
PASSAGES = REPO / "data/sources/passages.jsonl"
AXIS_ID = {"entity": "entity_scope", "metric": "metric_definition", "recognition": "recognition_policy"}
PROBE = {"entity": REPO/"data/coevolve/entity/probe_human_entity.jsonl",
         "metric": REPO/"data/coevolve/metric/probe_human_metric.jsonl",
         "recognition": REPO/"data/coevolve/round0/probe_human_recognition.jsonl"}

def log(m): print(f"[matchN {time.strftime('%H:%M:%S')}] {m}", flush=True)
def read_jsonl(p): return [json.loads(l) for l in open(p) if l.strip()]
def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    open(p,"w").writelines(json.dumps(r,ensure_ascii=False)+"\n" for r in rows)
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
        lf=open(REPO/f"data/coevolve/mr_v2/_matchedN/srv_{s.port}.log","w")
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

def run(cmd, env, logp):
    r=subprocess.run(cmd, env=env, stdout=open(logp,"w"), stderr=subprocess.STDOUT)
    if r.returncode!=0: raise RuntimeError(f"failed: see {logp}")

def eval_axishit(server, adapter, instances, out, env):
    if not (Path(out).exists() and read_jsonl(out)):
        server.start(adapter); u=f"http://localhost:{server.port}/v1"
        run([PY,str(REPO/"scripts/evaluate.py"),"--instances",str(instances),"--models",SERVED,
             "--modes","answer+search+interact","--passage-file",str(PASSAGES),
             "--agent-base-url",u,"--out",str(out),"--summary","/dev/null"], env, str(out)+".log")
    rows=read_jsonl(out)
    hit=sum(1 for r in rows if r.get("axis_hits") and r["axis_hits"][0].get("is_hit"))
    return hit/len(rows) if rows else 0.0

def main(a):
    axis_id=AXIS_ID[a.axis]; root=REPO/"data/coevolve/mr_v2/_matchedN"/a.axis; root.mkdir(parents=True,exist_ok=True)
    env=base_env(); srv=Server(a.cuda,a.port)
    pool=read_jsonl(REPO/"data/coevolve/mr/_pools"/f"{axis_id}.jsonl")
    results={}
    try:
        for N in a.sizes:
            if N>len(pool): log(f"N={N} > pool {len(pool)} for {a.axis}; skip"); continue
            ahs=[]
            for seed in a.seeds:
                tag=f"{a.axis}_N{N}_s{seed}"
                sample=list(pool); random.Random(seed).shuffle(sample); sample=sample[:N]
                write_jsonl(root/f"{tag}_train.jsonl", sample)
                demos=root/f"{tag}_demos.jsonl"
                if not (demos.exists() and read_jsonl(demos)):
                    run([PY,str(REPO/"scripts/gen_axis_guided_demos.py"),"--instances",str(root/f"{tag}_train.jsonl"),
                         "--axis",axis_id,"--teacher-model","gpt-5-mini","--workers","6","--out",str(demos)],
                        env, str(root/f"{tag}_demos.log"))
                adapter=REPO/"outputs/coevolve/mr_v2/_matchedN"/tag
                if not (adapter/"adapter_model.safetensors").exists():
                    srv.stop(); te=base_env(); te["CUDA_VISIBLE_DEVICES"]=a.cuda
                    run([PY_TRAIN,"/tmp/sft_gate.py","--model",BASE,"--data",str(demos),"--output",str(adapter)],
                        te, str(root/f"{tag}_sft.log"))
                ah=eval_axishit(srv, adapter, PROBE[a.axis], root/f"{tag}_probe.jsonl", env)
                ahs.append(ah); log(f"{tag}: human AxisHit@1={ah:.3f}")
            results[str(N)]={"seeds":a.seeds,"axishit":[round(x,3) for x in ahs],
                             "mean":round(statistics.mean(ahs),3),
                             "sd":round(statistics.pstdev(ahs),3) if len(ahs)>1 else 0.0,
                             "n_train":N}
        srv.stop()
        out=REPO/f"data/results/matchedN_{a.axis}.json"
        json.dump({"axis":a.axis,"axis_id":axis_id,"pool_size":len(pool),"results":results},open(out,"w"),indent=2)
        print(f"\n===== MATCHED-N {a.axis} =====")
        for N,r in results.items(): print(f"  N={N}: AxisHit@1 = {r['mean']:.3f} +- {r['sd']:.3f}  (seeds {r['axishit']})")
        print(f"-> {out}")
    finally: srv.stop()

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--axis",required=True,choices=list(AXIS_ID))
    p.add_argument("--cuda",required=True); p.add_argument("--port",type=int,required=True)
    p.add_argument("--sizes",type=int,nargs="+",default=[9,26])
    p.add_argument("--seeds",type=int,nargs="+",default=[1234,7,21,99,2025])
    main(p.parse_args())
