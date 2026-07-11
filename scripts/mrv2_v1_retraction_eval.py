#!/usr/bin/env python3
"""v1 RETRACTION: re-eval every v1 multi-round adapter (outputs/coevolve/mr/<axis>/S{1,2,3}) on
its frozen human probe with the FIXED (adapter-loading) server, and compare to v1's committed
probe_S*.jsonl. v1 reported human AxisHit->0 ('multi-round overfits'); if the same adapter now
scores high, that number was the adapter-less-server artifact."""
import argparse, json, os, subprocess, time, urllib.request
from pathlib import Path

REPO = Path("/ceph/workspace/xinyu/fininteract_task")
PY = "/home/xinyu/fininteract_venv/bin/python"
BASE = "Qwen/Qwen3-4B-Instruct-2507"; SERVED = "qwen3-4b"
PASSAGES = REPO / "data/sources/passages.jsonl"
PROBE = {"entity": REPO/"data/coevolve/entity/probe_human_entity.jsonl",
         "metric": REPO/"data/coevolve/metric/probe_human_metric.jsonl",
         "recognition": REPO/"data/coevolve/round0/probe_human_recognition.jsonl"}

def log(m): print(f"[retract {time.strftime('%H:%M:%S')}] {m}", flush=True)
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
        lf=open(REPO/f"data/coevolve/mr_v2/retract_server_{s.port}.log","w")
        s.proc=subprocess.Popen([PY,str(REPO/"experiments/gpu_eval/hf_openai_server.py")],env=e,stdout=lf,stderr=subprocess.STDOUT)
        for _ in range(120):
            time.sleep(2)
            if s.proc.poll() is not None: raise RuntimeError("server died")
            try: urllib.request.urlopen(f"http://localhost:{s.port}/v1/models",timeout=3).read(); log(f"up adapter={Path(adapter).name if adapter else 'BASE'}"); return
            except Exception: pass
        raise RuntimeError("timeout")
    def stop(s):
        if s.proc and s.proc.poll() is None:
            s.proc.terminate()
            try: s.proc.wait(timeout=30)
            except Exception: s.proc.kill()
        s.proc=None

def eval_axishit(server,adapter,instances,out,env):
    if not (Path(out).exists() and read_jsonl(out)):
        server.start(adapter); u=f"http://localhost:{server.port}/v1"
        r=subprocess.run([PY,str(REPO/"scripts/evaluate.py"),"--instances",str(instances),
            "--models",SERVED,"--modes","answer+search+interact","--passage-file",str(PASSAGES),
            "--agent-base-url",u,"--out",str(out),"--summary","/dev/null"],
            env=env,stdout=open(str(out)+".log","w"),stderr=subprocess.STDOUT)
        if r.returncode!=0: raise RuntimeError(f"eval failed; see {out}.log")
    return read_jsonl(out)

def axishit1(rows):
    if not rows: return 0.0
    return sum(1 for x in rows if x.get("axis_hits") and x["axis_hits"][0].get("is_hit"))/len(rows)

def main(a):
    root=REPO/"data/coevolve/mr_v2/_retraction"; root.mkdir(parents=True,exist_ok=True)
    env=base_env(); srv=Server(a.cuda,a.port); table={}
    try:
        for axis in a.axes:
            probe=PROBE[axis]
            for i in (1,2,3):
                ad=REPO/f"outputs/coevolve/mr/{axis}/S{i}"
                if not (ad/"adapter_model.safetensors").exists(): log(f"skip {axis} S{i} (missing)"); continue
                rows=eval_axishit(srv,ad,probe,root/f"{axis}_S{i}_human.jsonl",env)
                corrected=axishit1(rows)
                # v1's committed number (if the v1 probe file exists)
                v1f=REPO/f"data/coevolve/mr/{axis}/probe_S{i}.jsonl"
                v1val=axishit1(read_jsonl(v1f)) if v1f.exists() else None
                table.setdefault(axis,{})[f"S{i}"]={"v1_reported":v1val,"corrected":round(corrected,3),"n":len(rows)}
                log(f"{axis} S{i}: v1_reported={v1val} -> corrected={corrected:.3f} (n={len(rows)})")
        srv.stop()
        json.dump(table,open(root/"retraction_table.json","w"),indent=2)
        print("\n===== v1 RETRACTION TABLE (human AxisHit@1) =====")
        for axis,d in table.items():
            for s,v in d.items():
                print(f"  {axis:12s} {s}: v1_reported={v['v1_reported']}  corrected={v['corrected']}")
    finally: srv.stop()

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--axes",nargs="+",default=["entity","metric","recognition"])
    p.add_argument("--cuda",required=True); p.add_argument("--port",type=int,default=8020)
    main(p.parse_args())
