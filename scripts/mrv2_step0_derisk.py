#!/usr/bin/env python3
"""Step-0 de-risk (NO training): does the AxisHit@1 game separate checkpoints we ALREADY have?
Score base + single-round sft_<axis> + multi-round mr/<axis>/S{1,2,3} on the frozen v2 val set
(and the human probe) by AxisHit@1, and build a tiny AxisHit Elo. If it separates
(base low, single-round high, multi-round-S3 low=overfit) the instrument is non-degenerate."""
import argparse, json, os, subprocess, time, urllib.request, random
from pathlib import Path

REPO = Path("/ceph/workspace/xinyu/fininteract_task")
PY = "/home/xinyu/fininteract_venv/bin/python"
BASE = "Qwen/Qwen3-4B-Instruct-2507"; SERVED = "qwen3-4b"
PASSAGES = REPO / "data/sources/passages.jsonl"
AXIS_ID = {"entity": "entity_scope", "metric": "metric_definition", "recognition": "recognition_policy"}
PROBE = {"entity": REPO/"data/coevolve/entity/probe_human_entity.jsonl",
         "metric": REPO/"data/coevolve/metric/probe_human_metric.jsonl",
         "recognition": REPO/"data/coevolve/round0/probe_human_recognition.jsonl"}

def log(m): print(f"[step0 {time.strftime('%H:%M:%S')}] {m}", flush=True)
def read_jsonl(p): return [json.loads(l) for l in open(p) if l.strip()]
def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    open(p,"w").writelines(json.dumps(r,ensure_ascii=False)+"\n" for r in rows)

def base_env():
    e = dict(os.environ)
    for line in open("/home/xinyu/.fininteract_env"):
        if line.startswith("export "):
            k,_,v = line[7:].strip().partition("="); e[k]=v
    e.pop("OPENAI_BASE_URL", None); return e

class Server:
    def __init__(s, cuda, port): s.cuda,s.port,s.proc=cuda,port,None
    def start(s, adapter):
        s.stop(); e=base_env()
        e.update({"CUDA_VISIBLE_DEVICES":s.cuda,"HF_MODEL_ID":BASE,"SERVED_NAME":SERVED,
                  "PORT":str(s.port),"MAX_NEW":"512","HF_ADAPTER":str(adapter) if adapter else ""})
        lf=open(REPO/f"data/coevolve/mr_v2/step0_server_{s.port}.log","w")
        s.proc=subprocess.Popen([PY,str(REPO/"experiments/gpu_eval/hf_openai_server.py")],env=e,stdout=lf,stderr=subprocess.STDOUT)
        for _ in range(120):
            time.sleep(2)
            if s.proc.poll() is not None: raise RuntimeError("server died")
            try: urllib.request.urlopen(f"http://localhost:{s.port}/v1/models",timeout=3).read(); log(f"server up adapter={Path(adapter).name if adapter else 'BASE'}"); return
            except Exception: pass
        raise RuntimeError("server timeout")
    def stop(s):
        if s.proc and s.proc.poll() is None:
            s.proc.terminate()
            try: s.proc.wait(timeout=30)
            except Exception: s.proc.kill()
        s.proc=None

def eval_axishit(server, adapter, instances, out, env):
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
    return sum(1 for r in rows if r.get("axis_hits") and r["axis_hits"][0].get("is_hit"))/len(rows)
def hit(r): return 1 if (r.get("axis_hits") and r["axis_hits"][0].get("is_hit")) else 0

def main(a):
    axis_id=AXIS_ID[a.axis]; root=REPO/"data/coevolve/mr_v2"/a.axis; root.mkdir(parents=True,exist_ok=True)
    pool=read_jsonl(REPO/"data/coevolve/mr/_pools"/f"{axis_id}.jsonl")
    random.Random(1234).shuffle(pool); val=pool[:a.val_size]; write_jsonl(root/"val_fixed.jsonl",val)
    log(f"axis={a.axis} val(frozen)={len(val)}")
    ckpts=[("base",None),
           (f"single_sft_{a.axis}", REPO/f"outputs/coevolve/sft_{a.axis}"),
           ("multi_S1", REPO/f"outputs/coevolve/mr/{a.axis}/S1"),
           ("multi_S3", REPO/f"outputs/coevolve/mr/{a.axis}/S3")]
    env=base_env(); srv=Server(a.cuda,a.port); results={}
    try:
        for name,ad in ckpts:
            if ad is not None and not (Path(ad)/"adapter_model.safetensors").exists():
                log(f"skip {name}: adapter missing {ad}"); continue
            vr=eval_axishit(srv,ad,root/"val_fixed.jsonl",root/f"step0_val_{name}.jsonl",env)
            pr=eval_axishit(srv,ad,PROBE[a.axis],root/f"step0_probe_{name}.jsonl",env) if PROBE.get(a.axis) else []
            results[name]={"val_axishit":round(axishit1(vr),3),"test_axishit":round(axishit1(pr),3),
                           "val_rows":vr}
            log(f"{name}: VAL AxisHit={results[name]['val_axishit']}  TEST(human)={results[name]['test_axishit']}")
        srv.stop()
        # tiny AxisHit Elo: solvers=checkpoints; miner battle per solver = its worst val items
        names=[n for n,_ in ckpts if n in results]
        battles={}
        for n in names:
            vr=results[n]["val_rows"]; srt=sorted(vr,key=lambda r:(hit(r),r["instance_id"]))
            battles[n]=[r["instance_id"] for r in srt[:a.battle_size]]
        matches=[]
        for gi,gn in enumerate(names):
            bset=set(battles[gn])
            for si,sn in enumerate(names):
                vr=results[sn]["val_rows"]; sub=[r for r in vr if r["instance_id"] in bset]
                w=sum(hit(r) for r in sub)
                matches.append({"gen_round":gi,"solver_round":si,"solver_wins":w,"n":len(sub)})
        mj=root/"step0_elo_matches.json"; json.dump({"n_rounds":len(names),"axis":a.axis,"names":names,"matches":matches},open(mj,"w"),indent=2)
        subprocess.run([PY,str(REPO/"scripts/coevolve_elo.py"),"--matches",str(mj),
            "--out",str(root/"step0_elo.json"),"--fig",str(root/"step0_elo.png")],
            env=env,stdout=open(root/"step0_elo.log","w"),stderr=subprocess.STDOUT)
        summ={n:{"val":results[n]["val_axishit"],"test":results[n]["test_axishit"]} for n in names}
        json.dump({"axis":a.axis,"checkpoints":summ,"solver_names":names},open(root/"step0_summary.json","w"),indent=2)
        print("\n===== STEP-0 SEPARATION ("+a.axis+") =====")
        for n in names: print(f"  {n:20s} VAL AxisHit={summ[n]['val']:.3f}  TEST human={summ[n]['test']:.3f}")
        try: print("  Elo verdict:", json.load(open(root/"step0_elo.json")).get("verdict"))
        except Exception: pass
    finally: srv.stop()

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--axis",required=True,choices=list(AXIS_ID))
    p.add_argument("--cuda",required=True); p.add_argument("--port",type=int,default=8010)
    p.add_argument("--val-size",type=int,default=40); p.add_argument("--battle-size",type=int,default=20)
    main(p.parse_args())
