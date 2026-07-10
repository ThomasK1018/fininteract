#!/usr/bin/env python3
"""Multi-round co-evolution v2 -- fixes the two degeneracies found in v1.

v1 outcome: (a) the Elo tournament was DEGENERATE (0 solver wins in every cell) because
the win = interactive RESOLUTION (recall-walled -> always 0) on a battle set = the
FRONTIER (instances defined by the solver failing -> circular); (b) multi-round training
OVERFIT (cumulative SFT washed out the single-round gain; human-probe AxisHit -> 0.0).

v2 changes (everything else identical to mr_coevolve.py):
  1. WIN = AxisHit@1 (targeting), NOT resolution. AxisHit is leak-proof and NOT recall-walled
     -- a solver can target the right axis even when retrieval fails, so outcomes vary and the
     Elo matrix is non-degenerate.
  2. BATTLE = a FIXED held-out VAL set (never trained on), plus per-round MINER battles
     (the generator role): battle_g = the instances the current solver S_g targets WORST on the
     fixed val set (solver-conditioned hard-example mining). Not the failure-defined frontier.
  3. REGULARIZED training: capped-cumulative window (--train-window) + VALIDATION-BASED
     checkpoint selection -- promote S_{i+1} only if its VAL AxisHit does not regress the best
     so far; otherwise roll back (keep the best solver). This is the overfit guard v1 lacked
     (v1 only *logged* the probe, never acted on it).
  Reported TEST = the frozen human probe, used ONLY for final reporting, never for selection.

Elo tool (`coevolve_elo.py`) is unchanged -- it is metric-agnostic (takes any solver_wins/n
matrix); we simply feed it an AxisHit win matrix.
"""
from __future__ import annotations
import argparse, json, os, subprocess, time, urllib.request, random, glob, re
from pathlib import Path

REPO = Path("/ceph/workspace/xinyu/fininteract_task")
PY_EVAL  = "/home/xinyu/fininteract_venv/bin/python"
PY_TRAIN = "/home/xinyu/fininteract_train_venv/bin/python"
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SERVED = "qwen3-4b"
PASSAGES = REPO / "data/sources/passages.jsonl"
PROBE = {  # frozen human TEST probe per axis -- reporting only, NEVER used for selection
    "entity": REPO / "data/coevolve/entity/probe_human_entity.jsonl",
    "metric": REPO / "data/coevolve/metric/probe_human_metric.jsonl",
    "recognition": REPO / "data/coevolve/round0/probe_human_recognition.jsonl",
}
AXIS_ID = {"entity": "entity_scope", "recognition": "recognition_policy",
           "metric": "metric_definition", "temporal": "temporal_scope"}


def log(m): print(f"[mrv2 {time.strftime('%H:%M:%S')}] {m}", flush=True)
def read_jsonl(p): return [json.loads(l) for l in open(p) if l.strip()]
def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")


def base_env():
    e = dict(os.environ)
    for line in open("/home/xinyu/.fininteract_env"):
        line = line.strip()
        if line.startswith("export "):
            k, _, v = line[len("export "):].partition("="); e[k] = v
    e.pop("OPENAI_BASE_URL", None)
    return e


class Server:
    def __init__(self, cuda, port): self.cuda, self.port, self.proc = cuda, port, None
    def start(self, adapter):
        self.stop(); e = base_env()
        e.update({"CUDA_VISIBLE_DEVICES": self.cuda, "HF_MODEL_ID": BASE_MODEL,
                  "SERVED_NAME": SERVED, "PORT": str(self.port), "MAX_NEW": "512",
                  "HF_ADAPTER": str(adapter) if adapter else ""})
        logf = open(REPO / f"data/coevolve/mr_v2/server_{self.port}.log", "w")
        self.proc = subprocess.Popen(
            [PY_EVAL, str(REPO / "experiments/gpu_eval/hf_openai_server.py")],
            env=e, stdout=logf, stderr=subprocess.STDOUT)
        url = f"http://localhost:{self.port}/v1/models"
        for _ in range(120):
            time.sleep(2)
            if self.proc.poll() is not None:
                raise RuntimeError(f"server died, see server_{self.port}.log")
            try:
                urllib.request.urlopen(url, timeout=3).read()
                log(f"server up :{self.port} adapter={Path(adapter).name if adapter else 'BASE'}"); return
            except Exception: pass
        raise RuntimeError("server did not come up")
    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try: self.proc.wait(timeout=30)
            except Exception: self.proc.kill()
        self.proc = None


def run(cmd, env, logpath):
    with open(logpath, "w") as f:
        r = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed ({r.returncode}): {' '.join(map(str,cmd))}; see {logpath}")


def eval_axishit(server, adapter, instances, out, env):
    """Serve `adapter`, run answer+search+interact, return per-instance rows (with axis_hits)."""
    if not (Path(out).exists() and read_jsonl(out)):
        server.start(adapter)
        u = f"http://localhost:{server.port}/v1"
        run([PY_EVAL, str(REPO / "scripts/evaluate.py"), "--instances", str(instances),
             "--models", SERVED, "--modes", "answer+search+interact",
             "--passage-file", str(PASSAGES), "--agent-base-url", u,
             "--out", str(out), "--summary", "/dev/null"], env, str(out) + ".log")
    return read_jsonl(out)


def axishit1(rows):
    """Unconditional AxisHit@1: fraction of instances whose FIRST clarifying question hit gold."""
    if not rows: return 0.0, 0
    hit = sum(1 for r in rows if r.get("axis_hits") and r["axis_hits"][0].get("is_hit"))
    return hit / len(rows), len(rows)


def instance_hit(r):
    return 1 if (r.get("axis_hits") and r["axis_hits"][0].get("is_hit")) else 0


def main(a):
    axis_id = AXIS_ID[a.axis]
    tag = a.axis if a.train_window == 1 else f"{a.axis}_w{a.train_window}"   # namespace non-default windows
    if a.val_size != 40: tag += f"_v{a.val_size}"                            # namespace non-default val
    root = REPO / "data/coevolve/mr_v2" / tag
    outdir = REPO / "outputs/coevolve/mr_v2" / tag
    root.mkdir(parents=True, exist_ok=True); outdir.mkdir(parents=True, exist_ok=True)
    env = base_env(); server = Server(a.cuda, a.port); K = a.k
    state_path = root / "state.json"
    state = json.load(open(state_path)) if state_path.exists() else {"rounds": []}

    # ---- fixed pool -> disjoint TRAIN pool + frozen VAL set (validation/tournament) --------
    pool = read_jsonl(REPO / "data/coevolve/mr/_pools" / f"{axis_id}.jsonl")
    random.Random(1234).shuffle(pool)
    val = pool[:a.val_size]                 # FROZEN held-out val (selection + tournament); never trained
    train_pool = pool[a.val_size:]
    write_jsonl(root / "val_fixed.jsonl", val)
    log(f"axis={a.axis}: pool={len(pool)}  val(frozen)={len(val)}  train_pool={len(train_pool)}")

    try:
        solver = {0: None}                  # S_0 = base
        best_adapter, best_val_ah = None, -1.0     # validation-based selection state
        for i in range(K):
            log(f"===== ROUND {i} / axis={a.axis} =====")
            # 1. this round's fresh TRAIN slice (disjoint from val)
            gen = root / f"train_slice_r{i}.jsonl"
            if not gen.exists():
                write_jsonl(gen, train_pool[i*a.slice_size:(i+1)*a.slice_size])
            slice_rows = read_jsonl(gen)
            if not slice_rows:
                log(f"round {i}: STALL -- train pool exhausted");
            # 2. REGULARIZED train set: capped-cumulative window (last W rounds) -> curbs the
            #    drift that overfit v1 (full-cumulative). W=1 == repeated single-round.
            lo = max(0, i - a.train_window + 1)
            cum = []
            for j in range(lo, i + 1):
                sf = root / f"train_slice_r{j}.jsonl"
                if sf.exists(): cum += read_jsonl(sf)
            cumf = root / f"cum_r{i}.jsonl"; write_jsonl(cumf, cum)

            adapter_out = outdir / f"S{i+1}"
            if cum and not (adapter_out / "adapter_model.safetensors").exists():
                demos = root / f"demos_r{i}.jsonl"
                if not (demos.exists() and read_jsonl(demos)):
                    run([PY_EVAL, str(REPO / "scripts/gen_axis_guided_demos.py"),
                         "--instances", str(cumf), "--axis", axis_id,
                         "--teacher-model", "gpt-5-mini", "--workers", "6", "--out", str(demos)],
                        env, str(root / f"demos_r{i}.log"))
                server.stop()
                te = base_env(); te["CUDA_VISIBLE_DEVICES"] = a.cuda
                run([PY_TRAIN, "/tmp/sft_gate.py", "--model", BASE_MODEL,
                     "--data", str(demos), "--output", str(adapter_out)],
                    te, str(root / f"sft_r{i}.log"))
            cand = adapter_out if cum else solver[i]

            # 3. VALIDATION-based selection (the overfit guard): AxisHit on the frozen val set
            vrows = eval_axishit(server, cand, root / "val_fixed.jsonl",
                                 root / f"val_S{i+1}.jsonl", env)
            val_ah, _ = axishit1(vrows)
            promote = val_ah >= best_val_ah - a.tol
            log(f"round {i}: val AxisHit(S{i+1})={val_ah:.3f} best={best_val_ah:.3f} "
                f"-> {'PROMOTE' if promote else 'ROLLBACK (overfit/regression)'}")
            if promote:
                best_val_ah, best_adapter = val_ah, cand
            solver[i+1] = best_adapter          # carry the BEST solver forward, not the latest

            # 4. TEST (report only): frozen human probe AxisHit -- never used for selection
            trows = eval_axishit(server, solver[i+1], PROBE.get(a.axis),
                                 root / f"test_probe_S{i+1}.jsonl", env) if PROBE.get(a.axis) else []
            test_ah, _ = axishit1(trows)
            state["rounds"] = [r for r in state.get("rounds", []) if r.get("round") != i]
            state["rounds"].append({"round": i, "val_axishit": round(val_ah,3),
                                    "promoted": promote, "test_human_axishit": round(test_ah,3),
                                    "slice": len(slice_rows)})
            json.dump(state, open(state_path, "w"), indent=2)
            log(f"round {i}: TEST human AxisHit(S{i+1})={test_ah:.3f}")

        # ---------------------------------------------------------- MINER + AxisHit TOURNAMENT
        # generator role = miner: battle_g = the instances S_g targets WORST on the FIXED val set.
        # WIN(g,s) = # of battle_g instances S_s hits. Computed by LOOKUP from each solver's val
        # AxisHit (val_S{g}_forminer) -- NOT by re-serving battle files (which lack instance fields
        # and errored 'question' -> all-zero). Free and correct.
        log("===== TOURNAMENT (AxisHit; fixed val + miner battles) =====")
        H = {}  # solver g -> {instance_id: 0/1 AxisHit}
        for g in range(K + 1):
            gr = eval_axishit(server, solver.get(g), root / "val_fixed.jsonl",
                              root / f"val_S{g}_forminer.jsonl", env)
            H[g] = {r["instance_id"]: instance_hit(r) for r in gr}
        server.stop()
        matches = []
        for g in range(K + 1):
            order = sorted(H[g], key=lambda iid: (H[g][iid], iid))    # S_g's worst first
            battle = order[:a.battle_size]
            write_jsonl(root / f"battle_g{g}.jsonl", [{"instance_id": iid} for iid in battle])
            for s in range(K + 1):
                w = sum(H[s].get(iid, 0) for iid in battle)          # WIN = S_s AxisHit on S_g's hard set
                matches.append({"gen_round": g, "solver_round": s, "solver_wins": w, "n": len(battle)})
                log(f"tourney g{g} s{s}: AxisHit {w}/{len(battle)}")
        mj = root / "elo_matches.json"
        json.dump({"n_rounds": K, "axis": a.axis, "matches": matches}, open(mj, "w"), indent=2)
        run([PY_EVAL, str(REPO / "scripts/coevolve_elo.py"), "--matches", str(mj),
             "--out", str(REPO / f"data/results/coevolve_elo_v2_{tag}.json"),
             "--fig", str(REPO / f"data/results/coevolve_elo_v2_{tag}.png")],
            env, str(root / "elo.log"))
        log(f"===== axis={a.axis} DONE (best val AxisHit={best_val_ah:.3f}) =====")
    finally:
        server.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--axis", required=True, choices=list(AXIS_ID))
    p.add_argument("--cuda", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--slice-size", type=int, default=30, help="fresh train instances per round")
    p.add_argument("--val-size", type=int, default=40, help="frozen val held out of the pool")
    p.add_argument("--battle-size", type=int, default=20, help="miner hard-set size per generator")
    p.add_argument("--train-window", type=int, default=1,
                   help="capped-cumulative window (rounds). 1=non-cumulative (single-round strength); "
                        "raise to test whether accumulation helps once overfit is guarded")
    p.add_argument("--tol", type=float, default=0.03, help="val-AxisHit tolerance for promotion")
    main(p.parse_args())
