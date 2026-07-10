#!/usr/bin/env python3
"""Multi-round co-evolution + Elo arms race (TASK_coevolve_multiround_elo).

K-round curriculum per axis. For round i=0..K-1:
  1. GENERATE against the current solver S_i: construct_fast on FRESH axis-primary passages
     (never reused across rounds -> the generator must find new frontier, which is exactly
     what abundance gates: entity has 905 primary passages, recognition only 30).
  2. FRONTIER (recall-wall adaptation): the task defines frontier as "S_i fails ANSWER-ONLY",
     but FinInteract has a recall wall (answer-only == 0 for every solver, so that filter
     never moves). We use the meaningful analogue: frontier_i = generated instances S_i fails
     to RESOLVE (answer+search+interact graded wrong). This DOES move as S_i learns to ask.
     Split frontier_i 80/20 -> train_i / battle_i (battle_i held out, never trained on).
  3. TRAIN S_{i+1}: axis-guided gated demos on the CUMULATIVE train set (train_0..train_i),
     SFT from base (accumulate-data-from-base cumulative curriculum -- logged; robust vs
     adapter-chaining). -> outputs/coevolve/mr/<axis>/S{i+1}.
  Log per round: frontier yield (the abundance ceiling shows up here), diversity
  (distinct entities + mean pairwise TF-IDF distance; no local embedding encoder available),
  and held-out human AxisHit on the frozen probe (generalization trajectory).

Tournament: every generator battle set g in 0..K-1 x every solver S_s in 0..K, INTERACTIVE
resolution (solver 'wins' iff it RESOLVES). Assemble win matrix -> coevolve_elo.py.

Resumable: every step checkpoints to a file; existing outputs are skipped.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, signal, urllib.request
from pathlib import Path

REPO = Path("/ceph/workspace/xinyu/fininteract_task")
PY_EVAL  = "/home/xinyu/fininteract_venv/bin/python"        # transformers5, serve+eval+construct
PY_TRAIN = "/home/xinyu/fininteract_train_venv/bin/python"  # transformers4.51 + bnb0.45 for QLoRA
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SERVED = "qwen3-4b"
PASSAGES = REPO / "data/sources/passages.jsonl"
PROBE = {  # frozen human held-out probe per axis (never trained on)
    "entity": REPO / "data/coevolve/entity/probe_human_entity.jsonl",
    "recognition": REPO / "data/coevolve/round0/probe_human_recognition.jsonl",
}
AXIS_ID = {"entity": "entity_scope", "recognition": "recognition_policy",
           "metric": "metric_definition", "temporal": "temporal_scope"}


def log(m):
    print(f"[mr {time.strftime('%H:%M:%S')}] {m}", flush=True)


def base_env():
    e = dict(os.environ)
    # OPENAI_API_KEY for constructor / user-sim / grader; must hit real OpenAI (not local)
    for line in open("/home/xinyu/.fininteract_env"):
        line = line.strip()
        if line.startswith("export "):
            k, _, v = line[len("export "):].partition("=")
            e[k] = v
    e.pop("OPENAI_BASE_URL", None)
    return e


# ---------------------------------------------------------------- server mgmt
class Server:
    def __init__(self, cuda, port):
        self.cuda, self.port, self.proc = cuda, port, None

    def start(self, adapter):
        self.stop()
        e = base_env()
        e["CUDA_VISIBLE_DEVICES"] = self.cuda
        e["HF_MODEL_ID"] = BASE_MODEL
        e["SERVED_NAME"] = SERVED
        e["PORT"] = str(self.port)
        e["MAX_NEW"] = "512"
        e["HF_ADAPTER"] = str(adapter) if adapter else ""
        logf = open(REPO / f"data/coevolve/mr/server_{self.port}.log", "w")
        self.proc = subprocess.Popen(
            [PY_EVAL, str(REPO / "experiments/gpu_eval/hf_openai_server.py")],
            env=e, stdout=logf, stderr=subprocess.STDOUT)
        url = f"http://localhost:{self.port}/v1/models"
        for _ in range(120):  # up to ~4 min for model load
            time.sleep(2)
            if self.proc.poll() is not None:
                raise RuntimeError(f"server died on startup, see server_{self.port}.log")
            try:
                urllib.request.urlopen(url, timeout=3).read();
                log(f"server up :{self.port} adapter={Path(adapter).name if adapter else 'BASE'}")
                return
            except Exception:
                pass
        raise RuntimeError("server did not come up in time")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try: self.proc.wait(timeout=30)
            except Exception: self.proc.kill()
        self.proc = None


# ---------------------------------------------------------------- steps
def run(cmd, env, logpath):
    with open(logpath, "w") as f:
        r = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed ({r.returncode}): {' '.join(map(str,cmd))}\n  see {logpath}")


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_fresh_source(axis_id, used_pids, out, cap):
    """Take up to `cap` unused axis-primary passages as this round's fresh slice, and mark the
    WHOLE slice used (deterministic exhaustion: recognition's 30-passage pool empties cleanly)."""
    rows = [r for r in read_jsonl(PASSAGES)
            if (r.get("candidate_axes") or [""])[0] == axis_id
            and r["passage_id"] not in used_pids]
    rows = rows[:cap]
    write_jsonl(out, rows)
    return rows


def diversity(rows):
    """distinct entities + mean pairwise TF-IDF cosine distance (lexical proxy; no local encoder)."""
    import numpy as np
    ents = {(r.get("ticker") or r.get("company") or "").upper() for r in rows if rows}
    qs = [r.get("question", "") for r in rows]
    mpd = None
    if len(qs) >= 2:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            X = TfidfVectorizer(min_df=1).fit_transform(qs)
            S = (X @ X.T).toarray()
            import numpy as np
            n = S.shape[0]; iu = np.triu_indices(n, 1)
            mpd = float(1.0 - S[iu].mean())
        except Exception:
            mpd = None
    return {"n": len(rows), "distinct_entities": len(ents), "mean_pairwise_dist": mpd}


def eval_interactive(server, adapter, instances, out, env):
    """Serve `adapter` and run answer+search+interact resolution eval. Returns rows."""
    if Path(out).exists() and read_jsonl(out):
        return read_jsonl(out)
    server.start(adapter)
    u = f"http://localhost:{server.port}/v1"
    e = dict(env); e["AGENT_BASE_URL"] = u; e["AGENT_API_KEY"] = "EMPTY"
    run([PY_EVAL, str(REPO / "scripts/evaluate.py"),
         "--instances", str(instances), "--models", SERVED,
         "--modes", "answer+search+interact",
         "--passage-file", str(PASSAGES), "--agent-base-url", u,
         "--out", str(out), "--summary", "/dev/null"],
        e, str(out) + ".log")
    return read_jsonl(out)


def axishit_on_probe(server, adapter, axis_id, probe, out, env):
    """Held-out human AxisHit@1 on the frozen probe: of instances where the agent asked, did the
    FIRST clarifying question hit the gold axis (axis_hits[0].is_hit)."""
    if not probe or not Path(probe).exists():
        return None
    rows = eval_interactive(server, adapter, probe, out, env)
    if not rows:
        return None
    asked = [r for r in rows if r.get("axis_hits")]
    n_ask = len(asked)
    ah1 = (sum(1 for r in asked if r["axis_hits"][0].get("is_hit")) / n_ask) if n_ask else 0.0
    # unconditional AxisHit@1 (miss counts as non-hit) + resolution accuracy
    ah1_uncond = (sum(1 for r in rows if r.get("axis_hits") and r["axis_hits"][0].get("is_hit"))
                  / len(rows))
    acc = sum(1 for r in rows if r.get("correct")) / len(rows)
    return {"axishit1_of_asked": round(ah1, 3), "axishit1_uncond": round(ah1_uncond, 3),
            "resolve_acc": round(acc, 3), "n": len(rows), "n_ask": n_ask,
            "ir": round(n_ask / len(rows), 3)}


def main(a):
    axis_id = AXIS_ID[a.axis]
    root = REPO / "data/coevolve/mr" / a.axis
    root.mkdir(parents=True, exist_ok=True)
    outdir = REPO / "outputs/coevolve/mr" / a.axis
    outdir.mkdir(parents=True, exist_ok=True)
    env = base_env()
    server = Server(a.cuda, a.port)
    K = a.k
    state_path = root / "state.json"
    state = json.load(open(state_path)) if state_path.exists() else {"rounds": [], "used_pids": []}
    used = set(state["used_pids"])

    def save_state():
        state["used_pids"] = sorted(used)
        json.dump(state, open(state_path, "w"), indent=2)

    solver_adapter = {0: None}  # S_0 = base
    train_files = []

    try:
        # Pre-built, solver-INDEPENDENT instance pool (generation is solver-independent; the
        # arms race lives in the frontier filter + tournament, not in construct). Its SIZE is
        # the abundance ceiling: entity 119 sustains K rounds; recognition 41 exhausts after r0.
        import random
        pool = read_jsonl(REPO / "data/coevolve/mr/_pools" / f"{axis_id}.jsonl")
        random.Random(1234).shuffle(pool)   # fixed seed -> identical slices on resume
        log(f"axis={a.axis}: pool size={len(pool)} | slice={a.slice_size}/round (abundance ceiling)")

        for i in range(K):
            log(f"===== ROUND {i} / axis={a.axis} =====")
            # 1. FRESH SLICE (partition the pool; empty slice = pool exhausted = STALL)
            gen = root / f"gen_r{i}.jsonl"
            if not gen.exists():
                slice_rows = pool[i * a.slice_size:(i + 1) * a.slice_size]
                write_jsonl(gen, slice_rows)
                if not slice_rows:
                    log(f"round {i}: STALL -- pool exhausted; generator cannot out-pace the solver")
            genrows = read_jsonl(gen)
            log(f"round {i}: fresh slice = {len(genrows)} instances")
            div = diversity(genrows)
            log(f"round {i}: generated {len(genrows)} axis instances | diversity={div}")

            # 2. FRONTIER (interactive-resolution: S_i fails answer+search+interact)
            fr_eval = root / f"gen_r{i}_Sieval.jsonl"
            frontier_yield = {"generated": len(genrows), "frontier": 0}
            if genrows:
                erows = eval_interactive(server, solver_adapter[i], gen, fr_eval, env)
                emap = {r["instance_id"]: r for r in erows}
                frontier = [r for r in genrows
                            if not emap.get(r["instance_id"], {}).get("correct", False)]
            else:
                frontier = []
            write_jsonl(root / f"frontier_r{i}.jsonl", frontier)
            # 80/20 split (deterministic by instance_id hash order)
            frontier_sorted = sorted(frontier, key=lambda r: r["instance_id"])
            nb = max(1, len(frontier_sorted) // 5) if frontier_sorted else 0
            battle = frontier_sorted[:nb]
            train_i = frontier_sorted[nb:]
            write_jsonl(root / f"battle_r{i}.jsonl", battle)
            write_jsonl(root / f"train_r{i}.jsonl", train_i)
            frontier_yield["frontier"] = len(frontier)
            frontier_yield["train"] = len(train_i); frontier_yield["battle"] = len(battle)
            log(f"round {i}: frontier={len(frontier)}/{len(genrows)} "
                f"(train {len(train_i)} / battle {len(battle)})")

            # 3. TRAIN S_{i+1} on CUMULATIVE train set, axis-guided demos, SFT from base
            adapter_out = outdir / f"S{i+1}"
            if not (adapter_out / "adapter_model.safetensors").exists():
                cum = root / f"cum_train_r{i}.jsonl"
                rows = []
                for j in range(i + 1):          # cumulative: train_0..train_i (resume-safe, from disk)
                    tf = root / f"train_r{j}.jsonl"
                    if tf.exists():
                        rows += read_jsonl(tf)
                write_jsonl(cum, rows)
                if not rows:
                    log(f"round {i}: no cumulative train rows -> S{i+1} := S{i} (cannot improve; STALL)")
                    solver_adapter[i+1] = solver_adapter[i]
                else:
                    demos = root / f"demos_r{i}.jsonl"
                    if not (demos.exists() and read_jsonl(demos)):
                        run([PY_EVAL, str(REPO / "scripts/gen_axis_guided_demos.py"),
                             "--instances", str(cum), "--axis", axis_id,
                             "--teacher-model", "gpt-5-mini", "--workers", "6",
                             "--out", str(demos)],
                            env, str(root / f"demos_r{i}.log"))
                    server.stop()  # free GPU for SFT
                    te = base_env(); te["CUDA_VISIBLE_DEVICES"] = a.cuda
                    run([PY_TRAIN, "/tmp/sft_gate.py", "--model", BASE_MODEL,
                         "--data", str(demos), "--output", str(adapter_out)],
                        te, str(root / f"sft_r{i}.log"))
                    solver_adapter[i+1] = adapter_out
            else:
                solver_adapter[i+1] = adapter_out
            log(f"round {i}: S{i+1} ready -> {solver_adapter[i+1]}")

            # per-round human AxisHit generalization
            ah = axishit_on_probe(server, solver_adapter[i+1], axis_id,
                                  PROBE.get(a.axis), root / f"probe_S{i+1}.jsonl", env)
            log(f"round {i}: held-out human AxisHit(S{i+1}) = {ah}")

            state["rounds"] = state.get("rounds", [])
            # record/replace round i
            state["rounds"] = [r for r in state["rounds"] if r.get("round") != i]
            state["rounds"].append({"round": i, "diversity": div,
                                    "frontier_yield": frontier_yield,
                                    "human_axishit_next_solver": ah})
            save_state()

        # ---------------------------------------------------------- TOURNAMENT
        log("===== TOURNAMENT =====")
        for s in range(K + 1):
            adapter = solver_adapter.get(s)
            for g in range(K):
                battle = root / f"battle_r{g}.jsonl"
                if not (battle.exists() and read_jsonl(battle)):
                    log(f"tourney g{g} s{s}: EMPTY battle set (round {g} stalled) -> skip, logged")
                    continue
                out = root / f"tourney_g{g}_s{s}.jsonl"
                eval_interactive(server, adapter, battle, out, env)
                rows = read_jsonl(out)
                w = sum(1 for r in rows if r.get("correct"))
                log(f"tourney g{g} s{s}: solver resolved {w}/{len(rows)}")
        server.stop()

        # assemble matches + Elo
        import glob, re
        matches = []
        for f in sorted(glob.glob(str(root / "tourney_g*_s*.jsonl"))):
            m = re.search(r"_g(\d+)_s(\d+)", f)
            g, s = int(m.group(1)), int(m.group(2))
            rows = read_jsonl(f)
            w = sum(1 for r in rows if r.get("correct"))
            matches.append({"gen_round": g, "solver_round": s, "solver_wins": w, "n": len(rows)})
        mj = root / "elo_matches.json"
        json.dump({"n_rounds": K + 1, "axis": a.axis, "matches": matches}, open(mj, "w"), indent=2)
        log(f"wrote {mj} ({len(matches)} cells)")
        run([PY_EVAL, str(REPO / "scripts/coevolve_elo.py"), "--matches", str(mj),
             "--out", str(REPO / f"data/results/coevolve_elo_{a.axis}.json"),
             "--fig", str(REPO / f"data/results/coevolve_elo_{a.axis}.png")],
            env, str(root / "elo.log"))
        log(f"===== axis={a.axis} DONE =====")
    finally:
        server.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--axis", required=True, choices=list(AXIS_ID))
    p.add_argument("--cuda", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--slice-size", type=int, default=40,
                   help="fresh instances drawn from the pool per round (empty slice = STALL)")
    main(p.parse_args())
