"""
Parallel drop-in for gen_trajectories.py — same outputs (sft.jsonl + kto.jsonl),
but runs episodes concurrently (the task is 100% API-latency-bound) and tracks
token usage so we get a real cost number.

Adds two knobs over the original:
  --concurrency N     number of episodes in flight at once (default 16)
  --sft-filter MODE   'strict' = correct AND (no-ask OR first-ask-on-gold-axis)  [original]
                      'correct' = any correct trajectory                          [higher yield]

Usage:
  export OPENAI_API_KEY=...
  python src/gen_trajectories_par.py --data data/train.jsonl --teacher gpt-5 \
      --rollouts 4 --concurrency 16 --sft-filter correct --out-dir data/
"""
from __future__ import annotations
import os, json, argparse, threading, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI

from env import run_episode
from reward import compute_reward, reward_breakdown
from gen_trajectories import make_teacher, to_sft_record, to_kto_record

# Approximate USD per 1M tokens (input, output). Edit if your account differs.
PRICES = {
    "gpt-5":       (1.25, 10.00),
    "gpt-5-mini":  (0.25,  2.00),
    "gpt-5-nano":  (0.05,  0.40),
    "gpt-4o":      (2.50, 10.00),
    "gpt-4o-mini": (0.15,  0.60),
}

usage_lock = threading.Lock()
usage = defaultdict(lambda: {"prompt": 0, "completion": 0, "calls": 0})


def build_axis_hint(inst: dict) -> str:
    """Privileged hint shown ONLY to the teacher (never stored in the trajectory).
    Steers the teacher to ask its first clarifying question on the gold ambiguity axis."""
    axes = ", ".join(inst.get("axes", [])) or "unknown"
    dist = "; ".join(inst.get("distinctive_attributes", []) or [])
    intended = inst.get("intended_interpretation", {})
    default = inst.get("default_interpretation", {})
    return (
        "PRIVATE GUIDANCE — the user CANNOT see this; never quote it. "
        f"This question is under-specified specifically on the ambiguity axis: {axes}. "
        f"The distinguishing detail is: {dist}. "
        f"Intended (correct) interpretation: {intended}. "
        f"Naive/default (wrong) interpretation: {default}. "
        "Your FIRST action MUST be a single 'interact' yes/no question that pins down THIS "
        "exact ambiguity axis (not the time period unless the axis above is temporal). "
        "Ask exactly ONE targeted question; after the user's reply, search, then answer. "
        "Do not ask about unrelated dimensions."
    )


def install_usage_tracker(client: OpenAI):
    orig = client.chat.completions.create
    def tracked(*a, **k):
        r = orig(*a, **k)
        try:
            u = r.usage
            with usage_lock:
                row = usage[k.get("model", "?")]
                row["prompt"] += u.prompt_tokens
                row["completion"] += u.completion_tokens
                row["calls"] += 1
        except Exception:
            pass
        return r
    client.chat.completions.create = tracked


def cost_report():
    total = 0.0
    print("\n=== token usage / cost (approx) ===")
    for m, row in sorted(usage.items()):
        pin, pout = PRICES.get(m, (0.0, 0.0))
        c = row["prompt"]/1e6*pin + row["completion"]/1e6*pout
        total += c
        print(f"  {m:14} calls={row['calls']:5d}  in={row['prompt']:9d}  out={row['completion']:9d}  ${c:6.2f}")
    print(f"  {'TOTAL':14} {'':5} {'':9} {'':9} {'':9} ${total:6.2f}")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/train.jsonl"))
    ap.add_argument("--teacher", default="gpt-5")
    ap.add_argument("--rollouts", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--sft-filter", choices=["strict", "correct"], default="strict")
    ap.add_argument("--teacher-base-url", default=None,
                    help="Separate OpenAI-compatible endpoint for the teacher (e.g. a 72B "
                         "vLLM server on :8001). If unset, teacher shares the judge client.")
    ap.add_argument("--teacher-api-key", default="local-vllm-dummy")
    ap.add_argument("--axis-guided", action="store_true",
                    help="Privately steer the teacher to ask its first question on the gold "
                         "axis (hint is NOT stored in the trajectory). Boosts SFT axis-hits.")
    args = ap.parse_args()

    # Judge client (sim/grader/axis) — uses OPENAI_BASE_URL / OPENAI_API_KEY from env.
    client = OpenAI(max_retries=6, timeout=180)
    install_usage_tracker(client)
    # Teacher client — optionally a separate endpoint (stronger model on its own server).
    if args.teacher_base_url:
        tclient = OpenAI(base_url=args.teacher_base_url, api_key=args.teacher_api_key,
                         max_retries=6, timeout=300)
        install_usage_tracker(tclient)
        print(f"teacher endpoint: {args.teacher_base_url} (model={args.teacher})")
    else:
        tclient = client
    teacher = make_teacher(tclient, args.teacher, args.temperature)

    instances = [json.loads(l) for l in args.data.open() if l.strip()]
    if args.limit:
        instances = instances[:args.limit]
    tasks = [(i, inst, r) for i, inst in enumerate(instances) for r in range(args.rollouts)]
    total_n = len(tasks)
    print(f"Running {total_n} episodes ({len(instances)} instances x {args.rollouts} rollouts) "
          f"teacher={args.teacher} concurrency={args.concurrency} sft-filter={args.sft_filter}")

    sft, kto = [], []
    done = {"n": 0, "ok": 0}
    done_lock = threading.Lock()
    t0 = time.time()

    def work(task):
        idx, inst, r = task
        if args.axis_guided:
            hint = build_axis_hint(inst)
            # Inject the hint only into what the teacher sees; run_episode appends only the
            # returned action text to the real (stored) message list.
            def tch(messages, _hint=hint):
                guided = [messages[0], {"role": "system", "content": _hint}] + messages[1:]
                return teacher(guided)
        else:
            tch = teacher
        traj = run_episode(inst, tch, client)
        return idx, r, inst, traj

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(work, t) for t in tasks]
        for fut in as_completed(futs):
            idx, r, inst, traj = fut.result()
            kto.append(to_kto_record(traj))
            if args.sft_filter == "strict":
                good = traj["correct"] and (traj["n_asks"] == 0 or traj["first_ask_hit"])
            else:  # 'correct'
                good = bool(traj["correct"])
            if good:
                sft.append(to_sft_record(traj))
            bd = reward_breakdown(traj)
            with done_lock:
                done["n"] += 1
                done["ok"] += int(good)
                n = done["n"]
            if n % 10 == 0 or n == total_n:
                rate = n / max(time.time() - t0, 1e-6)
                eta = (total_n - n) / max(rate, 1e-6)
                print(f"[{n}/{total_n}] last={inst.get('instance_id')} r{r} "
                      f"correct={traj['correct']} asks={traj['n_asks']} hit={traj['first_ask_hit']} "
                      f"R={bd['total']:.2f} | good_so_far={done['ok']} | {rate:.2f} ep/s ETA {eta/60:.1f}m",
                      flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "sft.jsonl").open("w") as f:
        for x in sft: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with (args.out_dir / "kto.jsonl").open("w") as f:
        for x in kto: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    dt = time.time() - t0
    pos = sum(x["label"] for x in kto)
    print(f"\nWrote {len(sft)} SFT records, {len(kto)} KTO records "
          f"(KTO label balance: {pos} desirable / {len(kto)-pos} undesirable).")
    print(f"Wall-clock: {dt/60:.1f} min for {total_n} episodes ({total_n/dt:.2f} ep/s).")
    cost_report()


if __name__ == "__main__":
    main()
