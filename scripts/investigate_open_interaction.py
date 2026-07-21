"""Decompose the open-model +Interact vs +Search accuracy delta by ask-subset.

The dense Qwen ladder (4B/8B/14B) shows a positive overall +Interact minus +Search
accuracy delta in Table~main. This checks whether the gain is a genuine interaction
effect. It splits each model's instances into those where a clarification actually
occurred (asked) and those answered directly (not-asked), and reports the search-vs-
interact accuracy on each subset plus the wrong->correct / correct->wrong conversions
on the asked subset. If the gain lives in the not-asked subset, it is answer variance
between two stochastic runs, not an interaction benefit (the model never asked).

No API calls. Usage:
  python scripts/investigate_open_interaction.py
"""
import json, glob, argparse


def load(pats):
    seen = {}
    for pat in pats:
        for f in glob.glob(pat):
            for l in open(f, encoding="utf-8"):
                l = l.strip()
                if not l:
                    continue
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    continue
                seen[(r.get("model"), r.get("mode"), r.get("instance_id"), r.get("forced_n", 0))] = r
    return list(seen.values())


def asked(r):
    n = r.get("n_asks")
    return (n > 0) if n is not None else bool(r.get("axis_hits"))


def main(a):
    rows = load(a.results)
    out = {}
    for m in a.models:
        se, it = {}, {}
        for r in rows:
            if r.get("model") != m or r.get("forced_n", 0) != 0:
                continue
            iid = r.get("instance_id")
            if r.get("mode") == "answer+search":
                se[iid] = bool(r.get("correct"))
            elif r.get("mode") == "answer+search+interact":
                it[iid] = r
        common = [i for i in it if i in se]
        A = [i for i in common if asked(it[i])]
        NA = [i for i in common if not asked(it[i])]
        sacc = lambda ids: (sum(se[i] for i in ids) / len(ids)) if ids else None
        iacc = lambda ids: (sum(bool(it[i].get("correct")) for i in ids) / len(ids)) if ids else None
        w2c = sum(1 for i in A if not se[i] and it[i].get("correct"))
        c2w = sum(1 for i in A if se[i] and not it[i].get("correct"))
        out[m] = {
            "n": len(common), "n_asked": len(A), "n_not_asked": len(NA),
            "all": {"search": sacc(common), "interact": iacc(common)},
            "asked": {"search": sacc(A), "interact": iacc(A)},
            "not_asked": {"search": sacc(NA), "interact": iacc(NA)},
            "asked_wrong_to_correct": w2c, "asked_correct_to_wrong": c2w,
        }
        d = out[m]
        print(f"\n=== {m}  (n={d['n']}, asked={d['n_asked']}, not-asked={d['n_not_asked']}) ===")
        for k in ("all", "asked", "not_asked"):
            s_, i_ = d[k]["search"], d[k]["interact"]
            if s_ is not None:
                print(f"  {k:10s} search={s_:.3f} interact={i_:.3f} delta={i_-s_:+.3f}")
        print(f"  asked conversions: wrong->correct={w2c} correct->wrong={c2w} (net {w2c-c2w} on {len(A)} asks)")
    if a.out:
        open(a.out, "w").write(json.dumps(out, indent=2))
        print("\nwrote", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+",
                   default=["data/results/eval_open_*.jsonl", "data/results/eval_ceiling_*.jsonl"])
    p.add_argument("--models", nargs="+", default=["qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b"])
    p.add_argument("--out", default="data/results/open_interaction_decomp.json")
    main(p.parse_args())
