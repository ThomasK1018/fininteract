#!/usr/bin/env python3
"""
Anchored Elo for co-evolution arms-race tracking (generator vs. solver).

Model. A *match* pits generator checkpoint G_i against solver checkpoint S_j: the
solver attempts an instance from G_i WITH interaction; the solver "wins" if it
resolves the query, the generator "wins" otherwise. We fit a Bradley-Terry model
  P(solver S_j resolves an instance from generator G_i) = sigmoid(s_j - g_i)
by MLE (light L2 for numerical stability on near-separable cells), then report two
Elo trajectories over rounds: solver strength s and generator difficulty g, on one
scale, anchored so S_0 = 1000.

Reading the curves:
  - BOTH s and g climb over rounds, gap ~constant  -> healthy arms race (the generator
    keeps producing frontier as the solver improves) -> abundant axis (entity/metric).
  - s climbs, g flat, OR both flat                  -> generator can't keep pace / stall
    -> scarce axis hitting its availability ceiling (recognition).

Input JSON: {"n_rounds": K, "axis": "...",
             "matches": [{"gen_round": i, "solver_round": j, "solver_wins": w, "n": m}, ...]}
Run `--synthetic` for a self-test (no input needed).
"""
import json, argparse, math
import numpy as np

ELO_C = 400.0 / math.log(10.0)   # theta -> Elo scale
ANCHOR = 1000.0                  # S_0 Elo


def fit_bt(matches, K, l2=1e-3, iters=4000, lr=0.5):
    """MLE of solver strengths s[0..K] and generator difficulties g[0..K].
    Convex problem; plain gradient ascent on penalized log-likelihood."""
    s = np.zeros(K + 1); g = np.zeros(K + 1)
    M = [(m["gen_round"], m["solver_round"], m["solver_wins"], m["n"]) for m in matches]
    for _ in range(iters):
        gs = np.zeros(K + 1); gg = np.zeros(K + 1)
        for i, j, w, n in M:
            p = 1.0 / (1.0 + math.exp(-(s[j] - g[i])))
            resid = w - n * p          # d loglik / d(s_j - g_i)
            gs[j] += resid
            gg[i] -= resid
        gs -= l2 * s; gg -= l2 * g      # L2 gradient
        s += lr * gs / max(1, len(M))
        g += lr * gg / max(1, len(M))
    # shift-degeneracy: anchor solver_0, then express both on the same shifted scale
    shift = s[0]
    s = s - shift; g = g - shift
    return s, g


def to_elo(theta):
    return ANCHOR + ELO_C * theta


def analyze(matches, K, axis="?"):
    s, g = fit_bt(matches, K)
    s_elo, g_elo = to_elo(s), to_elo(g)
    gap = s_elo - g_elo
    # arms-race health: does the generator keep pace with the solver over rounds?
    solver_gain = float(s_elo[-1] - s_elo[0])
    gen_gain = float(g_elo[-1] - g_elo[0])
    if solver_gain < 25 and gen_gain < 25:
        verdict = "STALL (both flat) -- availability ceiling / not learnable"
    elif gen_gain >= 0.5 * solver_gain and solver_gain >= 25:
        verdict = "HEALTHY arms race -- generator keeps pace with the solver"
    else:
        verdict = "SOLVER RUNAWAY -- solver improves, generator cannot produce harder frontier"
    return {
        "axis": axis, "n_rounds": K,
        "solver_elo": [round(x, 1) for x in s_elo],
        "generator_elo": [round(x, 1) for x in g_elo],
        "gap_per_round": [round(x, 1) for x in gap],
        "solver_gain": round(solver_gain, 1),
        "generator_gain": round(gen_gain, 1),
        "verdict": verdict,
    }


def plot(results, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(matplotlib unavailable, skipping figure: {e})")
        return
    fig, ax = plt.subplots(figsize=(5, 3.2))
    rounds = list(range(results["n_rounds"] + 1))
    ax.plot(rounds, results["solver_elo"], "o-", label="Solver", color="#2b6cb0")
    ax.plot(rounds, results["generator_elo"], "s--", label="Generator", color="#c05621")
    ax.set_xlabel("co-evolution round"); ax.set_ylabel("Elo (S$_0$=1000)")
    ax.set_title(f"{results['axis']}: {results['verdict'].split(' -- ')[0]}")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


def synthetic():
    """Self-test: a healthy (abundant) race and a stalled (scarce) race."""
    rng = np.random.default_rng(0)
    def make(K, s_true, g_true, n=60):
        matches = []
        for i in range(K + 1):
            for j in range(K + 1):
                p = 1.0 / (1.0 + math.exp(-(s_true[j] - g_true[i])))
                w = int(rng.binomial(n, p))
                matches.append({"gen_round": i, "solver_round": j, "solver_wins": w, "n": n})
        return matches
    K = 4
    # healthy: solver and generator both climb together
    s_h = [0.0, 0.6, 1.2, 1.8, 2.4]; g_h = [0.0, 0.5, 1.1, 1.7, 2.2]
    # stall: nothing moves (generator can't make harder instances)
    s_s = [0.0, 0.1, 0.05, 0.1, 0.0]; g_s = [0.0, 0.0, 0.05, 0.0, 0.1]
    for name, s_t, g_t in [("entity (abundant, synthetic)", s_h, g_h),
                           ("recognition (scarce, synthetic)", s_s, g_s)]:
        r = analyze(make(K, s_t, g_t), K, axis=name)
        print(json.dumps(r, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matches", help="input JSON with matches + n_rounds")
    p.add_argument("--out", default="data/results/coevolve_elo.json")
    p.add_argument("--fig", default="data/results/coevolve_elo.png")
    p.add_argument("--synthetic", action="store_true", help="run self-test and exit")
    a = p.parse_args()
    if a.synthetic:
        synthetic(); return
    d = json.load(open(a.matches))
    r = analyze(d["matches"], d["n_rounds"], d.get("axis", "?"))
    print(json.dumps(r, indent=2))
    import os; os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(r, open(a.out, "w"), indent=2)
    plot(r, a.fig)


if __name__ == "__main__":
    main()
