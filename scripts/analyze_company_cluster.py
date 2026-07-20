"""Company-clustered bootstrap CIs + company-disjoint sensitivity (reviewer point 8).

The 173 instances come from only 61 companies, so several instances share a company/
filing. Instance-level i.i.d. bootstrap (analyze_breakdowns.py::boot_ci) then understates
uncertainty. This script resamples *companies* (clusters), keeping all within-company rows,
and reports the design effect (clustered CI width / i.i.d. CI width) so the widening is
explicit. It also reports a company-disjoint sensitivity: the spread of per-company mean
accuracy and a leave-one-company-out (jackknife) SE.

Rows in data/results/eval_*.jsonl have instance_id but no company; we join to the frozen
instance file to recover company/ticker. No API calls.

Usage:
    python scripts/analyze_company_cluster.py \
        --instances data/final/fininteract_v1.jsonl \
        --results "data/results/eval_*.jsonl" --B 4000
"""
import argparse, glob, json, math, random, collections

random.seed(0)
MODES = ["answer-only", "answer+search", "answer+search+interact"]


def acc(rs):
    return 100.0 * sum(bool(r["correct"]) for r in rs) / len(rs) if rs else float("nan")


def load_company_map(path):
    """instance_id -> cluster key (company, else ticker, else the id itself)."""
    m = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            iid = d.get("instance_id")
            key = d.get("company") or d.get("ticker") or iid
            m[iid] = key
    return m


def load_rows(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    seen = {}
    for r in rows:
        k = (r.get("model"), r.get("mode"), r.get("instance_id"), r.get("forced_n", 0))
        seen[k] = r
    return [r for r in seen.values() if r.get("forced_n", 0) == 0]


def iid_ci(rs, B):
    if not rs:
        return (float("nan"), float("nan"))
    n = len(rs); idx = list(range(n)); vals = []
    for _ in range(B):
        vals.append(acc([rs[random.choice(idx)] for _ in range(n)]))
    vals.sort()
    return vals[int(0.025 * B)], vals[int(0.975 * B)]


def cluster_ci(rs, cluster_of, B):
    """Cluster bootstrap: resample companies with replacement, keep all their rows."""
    if not rs:
        return (float("nan"), float("nan"))
    by_c = collections.defaultdict(list)
    for r in rs:
        by_c[cluster_of.get(r["instance_id"], r["instance_id"])].append(r)
    clusters = list(by_c)
    K = len(clusters); idx = list(range(K)); vals = []
    for _ in range(B):
        samp = []
        for _ in range(K):
            samp.extend(by_c[clusters[random.choice(idx)]])
        vals.append(acc(samp))
    vals.sort()
    return vals[int(0.025 * B)], vals[int(0.975 * B)]


def company_jackknife_se(rs, cluster_of):
    """Delete-one-company jackknife SE of accuracy (between-company variance)."""
    by_c = collections.defaultdict(list)
    for r in rs:
        by_c[cluster_of.get(r["instance_id"], r["instance_id"])].append(r)
    clusters = list(by_c)
    K = len(clusters)
    if K < 2:
        return float("nan")
    full = list(rs)
    theta = []
    for c in clusters:
        keep = [r for r in full if cluster_of.get(r["instance_id"], r["instance_id"]) != c]
        theta.append(acc(keep))
    mean = sum(theta) / K
    return math.sqrt((K - 1) / K * sum((t - mean) ** 2 for t in theta))


def per_company_spread(rs, cluster_of):
    by_c = collections.defaultdict(list)
    for r in rs:
        by_c[cluster_of.get(r["instance_id"], r["instance_id"])].append(r)
    means = sorted(acc(v) for v in by_c.values())
    n = len(means)
    med = means[n // 2] if n else float("nan")
    return n, med, (means[0] if means else float("nan")), (means[-1] if means else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="data/final/fininteract_v1.jsonl")
    ap.add_argument("--results", default="data/results/eval_*.jsonl")
    ap.add_argument("--B", type=int, default=4000)
    args = ap.parse_args()

    cluster_of = load_company_map(args.instances)
    rows = load_rows(args.results)
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["model"], r["mode"])].append(r)
    models = sorted({m for (m, _) in by})

    n_clusters_total = len(set(cluster_of.values()))
    print(f"loaded {len(rows)} rows | {len(cluster_of)} instances | "
          f"{n_clusters_total} companies (clusters) | models: {models}\n")

    print("=" * 104)
    print("ACCURACY: i.i.d. (instance) bootstrap  vs  COMPANY-CLUSTERED bootstrap 95% CI")
    print("=" * 104)
    print(f"{'model':16s} {'mode':24s} {'n':>4s} {'#co':>4s} {'acc%':>6s} "
          f"{'iid 95% CI':>16s} {'clustered 95% CI':>18s} {'width x':>8s} {'jk SE':>6s}")
    for m in models:
        for mode in MODES:
            rs = by.get((m, mode), [])
            if not rs:
                continue
            a = acc(rs)
            lo_i, hi_i = iid_ci(rs, args.B)
            lo_c, hi_c = cluster_ci(rs, cluster_of, args.B)
            w_i, w_c = hi_i - lo_i, hi_c - lo_c
            deff = (w_c / w_i) if w_i > 0 else float("nan")
            nco, *_ = per_company_spread(rs, cluster_of)
            jk = company_jackknife_se(rs, cluster_of)
            print(f"{m:16s} {mode:24s} {len(rs):4d} {nco:4d} {a:6.1f} "
                  f"[{lo_i:5.1f},{hi_i:5.1f}] [{lo_c:5.1f},{hi_c:5.1f}] "
                  f"{deff:7.2f}x {jk:6.1f}")

    print("\n" + "=" * 104)
    print("COMPANY-DISJOINT SENSITIVITY  (per-company mean accuracy spread; interact mode)")
    print("=" * 104)
    print(f"{'model':16s} {'#co':>4s} {'median%':>8s} {'min%':>6s} {'max%':>6s}")
    for m in models:
        rs = by.get((m, "answer+search+interact"), [])
        if not rs:
            continue
        nco, med, lo, hi = per_company_spread(rs, cluster_of)
        print(f"{m:16s} {nco:4d} {med:8.1f} {lo:6.1f} {hi:6.1f}")

    print("\nNote: 'width x' = clustered-CI width / iid-CI width (design effect > 1 means "
          "the honest CI is wider). 'jk SE' = delete-one-company jackknife SE.")


if __name__ == "__main__":
    main()
