"""Per-axis 'skill' analysis: are the five ambiguity axes distinct skills, which
ones do models lack, and do failures genuinely separate by axis?

This is the spine that unifies the paper: taxonomy -> per-axis skill metric ->
error distribution -> PCA/clustering showing the axes are *behaviorally* real ->
the skill gaps that motivate targeted training (GRPO).

We decompose each model's per-axis competence in the +interact mode into three
skills:
  - RECOGNITION = interaction rate (does it ask when it should?)
  - TARGETING   = AxisHit@1 given it asked (does it ask about the RIGHT axis?)
  - RESOLUTION  = accuracy (does asking yield the intended answer?)
A model 'has' an axis skill only if all three are high; a low value localizes the
failure (didn't ask / asked wrong / asked right but still wrong).

Then we test whether the axes are real in *behaviour space*: represent each
instance by the vector of per-model interact behaviours (correct / asked /
axis-hit / default-captured), run PCA, and measure how well unsupervised k-means
recovers the axis labels (ARI) vs. a label-permutation null. Separation => the
taxonomy carves real, distinct failure modes, not arbitrary labels.

Run with a python that has sklearn (the repo's /opt/anaconda3 sklearn is broken):
  python3 scripts/analyze_axis_skills.py --out-json data/results/axis_skills.json \
      --out-fig paper/fig_axis_skills.png
"""
import argparse, json, glob
from collections import defaultdict
import numpy as np

MODELS = ["gpt-5", "gpt-4o", "gpt-5-mini", "qwen3-30b-a3b", "qwen3p5-35b-a3b"]
FILES = ["data/results/eval_gpt5_gpt4o.jsonl", "data/results/eval_gpt5mini.jsonl",
         "data/results/eval_open_qwen3-30b-a3b.jsonl",
         "data/results/eval_open_qwen3p5-35b-a3b.jsonl"]
INTERACT = "answer+search+interact"


def load():
    by = defaultdict(dict)   # (model, instance_id) -> interact row
    for f in FILES:
        for line in open(f):
            r = json.loads(line)
            if r.get("mode") == INTERACT:
                by[(r["model"], r["instance_id"])] = r
    return by


def primary_axis(r):
    return (r.get("axes") or ["?"])[0]


def per_axis_skill(by):
    """For each model x primary-axis: n, recognition (IR), targeting (AxisHit),
    resolution (acc), default-capture."""
    out = {}
    # collect instances per axis from any model's rows (axes are instance-level)
    axis_of = {}
    for (m, iid), r in by.items():
        axis_of[iid] = primary_axis(r)
    axes = sorted(set(axis_of.values()))
    for m in MODELS:
        rows = {iid: r for (mm, iid), r in by.items() if mm == m}
        out[m] = {}
        for ax in axes:
            sub = [r for iid, r in rows.items() if primary_axis(r) == ax]
            n = len(sub)
            if not n:
                out[m][ax] = {"n": 0}; continue
            asked = [r for r in sub if (r.get("n_asks") or 0) > 0]
            out[m][ax] = {
                "n": n,
                "recognition_IR": 100*len(asked)/n,
                "targeting_AxisHit": (np.mean([r.get("axis_hit_rate", 0.0) for r in asked])
                                      if asked else float("nan")),
                "resolution_acc": 100*np.mean([bool(r.get("correct")) for r in sub]),
                "default_capture": 100*np.mean([bool(r.get("default_captured")) for r in sub]),
            }
    return axes, out


def behaviour_matrix(by, axis_filter=None):
    """Each instance -> [per-model: correct, asked, axishit, default]. Returns
    X (n_inst x 4*n_models), axis labels, instance ids."""
    iids = sorted({iid for (_m, iid) in by})
    rows, labels, kept = [], [], []
    for iid in iids:
        present = [(m, by.get((m, iid))) for m in MODELS]
        if any(r is None for _m, r in present):
            continue
        ax = primary_axis(present[0][1])
        if axis_filter and ax not in axis_filter:
            continue
        feat = []
        for _m, r in present:
            feat += [float(bool(r.get("correct"))),
                     float((r.get("n_asks") or 0) > 0),
                     float(r.get("axis_hit_rate") or 0.0),
                     float(bool(r.get("default_captured")))]
        rows.append(feat); labels.append(ax); kept.append(iid)
    return np.array(rows), np.array(labels), kept


def axis_separation(X, labels, seed=0):
    """PCA -> 2D, k-means(k=#axes), ARI vs axis labels, with a permutation null."""
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    pcs = PCA(n_components=2, random_state=seed).fit_transform(Xs)
    uniq = sorted(set(labels)); k = len(uniq)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Xs)
    ari = adjusted_rand_score(labels, km)
    sil = silhouette_score(Xs, labels) if k > 1 and len(Xs) > k else float("nan")
    rng = np.random.default_rng(seed)
    null = [adjusted_rand_score(rng.permutation(labels), km) for _ in range(1000)]
    p = (1 + sum(z >= ari for z in null)) / (1 + len(null))
    return {"ari": ari, "ari_null_mean": float(np.mean(null)), "ari_p": p,
            "silhouette_by_axis": sil, "k": k, "n": len(Xs)}, pcs


def main(a):
    by = load()
    axes, skill = per_axis_skill(by)

    print("\n=== Per-axis SKILL decomposition (+interact) ===")
    print(f"{'model':16s} {'axis':18s} {'n':>3s} {'IR%':>6s} {'AxisHit':>8s} {'Acc%':>6s} {'Def%':>6s}")
    for m in MODELS:
        for ax in axes:
            s = skill[m][ax]
            if not s.get("n"):
                continue
            ah = s["targeting_AxisHit"]
            print(f"{m:16s} {ax:18s} {s['n']:3d} {s['recognition_IR']:6.1f} "
                  f"{(ah if ah==ah else float('nan')):8.2f} {s['resolution_acc']:6.1f} "
                  f"{s['default_capture']:6.1f}")

    # behaviour-space axis separation: full, then the two well-powered axes
    res = {"axes": axes, "skill": skill, "separation": {}}
    X, lab, _ = behaviour_matrix(by)
    sep_all, pcs_all = axis_separation(X, lab)
    res["separation"]["all_axes"] = sep_all
    print(f"\n=== Do failures separate by axis? (behaviour-space PCA + k-means) ===")
    print(f"all {sep_all['k']} axes: ARI={sep_all['ari']:.3f} (null {sep_all['ari_null_mean']:.3f}, "
          f"p={sep_all['ari_p']:.3f}), silhouette={sep_all['silhouette_by_axis']:.3f}, n={sep_all['n']}")
    Xem, labem, _ = behaviour_matrix(by, axis_filter={"entity_scope", "metric_definition"})
    sep_em, pcs_em = axis_separation(Xem, labem)
    res["separation"]["entity_vs_metric"] = sep_em
    print(f"entity vs metric (well-powered): ARI={sep_em['ari']:.3f} "
          f"(null {sep_em['ari_null_mean']:.3f}, p={sep_em['ari_p']:.3f}), "
          f"silhouette={sep_em['silhouette_by_axis']:.3f}, n={sep_em['n']}")

    if a.out_fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for axi, (pcs, lab_, title) in enumerate([
                (pcs_all, lab, f"All axes (ARI={sep_all['ari']:.2f}, p={sep_all['ari_p']:.3f})"),
                (pcs_em, labem, f"Entity vs Metric (ARI={sep_em['ari']:.2f}, p={sep_em['ari_p']:.3f})")]):
            for ax_name in sorted(set(lab_)):
                mask = lab_ == ax_name
                ax[axi].scatter(pcs[mask, 0], pcs[mask, 1], s=14, alpha=.6, label=ax_name)
            ax[axi].set_title(title); ax[axi].set_xlabel("PC1"); ax[axi].set_ylabel("PC2")
            ax[axi].legend(fontsize=7)
        fig.suptitle("Failure structure in per-model behaviour space (each point = one instance)")
        fig.tight_layout(); fig.savefig(a.out_fig, dpi=150)
        print("wrote", a.out_fig)

    if a.out_json:
        json.dump(res, open(a.out_json, "w"), indent=2, default=float)
        print("wrote", a.out_json)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-json", default="data/results/axis_skills.json")
    p.add_argument("--out-fig", default="paper/fig_axis_skills.png")
    main(p.parse_args())
