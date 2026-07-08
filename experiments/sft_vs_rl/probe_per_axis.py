"""Per-axis ONE-VS-REST linear decodability of the operative ambiguity axis.

For a given model, extract per-layer last-token hidden states over the bare questions of
data/final/fininteract_v1.jsonl, then for each axis A in {entity, metric, recognition,
temporal} train a per-layer one-vs-rest logistic-regression probe ("primary axis == A vs
not") and report peak AUROC (5-fold CV, class_weight balanced). AUROC (not accuracy)
because the classes are imbalanced (recognition n=9, temporal n=7 of 173).

Mechanistic definition of "learnable": an axis is learnable iff it is linearly decodable
(peak AUROC >> 0.5). Usage mirrors probe_sft_vs_base.py:
  python probe_per_axis.py --base-model Qwen/Qwen3-4B-Instruct-2507 --out probe_peraxis_4b.json
  python probe_per_axis.py --base-model <base> --adapter outputs/coevolve/sft_entity --out ...
"""
from __future__ import annotations
import argparse, json
import numpy as np

AXES = ("entity_scope", "metric_definition", "recognition_policy", "temporal_scope")


def build_bare_prompt(tok, question):
    return tok.apply_chat_template([{"role": "user", "content": question}],
                                   tokenize=False, add_generation_prompt=True)


def extract(model, tok, prompts, max_len):
    import torch
    feats = []
    model.eval()
    for text in prompts:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        hs = out.hidden_states[1:]
        feats.append(torch.stack([h[0, -1] for h in hs]).float().cpu().numpy())
    return np.stack(feats)  # [N, L, H]


def load_model(base_model, adapter):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16,
                                             device_map="auto", trust_remote_code=True)
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter)
    return m


def probe_ovr_per_layer(X, y, seed):
    """Per-layer one-vs-rest AUROC (5-fold CV, balanced)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    L = X.shape[1]
    npos = int(y.sum())
    nsplits = min(5, npos) if npos >= 2 else 2
    skf = StratifiedKFold(n_splits=nsplits, shuffle=True, random_state=seed)
    aurocs = []
    for li in range(L):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
        try:
            s = cross_val_score(clf, X[:, li, :], y, cv=skf, scoring="roc_auc").mean()
        except Exception:
            s = 0.5
        aurocs.append(float(s))
    return aurocs


def main(a):
    import torch, gc
    from transformers import AutoTokenizer
    rows = [json.loads(l) for l in open(a.instances)]
    prim = [(r.get("axes") or [None])[0] for r in rows]
    tok = AutoTokenizer.from_pretrained(a.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = [build_bare_prompt(tok, r["question"]) for r in rows]
    print(f"loading {a.base_model} (adapter={a.adapter}) ...", flush=True)
    model = load_model(a.base_model, a.adapter)
    print("extracting activations for", len(rows), "instances ...", flush=True)
    X = extract(model, tok, prompts, a.max_len)          # [N, L, H]
    del model; gc.collect(); torch.cuda.empty_cache()

    result = {"model": a.base_model, "adapter": a.adapter, "n": len(rows),
              "n_layers": X.shape[1], "per_axis": {}}
    for ax in AXES:
        y = np.array([1 if p == ax else 0 for p in prim])
        npos = int(y.sum())
        if npos < 2:
            print(f"  {ax}: n_pos={npos} too few, skipping"); continue
        aurocs = probe_ovr_per_layer(X, y, a.seed)
        peak = int(np.argmax(aurocs))
        result["per_axis"][ax] = {"n_pos": npos, "peak_auroc": aurocs[peak],
                                  "peak_layer": peak, "per_layer_auroc": aurocs}
        print(f"  {ax:20s} n_pos={npos:3d}  peak AUROC={aurocs[peak]:.3f} @ L{peak}/{X.shape[1]-1}", flush=True)
    from pathlib import Path
    Path(a.out).write_text(json.dumps(result, indent=2))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--instances", default="data/final/fininteract_v1.jsonl")
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="probe_peraxis.json")
    main(p.parse_args())
