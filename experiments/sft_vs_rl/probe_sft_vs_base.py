"""Experiment 3 — does SFT change the AXIS REPRESENTATION or only the output policy?

For the bare ambiguous question, extract per-layer last-token (or mean-pooled)
activations from (a) the base model and (b) base + SFT adapter, then train a
per-layer linear probe to decode the ambiguity axis (entity_scope vs
metric_definition — the two well-populated classes, matching analyze_probes.py).

Interpretation:
  SFT peak  >> base peak  -> SFT SHARPENED the representation  (capability gain)
  SFT peak  ~= base peak  -> representation unchanged; SFT rewired only the
                            output policy (formatting-like) — the "represents-but-
                            doesn't-act -> now-acts" story is policy, not capability.

GPU required (transformers + torch [+ peft for --adapter]). Probe is CPU sklearn.

Usage:
  python probe_sft_vs_base.py \
    --instances data/final/fininteract_v1.jsonl \
    --base-model Qwen/Qwen3-4B-Instruct-2507 \
    --adapter   outputs/sft \
    --rep last --out probe_sft_vs_base.json --fig probe_sft_vs_base.png
  # or a fully-merged SFT checkpoint instead of a LoRA adapter:
  python probe_sft_vs_base.py ... --sft-model outputs/sft_merged
"""
import argparse, json
from pathlib import Path
import numpy as np

AXES = ("entity_scope", "metric_definition")  # the two probeable classes


def build_bare_prompt(tok, question: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True)


def extract(model, tok, prompts, rep, max_len):
    """Return [N, n_layers, hidden] per-prompt activations."""
    import torch
    feats = []
    model.eval()
    for text in prompts:
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=max_len).to(model.device)
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        hs = out.hidden_states[1:]  # drop embedding layer -> one per transformer layer
        if rep == "mean":
            layer_vecs = [h[0].mean(0) for h in hs]
        else:  # last token
            layer_vecs = [h[0, -1] for h in hs]
        feats.append(torch.stack(layer_vecs).float().cpu().numpy())
    return np.stack(feats)  # [N, L, H]


def load_model(base_model, adapter, sft_model):
    import torch
    from transformers import AutoModelForCausalLM
    if sft_model:
        m = AutoModelForCausalLM.from_pretrained(
            sft_model, torch_dtype=torch.float16, device_map="auto")
        return m
    m = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="auto")
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter)
    return m


def probe_per_layer(X, y, seed):
    """Per-layer 5-fold CV accuracy of a logistic-regression axis probe."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    L = X.shape[1]
    accs = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for li in range(L):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0))
        accs.append(float(cross_val_score(clf, X[:, li, :], y, cv=skf,
                                          scoring="accuracy").mean()))
    return accs


def main(a):
    import torch, gc
    from transformers import AutoTokenizer

    rows = [json.loads(l) for l in open(a.instances)]
    rows = [r for r in rows if (r.get("axes") or [None])[0] in AXES]
    y = np.array([0 if (r["axes"][0] == AXES[0]) else 1 for r in rows])
    baseline = max(np.mean(y == 0), np.mean(y == 1))
    print(f"probe set: {len(rows)} instances "
          f"({int((y==0).sum())} {AXES[0]} / {int((y==1).sum())} {AXES[1]}); "
          f"majority baseline = {baseline:.3f}")

    tok = AutoTokenizer.from_pretrained(a.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = [build_bare_prompt(tok, r["question"]) for r in rows]

    result = {"axes": AXES, "n": len(rows), "rep": a.rep,
              "majority_baseline": float(baseline)}

    for tag, adapter, sft_model in [("base", None, None),
                                    ("sft", a.adapter, a.sft_model)]:
        if tag == "sft" and not (adapter or sft_model):
            print("no --adapter/--sft-model given; skipping SFT arm"); continue
        print(f"\n[{tag}] loading model ...")
        model = load_model(a.base_model, adapter, sft_model)
        X = extract(model, tok, prompts, a.rep, a.max_len)
        accs = probe_per_layer(X, y, a.seed)
        peak = int(np.argmax(accs))
        result[tag] = {"per_layer_acc": accs, "peak_layer": peak,
                       "peak_acc": accs[peak], "n_layers": len(accs)}
        print(f"[{tag}] peak axis-decoding acc = {accs[peak]:.3f} at layer "
              f"{peak}/{len(accs)-1}  (baseline {baseline:.3f})")
        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "base" in result and "sft" in result:
        d = result["sft"]["peak_acc"] - result["base"]["peak_acc"]
        result["peak_delta_sft_minus_base"] = d
        verdict = ("SFT SHARPENED the representation (capability gain)" if d > 0.05
                   else "representation ~unchanged; SFT rewired the output policy "
                        "(formatting-like)" if abs(d) <= 0.05
                   else "SFT DEGRADED the decodable axis signal")
        print(f"\nPEAK DELTA (sft - base) = {d:+.3f}  ->  {verdict}")
        result["verdict"] = verdict

    Path(a.out).write_text(json.dumps(result, indent=2))
    print("wrote", a.out)

    if a.fig and "base" in result and "sft" in result:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"figure.dpi": 160, "font.size": 10,
                             "axes.grid": True, "grid.alpha": 0.3})
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        for tag, c in [("base", "C0"), ("sft", "C3")]:
            accs = result[tag]["per_layer_acc"]
            x = np.linspace(0, 1, len(accs))
            ax.plot(x, accs, color=c, lw=1.8,
                    label=f"{tag} (peak {result[tag]['peak_acc']:.2f})")
        ax.axhline(baseline, color="gray", ls="--", lw=1,
                   label=f"majority baseline ({baseline:.2f})")
        ax.set_xlabel("depth (fraction of layers)")
        ax.set_ylabel("entity-vs-metric probe accuracy")
        ax.set_title("Axis representation: base vs SFT")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout(); fig.savefig(a.fig, bbox_inches="tight")
        print("wrote", a.fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--instances", default="data/final/fininteract_v1.jsonl")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", default=None, help="LoRA adapter dir (on top of --base-model)")
    p.add_argument("--sft-model", default=None, help="fully-merged SFT checkpoint (alt to --adapter)")
    p.add_argument("--rep", choices=["last", "mean"], default="last")
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="probe_sft_vs_base.json")
    p.add_argument("--fig", default="probe_sft_vs_base.png")
    main(p.parse_args())
