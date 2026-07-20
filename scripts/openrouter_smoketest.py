"""
OpenRouter connectivity smoke test.

Verifies that each target model ID actually resolves on OpenRouter and returns a
completion, BEFORE spending budget on the full FinInteract harness. For each model:
  1. checks presence in OpenRouter's /models catalog (exact-id match)
  2. fires one live reasoning-enabled chat call (mirrors the user's snippet)
  3. reports OK/FAIL, latency, token usage, and any error message

Usage:
    python scripts/openrouter_smoketest.py --config configs/openrouter.json
    python scripts/openrouter_smoketest.py --config configs/openrouter.json --models anthropic/claude-sonnet-5
"""
import argparse
import json
import sys
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    sys.exit("pip install openai")

import urllib.request


def load_catalog(base_url: str, api_key: str) -> dict:
    """Fetch OpenRouter's model catalog: id -> pricing dict (prompt/completion $/token)."""
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
    except Exception as e:
        print(f"  [catalog] could not fetch /models: {e}", file=sys.stderr)
        return {}
    out = {}
    for m in data.get("data", []):
        out[m.get("id")] = m.get("pricing", {})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/openrouter.json")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Override the model list from the config file.")
    ap.add_argument("--prompt", default="How many r's are in the word 'strawberry'? Answer in one short sentence.")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    base_url = cfg["base_url"]
    api_key = cfg["api_key"]
    models = args.models or cfg.get("models", [])

    client = OpenAI(base_url=base_url, api_key=api_key)
    catalog = load_catalog(base_url, api_key)
    print(f"OpenRouter catalog: {len(catalog)} models visible\n")

    rows = []
    for model in models:
        in_catalog = model in catalog
        pricing = catalog.get(model, {})
        rec = {"model": model, "in_catalog": in_catalog, "pricing": pricing,
               "ok": False, "latency_s": None, "usage": None, "reply": "", "error": ""}
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": args.prompt}],
                max_tokens=2000,
                extra_body={"reasoning": {"enabled": True}},
            )
            rec["latency_s"] = round(time.time() - t0, 2)
            msg = resp.choices[0].message
            rec["reply"] = (msg.content or "").strip()[:160]
            rec["ok"] = bool(rec["reply"])
            if getattr(resp, "usage", None):
                rec["usage"] = {
                    "prompt": resp.usage.prompt_tokens,
                    "completion": resp.usage.completion_tokens,
                    "total": resp.usage.total_tokens,
                }
            rec["has_reasoning_details"] = bool(getattr(msg, "reasoning_details", None))
        except Exception as e:
            rec["latency_s"] = round(time.time() - t0, 2)
            rec["error"] = f"{type(e).__name__}: {e}"
        rows.append(rec)

        status = "OK  " if rec["ok"] else "FAIL"
        cat = "catalog:Y" if in_catalog else "catalog:N"
        print(f"[{status}] {model:<32} {cat}  {rec['latency_s']}s")
        if rec["usage"]:
            pp = float(pricing.get("prompt", 0) or 0)
            cp = float(pricing.get("completion", 0) or 0)
            print(f"        usage={rec['usage']}  price/1M in=${pp*1e6:.3f} out=${cp*1e6:.3f}")
        if rec["reply"]:
            print(f"        reply: {rec['reply']}")
        if rec["error"]:
            print(f"        error: {rec['error']}")
        print()

    ok = [r["model"] for r in rows if r["ok"]]
    bad = [r["model"] for r in rows if not r["ok"]]
    print("=" * 60)
    print(f"WORKING ({len(ok)}): {', '.join(ok) if ok else '(none)'}")
    print(f"FAILED  ({len(bad)}): {', '.join(bad) if bad else '(none)'}")

    out = Path("data/results/openrouter_smoketest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDetails written to {out}")


if __name__ == "__main__":
    main()
