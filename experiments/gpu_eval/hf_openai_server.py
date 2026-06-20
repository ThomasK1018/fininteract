"""Minimal OpenAI-compatible chat server backed by HF transformers.

Why this exists: the FinInteract Step-1 eval expects an OpenAI-compatible AGENT
endpoint (serve.sh normally uses vLLM). But on this box vLLM cannot serve the
two A3B models we need:
  - driver 535 (CUDA 12.2) cannot run vLLM 0.23's cu130 torch, and
  - vLLM 0.23 has no `qwen3_5_moe` in its registry (Model B unsupported).
Transformers 5.10 (fininteract_venv) loads BOTH models fine on this driver, so
we wrap it in a tiny stdlib server. Generation is serialized (one model, one
GPU pipeline); evaluate.py drives the agent sequentially anyway.

Endpoints: GET /v1/models, POST /v1/chat/completions  (subset of the OpenAI API
that scripts/evaluate.py + scripts/eval_context_ceiling.py actually use).

Env:
  HF_MODEL_ID   required, e.g. Qwen/Qwen3-30B-A3B
  SERVED_NAME   model id reported back / expected by --models (default = HF_MODEL_ID)
  PORT          default 8000
  MAX_NEW       hard cap on max_new_tokens (default 1024)
  ENABLE_THINK  "1" to keep Qwen thinking traces (default off -> clean, fast)
"""
import json, os, re, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID    = os.environ["HF_MODEL_ID"]
SERVED_NAME = os.environ.get("SERVED_NAME", MODEL_ID)
PORT        = int(os.environ.get("PORT", "8000"))
MAX_NEW     = int(os.environ.get("MAX_NEW", "1024"))
ENABLE_THINK = os.environ.get("ENABLE_THINK", "0") == "1"

print(f"[server] loading {MODEL_ID} (served as '{SERVED_NAME}') on "
      f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} ...", flush=True)
_t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model.eval()
INPUT_DEV = next(model.parameters()).device
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
print(f"[server] READY in {time.time()-_t0:.0f}s | input_dev={INPUT_DEV} | "
      f"layers≈{getattr(model.config,'num_hidden_layers','?')} | thinking={'on' if ENABLE_THINK else 'off'}",
      flush=True)

_GEN_LOCK = threading.Lock()
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _apply_template(messages, think=None):
    """Tokenize chat messages. `think` (per-request) overrides the global default."""
    enable = ENABLE_THINK if think is None else bool(think)
    kw = dict(add_generation_prompt=True, return_tensors="pt")
    if not enable:
        try:
            return tok.apply_chat_template(messages, enable_thinking=False, **kw)
        except TypeError:
            pass  # template doesn't take the kwarg
    else:
        try:
            return tok.apply_chat_template(messages, enable_thinking=True, **kw)
        except TypeError:
            pass
    return tok.apply_chat_template(messages, **kw)


@torch.no_grad()
def _generate(messages, max_tokens, temperature, think=None):
    enc = _apply_template(messages, think=think)
    # transformers 5.x returns a BatchEncoding (dict); older returns a bare tensor.
    if hasattr(enc, "keys"):
        enc = {k: v.to(INPUT_DEV) for k, v in enc.items()}
        input_ids = enc["input_ids"]
    else:
        input_ids = enc.to(INPUT_DEV)
        enc = {"input_ids": input_ids}
    n_in = input_ids.shape[1]
    n = max(1, min(int(max_tokens or 512), MAX_NEW))
    gen = dict(max_new_tokens=n, pad_token_id=tok.pad_token_id)
    if temperature and temperature > 0:
        gen.update(do_sample=True, temperature=float(temperature), top_p=0.95)
    else:
        gen.update(do_sample=False)
    with _GEN_LOCK:
        out = model.generate(**enc, **gen)
    new = out[0, n_in:]
    text = tok.decode(new, skip_special_tokens=True)
    # Always strip reasoning traces so the post-think answer/action is what the eval parses.
    text = _THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip(), n_in, int(new.shape[0])


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/v1/models") or self.path.rstrip("/") == "/v1/models":
            self._send(200, {"object": "list",
                             "data": [{"id": SERVED_NAME, "object": "model",
                                       "created": 0, "owned_by": "local"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            messages = req.get("messages", [])
            max_tokens = req.get("max_tokens") or req.get("max_completion_tokens") or 512
            temperature = req.get("temperature", 0.0)
            # per-request thinking toggle (evaluate.py --agent-thinking sends this)
            think = (req.get("chat_template_kwargs") or {}).get("enable_thinking", None)
            text, n_in, n_out = _generate(messages, max_tokens, temperature, think=think)
        except Exception as e:  # never crash the eval; return empty content
            import traceback
            sys.stderr.write(f"[server] gen error: {type(e).__name__}: {e}\n{traceback.format_exc()}\n")
            sys.stderr.flush()
            text, n_in, n_out = "", 0, 0
        self._send(200, {
            "id": "chatcmpl-local", "object": "chat.completion", "created": int(time.time()),
            "model": SERVED_NAME,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": n_in, "completion_tokens": n_out,
                      "total_tokens": n_in + n_out},
        })


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[server] listening on :{PORT}", flush=True)
    srv.serve_forever()
