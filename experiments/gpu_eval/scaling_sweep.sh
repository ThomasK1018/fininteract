#!/usr/bin/env bash
# Model-size scaling sweep: the SAME family across sizes, so the only variable is
# scale. Produces the "scale lifts the oracle ceiling but NOT self-elicitation"
# figure (AxisHit / +interact accuracy flat vs. ceiling rising with size).
#
# Serves each model in turn, waits until healthy, runs all modes + ceiling, then
# tears the server down before the next size. Run from the repo root with
# OPENAI_API_KEY set. Override the roster by editing SWEEP below.
#
#   OPENAI_API_KEY=sk-... ./experiments/gpu_eval/scaling_sweep.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8000}"

# label            HF model id                  TP   (dense Qwen3, fits 8xA100-40)
SWEEP=(
  "qwen3-0.6b      Qwen/Qwen3-0.6B              1"
  "qwen3-1.7b      Qwen/Qwen3-1.7B              1"
  "qwen3-4b        Qwen/Qwen3-4B                1"
  "qwen3-8b        Qwen/Qwen3-8B                1"
  "qwen3-14b       Qwen/Qwen3-14B               2"
  "qwen3-32b       Qwen/Qwen3-32B               4"
)

wait_healthy() {  # poll the OpenAI-compatible /models endpoint
  for _ in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then return 0; fi
    sleep 10
  done
  echo "server never became healthy on :$PORT" >&2; return 1
}

for row in "${SWEEP[@]}"; do
  read -r LABEL MODEL TP <<<"$row"
  echo "############ $LABEL ($MODEL, TP=$TP) ############"
  PORT="$PORT" bash "$HERE/serve.sh" "$LABEL" "$MODEL" "$TP" >"/tmp/vllm_${LABEL}.log" 2>&1 &
  SRV=$!
  if wait_healthy; then
    PORT="$PORT" bash "$HERE/eval_model.sh" "$LABEL" "$PORT" || echo "EVAL FAILED for $LABEL"
  fi
  kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
  sleep 5   # let VRAM free before the next size
done
echo "sweep complete -> data/results/eval_open_*.jsonl + eval_ceiling_*.jsonl"
