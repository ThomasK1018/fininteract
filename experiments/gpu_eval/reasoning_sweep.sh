#!/usr/bin/env bash
# Mechanistic depth (#4a): does test-time REASONING help RECOGNIZE ambiguity, or
# only compute answers? Matched-size reasoning vs non-reasoning pairs through the
# identical harness -- no code change, the model is the only variable. Expectation
# (supports the thesis): reasoning lifts the oracle ceiling but barely moves
# AxisHit / +interact accuracy, because the bottleneck is eliciting C, not
# computing once C is known.
#
#   OPENAI_API_KEY=sk-... ./experiments/gpu_eval/reasoning_sweep.sh
#
# Matched 32B pairs (all fit on 8xA100-40, TP=4):
#   QwQ-32B (reasoning)              vs Qwen2.5-32B-Instruct (non-reasoning)
#   DeepSeek-R1-Distill-Qwen-32B     vs Qwen2.5-32B-Instruct (shared control)
# Optional same-weights toggle (advanced): serve Qwen3-32B twice with
# enable_thinking true/false via --reasoning-parser / a chat template; see RUNBOOK.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8000}"

SWEEP=(
  "qwen2.5-32b   Qwen/Qwen2.5-32B-Instruct          4"
  "qwq-32b       Qwen/QwQ-32B                       4"
  "r1-distill-32b deepseek-ai/DeepSeek-R1-Distill-Qwen-32B  4"
)

wait_healthy() {
  for _ in $(seq 1 120); do
    curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 && return 0
    sleep 10
  done
  echo "server never healthy on :$PORT" >&2; return 1
}

for row in "${SWEEP[@]}"; do
  read -r LABEL MODEL TP <<<"$row"
  echo "############ $LABEL ($MODEL, TP=$TP) ############"
  PORT="$PORT" bash "$HERE/serve.sh" "$LABEL" "$MODEL" "$TP" >"/tmp/vllm_${LABEL}.log" 2>&1 &
  SRV=$!
  wait_healthy && { PORT="$PORT" bash "$HERE/eval_model.sh" "$LABEL" "$PORT" || echo "EVAL FAILED $LABEL"; }
  kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
  sleep 5
done
echo "reasoning sweep complete."
