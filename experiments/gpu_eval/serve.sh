#!/usr/bin/env bash
# Serve an open-weight model as an OpenAI-compatible AGENT endpoint (vLLM).
# The user-simulator (GPT-5) and grader (GPT-4o-mini) stay on the OpenAI API;
# only the agent-under-test runs here. Leave this running, then drive it with
# eval_model.sh / scaling_sweep.sh from another shell.
#
# Usage:
#   ./serve.sh <LABEL> <HF_MODEL_ID> [TP] [extra vllm flags...]
# Examples (8x A100-40GB, NVLink):
#   ./serve.sh qwen3-8b   Qwen/Qwen3-8B                 1
#   ./serve.sh qwen3-32b  Qwen/Qwen3-32B                4
#   ./serve.sh llama70b   meta-llama/Llama-3.3-70B-Instruct  8
#   ./serve.sh glm45air   zai-org/GLM-4.5-Air           8
#   ./serve.sh qwen3-235b Qwen/Qwen3-235B-A22B-Instruct-2507-AWQ  8 --quantization awq
#
# LABEL is the served-model-name; pass the SAME label to --models in evaluate.py
# so result files are keyed by model. Env: PORT (8000), MAXLEN (16384),
# GPU_UTIL (0.90).
set -euo pipefail
LABEL="${1:?need a LABEL, e.g. qwen3-8b}"
MODEL="${2:?need a HF model id, e.g. Qwen/Qwen3-8B}"
TP="${3:-1}"
shift $(( $# < 3 ? $# : 3 ))

exec vllm serve "$MODEL" \
  --served-model-name "$LABEL" \
  --tensor-parallel-size "$TP" \
  --port "${PORT:-8000}" \
  --max-model-len "${MAXLEN:-16384}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --disable-log-requests \
  "$@"
