#!/usr/bin/env bash
# Drive a served model through all evaluation modes + the context-oracle ceiling.
# Requires: a vLLM server up (serve.sh) AND OPENAI_API_KEY set (sim + grader).
#
# Usage:
#   OPENAI_API_KEY=sk-... ./eval_model.sh <LABEL> [PORT]
#   ./eval_model.sh qwen3-8b
#
# Produces (keyed by LABEL):
#   data/results/eval_open_<LABEL>.jsonl       (answer-only / +search / +interact)
#   data/results/eval_open_<LABEL>.summary.json
#   data/results/eval_ceiling_<LABEL>.jsonl    (context-oracle = latent ceiling)
set -euo pipefail
LABEL="${1:?need the served LABEL, e.g. qwen3-8b}"
PORT="${2:-8000}"
URL="http://localhost:${PORT}/v1"
DATA="${INSTANCES:-data/final/fininteract_v1.jsonl}"   # frozen eval snapshot (N=173)
PY="${PYTHON:-python}"

export AGENT_BASE_URL="$URL" AGENT_API_KEY="EMPTY"
mkdir -p data/results

echo "== [$LABEL] answer-only / +search / +interact =="
$PY scripts/evaluate.py \
  --instances "$DATA" \
  --models "$LABEL" \
  --modes answer-only answer+search answer+search+interact \
  --agent-base-url "$URL" \
  --out "data/results/eval_open_${LABEL}.jsonl" \
  --summary "data/results/eval_open_${LABEL}.summary.json"

echo "== [$LABEL] context-oracle ceiling =="
$PY scripts/eval_context_ceiling.py \
  --instances "$DATA" \
  --model "$LABEL" \
  --base-url "$URL" \
  --out "data/results/eval_ceiling_${LABEL}.jsonl"

echo "== [$LABEL] done. Roll into the tables with analyze_breakdowns.py =="
