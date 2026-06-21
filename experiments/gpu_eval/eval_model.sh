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
# CRITICAL: search/oracle retrieval reads passage_text from this file. Without it
# evaluate.py falls back to context[:500] (= the disambiguator C, NO answer values),
# which silently zeroes +search/+interact accuracy and default-capture. This file
# covers all 173 instances and is the SAME source the OpenAI ladder used.
PASSAGES="${PASSAGE_FILE:-data/sources/passages.jsonl}"
PY="${PYTHON:-python}"

export AGENT_BASE_URL="$URL" AGENT_API_KEY="EMPTY"
mkdir -p data/results

echo "== [$LABEL] answer-only / +search / +interact (passages: $PASSAGES) =="
$PY scripts/evaluate.py \
  --instances "$DATA" \
  --passage-file "$PASSAGES" \
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
