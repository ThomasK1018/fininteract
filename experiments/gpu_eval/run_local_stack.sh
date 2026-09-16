#!/usr/bin/env bash
# Fully API-free FinInteract replication on ONE GPU (reviewer request: reproducibility
# without API budget, and removing the GPT-5 family from every evaluation-time role).
#
#   agent      : Qwen3-8B (bf16, thinking off)            served as "qwen3-8b"   on :18000
#   simulator  : Qwen2.5-14B-Instruct (fp8 dynamic quant) served as "judge"      on :18001
#   grader     : same "judge" server (answer grading, default-capture, axis judge)
#
# Runs, in order, writing data/results/eval_local_*.jsonl:
#   1. answer-only / answer+search / answer+search+interact (N=173)     -> eval_local_qwen3-8b.jsonl
#   2. context-oracle ceiling (local grader)                             -> eval_local_ceiling_qwen3-8b.jsonl
#   3. unambiguous-control, generic-structured, axis-aware, always-ask   -> eval_local_policies_qwen3-8b.jsonl
#   4. free-form simulator, +interact                                    -> eval_local_freeform_qwen3-8b.jsonl
#   5. ask-once (forced 1, max 1) yes/no and free-form                   -> eval_local_askonce_{yn,ff}_qwen3-8b.jsonl
#   6. local-grader agreement with the stored GPT-4o-mini grades         -> data/results/local_grader_agreement.json
#
# Usage (on the GPU box, from the repo root, under nohup):
#   nohup bash experiments/gpu_eval/run_local_stack.sh > logs/run_local_stack.log 2>&1 &
# Env: PY (python with vllm+openai), JUDGE_MODEL_PATH, AGENT_MODEL_PATH, JUDGE_UTIL, AGENT_UTIL,
#      INSTALL_LOG (if set, waits until it contains FIVENV_DONE before starting).
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY="${PY:-$HOME/fivenv/bin/python}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-$HOME/models/Qwen2.5-14B-Instruct}"
AGENT_MODEL_PATH="${AGENT_MODEL_PATH:-$HOME/models/Qwen3-8B}"
JUDGE_UTIL="${JUDGE_UTIL:-0.17}"
AGENT_UTIL="${AGENT_UTIL:-0.22}"
JPORT=18001; APORT=18000
AGENT_LABEL=qwen3-8b
mkdir -p logs data/results
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
# Driver 570 (CUDA 12.8) with a CUDA-13 torch/vLLM build: use the forward-compat libs if present.
if [ -d "$HOME/cuda13compat" ]; then
  export LD_LIBRARY_PATH="$HOME/cuda13compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export TRITON_LIBCUDA_PATH="$HOME/cuda13compat"
fi

if [ -n "${INSTALL_LOG:-}" ]; then
  echo "[$(date)] waiting for $INSTALL_LOG to report FIVENV_DONE"
  until grep -q FIVENV_DONE "$INSTALL_LOG" 2>/dev/null; do sleep 60; done
fi
"$PY" -c "import vllm, openai; print('vllm', vllm.__version__)" || { echo "python env broken"; exit 2; }

serve() { # label model port util extra...
  local label="$1" model="$2" port="$3" util="$4"; shift 4
  if curl -s "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then echo "port $port already serving"; return; fi
  for attempt in 1 2 3 4 5 6; do
    nohup "$PY" -m vllm.entrypoints.openai.api_server --model "$model" --served-model-name "$label" \
      --host 127.0.0.1 --port "$port" --gpu-memory-utilization "$util" --enforce-eager \ "$@" > "logs/serve_${label}.log" 2>&1 &
    echo $! > "logs/serve_${label}.pid"
    for i in $(seq 1 120); do
      sleep 10
      curl -s "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && { echo "[$(date)] $label up on :$port"; return 0; }
      kill -0 "$(cat logs/serve_${label}.pid)" 2>/dev/null || break
    done
    echo "[$(date)] $label failed to start (attempt $attempt); tail:"; tail -5 "logs/serve_${label}.log"
    sleep 120   # VRAM race with the neighbour's job: just retry
  done
  return 1
}

serve judge "$JUDGE_MODEL_PATH" $JPORT "$JUDGE_UTIL" --quantization fp8 --max-model-len 8192 || exit 3
serve "$AGENT_LABEL" "$AGENT_MODEL_PATH" $APORT "$AGENT_UTIL" --max-model-len 16384 || exit 3

JUDGE="--judge-base-url http://127.0.0.1:$JPORT/v1 --sim-model judge --grader-model judge"
AGENT="--agent-base-url http://127.0.0.1:$APORT/v1 --agent-thinking off"
DATA="--instances data/final/fininteract_v1.jsonl --passage-file data/sources/passages.jsonl"
EV="$PY scripts/evaluate.py $DATA --models $AGENT_LABEL $AGENT $JUDGE --resume"

echo "== 1. main modes =="
$EV --modes answer-only answer+search answer+search+interact \
  --out data/results/eval_local_${AGENT_LABEL}.jsonl --summary data/results/eval_local_${AGENT_LABEL}.summary.json
echo "== 2. ceiling =="
$PY scripts/eval_context_ceiling.py --instances data/final/fininteract_v1.jsonl --model $AGENT_LABEL \
  --base-url http://127.0.0.1:$APORT/v1 --judge-base-url http://127.0.0.1:$JPORT/v1 --grader-model judge \
  --out data/results/eval_local_ceiling_${AGENT_LABEL}.jsonl
echo "== 3. policies + unambiguous control =="
$EV --modes unambiguous-control generic-structured axis-aware always-ask \
  --out data/results/eval_local_policies_${AGENT_LABEL}.jsonl --summary data/results/eval_local_policies_${AGENT_LABEL}.summary.json
echo "== 4. free-form simulator =="
$EV --modes answer+search+interact --user-sim freeform \
  --out data/results/eval_local_freeform_${AGENT_LABEL}.jsonl --summary data/results/eval_local_freeform_${AGENT_LABEL}.summary.json
echo "== 5. ask-once =="
$EV --modes answer+search+interact --forced-interact 1 --max-interact 1 \
  --out data/results/eval_local_askonce_yn_${AGENT_LABEL}.jsonl --summary data/results/eval_local_askonce_yn_${AGENT_LABEL}.summary.json
$EV --modes answer+search+interact --forced-interact 1 --max-interact 1 --user-sim freeform \
  --out data/results/eval_local_askonce_ff_${AGENT_LABEL}.jsonl --summary data/results/eval_local_askonce_ff_${AGENT_LABEL}.summary.json
echo "== 6. local grader agreement =="
$PY scripts/validate_local_grader.py --judge-base-url http://127.0.0.1:$JPORT/v1 --grader-model judge \
  --out data/results/local_grader_agreement.json
echo "[$(date)] ALL DONE"
