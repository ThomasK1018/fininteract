#!/usr/bin/env bash
# Leak-free end-to-end evaluation of category-guided SFT on a LARGER backbone (reviewer
# request). Trains Qwen3-14B with LoRA on the committed category-guided demos (231 demos,
# zero overlap with the 173 benchmark items), merges, then evaluates BASE and SFT under the
# identical API-free stack (local Qwen2.5-14B simulator/grader, retrieval passage containing
# BOTH candidate spans, so accuracy cannot be read off the search result).
#
#   data/results/eval_local_qwen3-14b.jsonl         base, answer+search / answer+search+interact
#   data/results/eval_local_qwen3-14b-catsft.jsonl  category-guided SFT, same modes
#
# Usage (repo root, GPU box):  nohup bash experiments/gpu_eval/run_sft_leakfree.sh > logs/sft_leakfree.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY="${PY:-$HOME/fivenv/bin/python}"
BASE="${BASE_MODEL_PATH:-$HOME/models/Qwen3-14B}"
SFT="${SFT_MODEL_PATH:-$HOME/models/Qwen3-14B-catsft}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-$HOME/models/Qwen2.5-14B-Instruct}"
JPORT=18001; APORT=18000
mkdir -p logs data/results
export HF_HUB_OFFLINE=0 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PATH="$(dirname "$PY"):/usr/local/cuda/bin:$PATH"
if [ -d "$HOME/cuda13compat" ]; then
  export LD_LIBRARY_PATH="$HOME/cuda13compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; export TRITON_LIBCUDA_PATH="$HOME/cuda13compat"
fi
PYMM=$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
if [ -f "$HOME/pyinclude/usr/include/python${PYMM}/Python.h" ]; then
  export C_INCLUDE_PATH="$HOME/pyinclude/usr/include:$HOME/pyinclude/usr/include/python${PYMM}:$HOME/pyinclude/usr/include/x86_64-linux-gnu/python${PYMM}"; export CPATH="$C_INCLUDE_PATH"
fi

log() { echo "[$(date '+%F %T')] $*"; }
kill_port() { for p in $(pgrep -f "vllm.entrypoints.*--port $1"); do kill "$p" 2>/dev/null; done; sleep 5; }

serve() { # label model port util extra...
  local label="$1" model="$2" port="$3" util="$4"; shift 4
  if curl -s "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -q "\"$label\""; then log "$label already on :$port"; return 0; fi
  kill_port "$port"
  for attempt in 1 2 3 4 5 6; do
    nohup "$PY" -m vllm.entrypoints.openai.api_server --model "$model" --served-model-name "$label" \
      --host 127.0.0.1 --port "$port" --gpu-memory-utilization "$util" --enforce-eager "$@" > "logs/serve_${label}.log" 2>&1 &
    echo $! > "logs/serve_${label}.pid"
    for i in $(seq 1 120); do
      sleep 10
      curl -s "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && { log "$label up on :$port"; return 0; }
      kill -0 "$(cat logs/serve_${label}.pid)" 2>/dev/null || break
    done
    log "$label failed to start (attempt $attempt)"; tail -3 "logs/serve_${label}.log"; sleep 120
  done
  return 1
}

log "== 0. deps + base model =="
"$PY" -c "import peft" 2>/dev/null || "$PY" -m pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple peft accelerate
if [ ! -f "$BASE/config.json" ]; then
  "$PY" - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-14B", local_dir="$BASE", max_workers=8)
PYEOF
fi
ls "$BASE"/*.safetensors >/dev/null 2>&1 || { log "base model download failed"; exit 2; }

log "== 1. demos =="
cat data/coevolve/entity/demos/sft.jsonl data/coevolve/metric/demos/sft.jsonl data/coevolve/round0/demos/sft.jsonl > data/coevolve/catsft_demos.jsonl
wc -l data/coevolve/catsft_demos.jsonl

if [ ! -f "$SFT/config.json" ]; then
  log "== 2. train (free VRAM: stop the agent and judge servers for the duration) =="
  kill_port $APORT; kill_port $JPORT
  "$PY" experiments/gpu_eval/sft_lora_train.py --model "$BASE" --data data/coevolve/catsft_demos.jsonl \
      --output "$SFT" --epochs 3 --lr 2e-4 --r 16 --max-len 2048 || { log "training failed"; exit 3; }
fi

log "== 3. judge =="
serve judge "$JUDGE_MODEL_PATH" $JPORT 0.17 --quantization fp8 --max-model-len 8192 || exit 4
JUDGE="--judge-base-url http://127.0.0.1:$JPORT/v1 --sim-model judge --grader-model judge"
DATA="--instances data/final/fininteract_v1.jsonl --passage-file data/sources/passages.jsonl"

for pair in "qwen3-14b:$BASE" "qwen3-14b-catsft:$SFT"; do
  label="${pair%%:*}"; path="${pair#*:}"
  log "== 4. eval $label =="
  serve "$label" "$path" $APORT 0.22 --quantization fp8 --max-model-len 16384 || exit 5
  "$PY" scripts/evaluate.py $DATA --models "$label" --agent-base-url http://127.0.0.1:$APORT/v1 --agent-thinking off $JUDGE --resume \
      --modes answer+search answer+search+interact \
      --out "data/results/eval_local_${label}.jsonl" --summary "data/results/eval_local_${label}.summary.json"
  kill_port $APORT
done
log "ALL DONE"
