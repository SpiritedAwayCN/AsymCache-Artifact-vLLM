#!/usr/bin/env bash
set -euo pipefail

export VLLM_USE_V1=0
export NCCL_ALGO=Ring
export VLLM_PORT=17272 # check 272->longbench_model_id
export VLLM_USE_POD_ATTN=1
# export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=INFO
export HF_ENDPOINT=https://hf-mirror.com
export VLLM_ALLOW_DEPRECATED_BLOCK_MANAGER_V1=1
export VLLM_HOST_IP=${VLLM_HOST_IP:-127.0.0.1}
# export HF_HOME=/jizhicfs/marvinhjia/sca/models/huggingface
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRODUCE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VLLM_ROOT="$(cd "${REPRODUCE_DIR}/../../../.." && pwd)"
RESULTS_ROOT=${RESULTS_ROOT:-$(cd "${VLLM_ROOT}/.." && pwd)/results}
export CSET_DIR=${CSET_DIR:-${VLLM_ROOT}/csrc/c_set}
export USE_CONTINUUM=0 DISABLE_MULTIPLIER=1
# export CSET_DIR=/jizhicfs/marvinhjia/sca/yilin/vllm-dev/csrc/c_set
# export HUGGINGFACE_HUB_CACHE=/jizhicfs/marvinhjia/sca/models/huggingface/hub
# export VLLM_ONLY_KEEP_SEG2=1 # check

# scheduler
export VLLM_SIMPLE_SHED_ALGO=0
export VLLM_SIMPLE_SHED_ALGO_MAX_CHUNK_SIZE=${VLLM_SIMPLE_SHED_ALGO_MAX_CHUNK_SIZE:-2048}

TP=4 # check
PORT=${PORT:-5272}
GPU=${GPU:-0,1,2,3}
BLOCK_SIZE=64
DTYPE="float16"
GPU_MEM_UTIL=0.8
MAX_MODEL_LEN=450000
MAX_BATCHED_TOKENS=8192
# PY_VLLM="/jizhicfs/marvinhjia/miniconda3/envs/chenyilin-vllm/bin/python3"
PY_VLLM=${PY_VLLM:-${PYTHON_BIN:-python3}}
# MODEL="/jizhicfs/marvinhjia/sca/models/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b"
MODEL=${MODEL_70B_PATH:-meta-llama/Llama-3.1-70B-Instruct}
# check
wait_for_server() {
  local port=$1
  echo "Waiting for server to start on port ${port}..."
  timeout 1200 bash -c "
    until curl -s ${VLLM_HOST_IP}:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

P=""
QPS=""
GRAD=""
USE_MINEXP=2

MINEXP_A=0
MINEXP_B=""
MINEXP_BETA=""
MINEXP_ALPHA=""

FRAC=""
DATASET=""
MODEL_SIZE=""
VLLM_EVICTOR_POLICY="mwf"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --qps)            QPS="$2"; shift 2;;
    --p)              P="$2"; shift 2;;
    --grad)           GRAD="$2"; shift 2;;
    --minexp-b)       MINEXP_B="$2"; shift 2;;
    --minexp-alpha)   MINEXP_ALPHA="$2"; shift 2;;
    --minexp-beta)    MINEXP_BETA="$2"; shift 2;;
    --dataset)        DATASET="$2"; shift 2;;
    --model-size)     MODEL_SIZE="$2"; shift 2;;
    --frac)           FRAC="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "${QPS}" || -z "${P}" || -z "${GRAD}" || -z "${MINEXP_B}" || -z "${MINEXP_ALPHA}" || -z "${MINEXP_BETA}" || -z "${DATASET}" || -z "${MODEL_SIZE}" || -z "${FRAC}" ]]; then
    echo "Missing required arguments"
    exit 1
fi

# BASE_DIR="/jizhicfs/marvinhjia/sca/chenyilei/correct_results/only_keep_seg2"
BASE_DIR="${RESULTS_ROOT}/e2e/asymcache_no_scheduler"
EXP_DIR="${BASE_DIR}/300_${DATASET}_${VLLM_EVICTOR_POLICY}_${MODEL_SIZE}_overall_qps${QPS}_frac${FRAC}_CV0.25_gamma"
mkdir -p "${EXP_DIR}"

SCHED_LOG="${EXP_DIR}/scheduler.log"
SERVER_LOG="${EXP_DIR}/server_output.log"
OUTPUT_JSONL="${EXP_DIR}/LongBench_simulator_minexp_300_overall_qps${QPS}_frac${FRAC}_CV0.25_P99_grad40_gamma_${MODEL_SIZE}.jsonl"

export USE_MINEXP
export MINEXP_A
export MINEXP_B
export MINEXP_BETA
export MINEXP_ALPHA
export VLLM_EVICTOR_POLICY
export SCHEDULE_LOG_PATH="${SCHED_LOG}"
export REPLAY_API_URL="http://${VLLM_HOST_IP}:${PORT}/v1/completions"
export REPLAY_MODEL="${MODEL}"

CUDA_VISIBLE_DEVICES="${GPU}" setsid "${PY_VLLM}" \
    -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    -tp "${TP}" \
    --port "${PORT}" \
    --block-size "${BLOCK_SIZE}" \
    --dtype "${DTYPE}" \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max_num_batched_tokens "${MAX_BATCHED_TOKENS}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    > "${SERVER_LOG}" 2>&1 &

SERVER_PID=$!
SERVER_PGID=$(ps -o pgid= -p "${SERVER_PID}" | tr -d ' ' || echo "")
SCRIPT_PGID=$(ps -o pgid= -p "$$" | tr -d ' ' || echo "")
export SERVER_PID SERVER_PGID SCRIPT_PGID
echo "[SERVER] pid=${SERVER_PID} pgid=${SERVER_PGID:-<empty>} script_pgid=${SCRIPT_PGID:-<empty>}"
if [[ -z "${SERVER_PGID:-}" || "${SERVER_PGID}" == "${SCRIPT_PGID}" ]]; then
  echo "[WARN] unsafe SERVER_PGID=${SERVER_PGID:-<empty>}; will fall back to SERVER_PID cleanup"
  SERVER_PGID=""
fi

cleanup() {
  set +e
  trap - EXIT INT TERM
  if [[ -n "${SERVER_PGID:-}" ]]; then
    echo "[CLEANUP] kill process group PGID=${SERVER_PGID}"
    kill -TERM -- -"${SERVER_PGID}" 2>/dev/null
    sleep 6
    kill -KILL -- -"${SERVER_PGID}" 2>/dev/null
  elif [[ -n "${SERVER_PID:-}" ]]; then
    echo "[CLEANUP] kill vLLM server pid=${SERVER_PID}"
    kill -TERM -- "${SERVER_PID}" 2>/dev/null
    sleep 6
    kill -KILL -- "${SERVER_PID}" 2>/dev/null
  else
    echo "[CLEANUP] SERVER_PID not set, skip cleanup"
  fi
  echo "[CLEANUP] cleanup finished"
}
trap cleanup EXIT INT TERM

# Wait for server to be ready
sleep "${STARTUP_SLEEP:-550}"
wait_for_server "${PORT}"
# Run the end to end interface
# check
"${PY_VLLM}" exp_interface2.py \
    --qps "${QPS}" \
    --frac "${FRAC}" \
    --dataset "${DATASET}" \
    --model_size "${MODEL_SIZE}" \
    --output_file "${OUTPUT_JSONL}"
echo "[DONE] experiment finished in ${EXP_DIR}"
