#!/usr/bin/env bash
set -uo pipefail

export VLLM_USE_V1=0
export NCCL_ALGO=Ring
export VLLM_PORT=12351
export VLLM_USE_POD_ATTN=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=INFO
export HF_ENDPOINT=https://hf-mirror.com
export VLLM_ALLOW_DEPRECATED_BLOCK_MANAGER_V1=1
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
export CSET_DIR=/jizhicfs/marvinhjia/sca/yilin/vllm-dev/csrc/c_set

TP=4
PORT=8048
GPU="0,1,2,3"
BLOCK_SIZE=64
DTYPE="float16"
GPU_MEM_UTIL=0.8
MAX_MODEL_LEN=450000
MAX_BATCHED_TOKENS=8192
MODEL="meta-llama/Llama-3.1-70B-Instruct"
PY_VLLM="/jizhicfs/marvinhjia/miniconda3/envs/chenyilin-vllm/bin/python3"

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

BASE_DIR="/jizhicfs/marvinhjia/sca/yilin/vllm-dev/examples/test/e2e-exp/correct_results"
EXP_DIR="${BASE_DIR}/300_${DATASET}_${VLLM_EVICTOR_POLICY}_${MODEL_SIZE}_overall_qps${QPS}_frac${FRAC}_CV0.25_gamma"
mkdir -p "${EXP_DIR}"

SCHED_LOG="${EXP_DIR}/scheduler.log"
OUTPUT_JSONL="${EXP_DIR}/output.jsonl"
SERVER_LOG="${EXP_DIR}/server_output.log"

export USE_MINEXP
export MINEXP_A
export MINEXP_B
export MINEXP_BETA
export MINEXP_ALPHA
export VLLM_EVICTOR_POLICY
export SCHEDULE_LOG_PATH="${SCHED_LOG}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PY_VLLM}" \
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
# SERVER_PGID=$(ps -o pgid= "${SERVER_PID}" | tr -d ' ' || echo "")
# export SERVER_PGID

cleanup() {
  set +e 
  if [[ -n "${SERVER_PID:-}" ]]; then
    echo "[CLEANUP] kill vLLM server pid=${SERVER_PID}"
    kill -TERM "${SERVER_PID}" 2>/dev/null
    sleep 5
    kill -KILL "${SERVER_PID}" 2>/dev/null
  fi

  ps -ef | grep -E 'vllm.entrypoints.openai.api_server' | grep -v grep | awk '{print $2}' | xargs -r kill -9
  ps -ef | grep -E 'multiprocessing.spawn|multiprocessing.resource_tracker' | grep -v grep | awk '{print $2}' | xargs -r kill -9

  sleep 2
  echo "[CLEANUP] cleanup finished"
}
trap cleanup EXIT

# Wait for server to be ready
sleep 320
wait_for_server "${PORT}"
# Run the end to end interface
python exp_interface.py \
    --qps "${QPS}" \
    --frac "${FRAC}" \
    --dataset "${DATASET}" \
    --model_size "${MODEL_SIZE}" \
    --output_file "${OUTPUT_JSONL}"

echo "[DONE] experiment finished in ${EXP_DIR}"