#!/usr/bin/env bash
set -uo pipefail

export VLLM_USE_V1=0
export NCCL_ALGO=Ring
export VLLM_PORT=12365
export VLLM_USE_POD_ATTN=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=INFO
export HF_ENDPOINT=https://hf-mirror.com
export VLLM_ALLOW_DEPRECATED_BLOCK_MANAGER_V1=1
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
export CSET_DIR=/jizhicfs/marvinhjia/sca/yilin/vllm-dev/csrc/c_set

TP=1
PORT=8065
GPU="7"
BLOCK_SIZE=64
DTYPE="float16"
GPU_MEM_UTIL=0.8
MAX_MODEL_LEN=450000
MAX_BATCHED_TOKENS=8192
MODEL="meta-llama/Llama-3.1-8B-Instruct"
PY_VLLM="/jizhicfs/marvinhjia/miniconda3/envs/chenyilin-vllm/bin/python3"

wait_for_server() {
  local port=$1
  echo "Waiting for server to start on port ${port}..."
  timeout 1200 bash -c "
    until curl -s ${VLLM_HOST_IP}:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

SET=""
QPS=""
GRAD=""
USE_MINEXP=2

MINEXP_A=0
MINEXP_B=""
MINEXP_BETA=""
MINEXP_ALPHA=""

FRAC=""
DATASET=""
VLLM_EVICTOR_POLICY="mwf"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --qps)            QPS="$2"; shift 2;;
    --set)            SET="$2"; shift 2;;
    --grad)           GRAD="$2"; shift 2;;
    --minexp-b)       MINEXP_B="$2"; shift 2;;
    --minexp-alpha)   MINEXP_ALPHA="$2"; shift 2;;
    --minexp-beta)    MINEXP_BETA="$2"; shift 2;;
    --dataset)        DATASET="$2"; shift 2;;
    --frac)           FRAC="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "${QPS}" || -z "${SET}" || -z "${GRAD}" || -z "${MINEXP_B}" || -z "${MINEXP_ALPHA}" || -z "${MINEXP_BETA}" || -z "${DATASET}" || -z "${FRAC}" ]]; then
    echo "Missing required arguments"
    exit 1
fi

BASE_DIR="/jizhicfs/marvinhjia/sca/yilin/vllm-dev/examples/test/e2e-exp/ablation_results"
EXP_DIR="${BASE_DIR}/LongBench_300_overall_qps${QPS}_frac${FRAC}_set${SET}_grad${GRAD}_gamma"
mkdir -p "${EXP_DIR}"

SCHED_LOG="${EXP_DIR}/scheduler.log"
OUTPUT_JSONL="${EXP_DIR}/LongBench_simulator_overall_qps${QPS}_frac${FRAC}_set${SET}_grad${GRAD}_gamma.jsonl"
SERVER_LOG="${EXP_DIR}/server_output.log"

export USE_MINEXP
export MINEXP_A
export MINEXP_B
export MINEXP_BETA
export MINEXP_ALPHA
export VLLM_EVICTOR_POLICY
export SCHEDULE_LOG_PATH="${SCHED_LOG}"

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
SERVER_PGID=$(ps -o pgid= "${SERVER_PID}" | tr -d ' ' || echo "")
export SERVER_PID SERVER_PGID

cleanup() {
  set +e
  if [[ -n "${SERVER_PGID:-}" ]]; then
    echo "[CLEANUP] kill process group PGID=${SERVER_PGID}"
    kill -TERM -"${SERVER_PGID}" 2>/dev/null
    sleep 6
    kill -KILL -"${SERVER_PGID}" 2>/dev/null
  else
    echo "[CLEANUP] SERVER_PGID not set, skip group kill"
  fi
  echo "[CLEANUP] cleanup finished"
}
trap cleanup EXIT

# Wait for server to be ready
sleep 150
wait_for_server "${PORT}"
# Run the end to end interface
python ab_interface5.py \
    --qps "${QPS}" \
    --frac "${FRAC}" \
    --dataset "${DATASET}" \
    --output_file "${OUTPUT_JSONL}"

echo "[DONE] experiment finished in ${EXP_DIR}"