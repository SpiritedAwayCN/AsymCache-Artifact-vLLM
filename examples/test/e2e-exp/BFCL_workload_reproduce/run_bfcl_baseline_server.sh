#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then
  source "${SCRIPT_DIR}/env.sh"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline)
      BASELINE="$2"
      shift 2
      ;;
    --agent-script)
      AGENT_SCRIPT="$2"
      shift 2
      ;;
    --qps)
      QPS="$2"
      shift 2
      ;;
    --model-size)
      MODEL_SIZE="$2"
      shift 2
      ;;
    --input-file)
      INPUT_FILE="$2"
      shift 2
      ;;
    --output-file)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    --result-prefix)
      RESULT_PREFIX="$2"
      shift 2
      ;;
    --result-root)
      RESULT_ROOT="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  bash run_bfcl_baseline_server.sh --baseline BASELINE --model-size 8B|70B --qps QPS [options]

Required:
  --baseline      LRU, MWF, CONTINUUM, or CONTINUUM_MWF
  --model-size    8B or 70B
  --qps           e.g. 1.5, 1.75, 2.0

Optional:
  --agent-script  Override matching agent interface script
  --input-file    Override timestamp CSV path
  --output-file   Override replay JSONL path
  --result-prefix Override result prefix, default 2x
  --result-root   Override result root directory
  --run-name      Override run directory name
  --port          Override vLLM API port
  --model-path    Override model path
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

BASELINE=${BASELINE:-}
QPS=${QPS:-}
MODEL_SIZE=${MODEL_SIZE:-}
if [[ -z "${BASELINE}" ]]; then
  echo "Set BASELINE or pass --baseline LRU|MWF|CONTINUUM|CONTINUUM_MWF" >&2
  exit 1
fi
if [[ -z "${MODEL_SIZE}" ]]; then
  echo "Set MODEL_SIZE or pass --model-size 8B|70B" >&2
  exit 1
fi
if [[ -z "${QPS}" ]]; then
  echo "Set QPS or pass --qps 1.5|1.75|2.0" >&2
  exit 1
fi

RESULT_PREFIX=${RESULT_PREFIX:-2x}
DATA_DIR=${DATA_DIR:-/jizhicfs/marvinhjia/sca/yilin/multi-segment-attention/BFCL_dataset}
RESULT_ROOT=${RESULT_ROOT:-./${MODEL_SIZE}_results}
RUN_NAME=${RUN_NAME:-${RESULT_PREFIX}_${BASELINE}_qps${QPS}}
RESULT_DIR=${RESULT_DIR:-${RESULT_ROOT}/${RUN_NAME}}
INPUT_FILE=${INPUT_FILE:-${DATA_DIR}/${RESULT_PREFIX}_bfcl_web_search_prompts_qps${QPS}.csv}
OUTPUT_FILE=${OUTPUT_FILE:-${RESULT_DIR}/${RESULT_PREFIX}_${BASELINE}_replay_output_qps${QPS}.jsonl}
SCHEDULE_LOG_PATH=${SCHEDULE_LOG_PATH:-${RESULT_DIR}/schedule_qps${QPS}.log}
LOG_FILE=${LOG_FILE:-${RESULT_DIR}/server_output_qps${QPS}.log}

if [[ "${MODEL_SIZE}" == "70B" ]]; then
  MODEL_PATH=${MODEL_PATH:-${MODEL_70B_PATH}}
  TP=${TP:-4}
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
  STARTUP_SLEEP=${STARTUP_SLEEP:-550}
  MINEXP_B=${MINEXP_B:--2.010611e+01}
  MINEXP_ALPHA=${MINEXP_ALPHA:-7.124777e+03}
  MINEXP_BETA=${MINEXP_BETA:-1.781194e+02}
elif [[ "${MODEL_SIZE}" == "8B" ]]; then
  MODEL_PATH=${MODEL_PATH:-${MODEL_8B_PATH}}
  TP=${TP:-1}
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
  STARTUP_SLEEP=${STARTUP_SLEEP:-90}
  MINEXP_B=${MINEXP_B:--1.353967e+01}
  MINEXP_ALPHA=${MINEXP_ALPHA:-4.797900e+03}
  MINEXP_BETA=${MINEXP_BETA:-1.199475e+02}
else
  echo "Unsupported MODEL_SIZE=${MODEL_SIZE}; expected 8B or 70B" >&2
  exit 1
fi

case "${BASELINE}" in
  LRU)
    AGENT_SCRIPT=${AGENT_SCRIPT:-agent_interface_lru_agent.py}
    PORT=${PORT:-8052}
    export DISABLE_MULTIPLIER=${DISABLE_MULTIPLIER:-1}
    export USE_MINEXP=${USE_MINEXP:-0}
    export VLLM_EVICTOR_POLICY=${VLLM_EVICTOR_POLICY:-lru}
    unset USE_CONTINUUM
    ;;
  MWF)
    AGENT_SCRIPT=${AGENT_SCRIPT:-agent_interface_mwf_agent.py}
    PORT=${PORT:-8059}
    export USE_MINEXP=${USE_MINEXP:-2}
    export VLLM_EVICTOR_POLICY=${VLLM_EVICTOR_POLICY:-mwf}
    unset USE_CONTINUUM
    ;;
  CONTINUUM)
    AGENT_SCRIPT=${AGENT_SCRIPT:-agent_interface_continuum.py}
    PORT=${PORT:-8166}
    export USE_CONTINUUM=${USE_CONTINUUM:-1}
    export USE_MINEXP=${USE_MINEXP:-0}
    export VLLM_EVICTOR_POLICY=${VLLM_EVICTOR_POLICY:-lru}
    ;;
  CONTINUUM_MWF)
    AGENT_SCRIPT=${AGENT_SCRIPT:-agent_interface_continuum_mwf.py}
    PORT=${PORT:-8153}
    export USE_CONTINUUM=${USE_CONTINUUM:-1}
    export USE_MINEXP=${USE_MINEXP:-2}
    export VLLM_EVICTOR_POLICY=${VLLM_EVICTOR_POLICY:-mwf}
    ;;
  *)
    echo "Unsupported BASELINE=${BASELINE}" >&2
    exit 1
    ;;
esac

export HF_HOME
export HF_ENDPOINT
export HUGGINGFACE_HUB_CACHE
export CSET_DIR
export MINEXP_A=${MINEXP_A:-0}
export MINEXP_B
export MINEXP_ALPHA
export MINEXP_BETA
export SCHEDULE_LOG_PATH
mkdir -p "${RESULT_DIR}"

if "${PYTHON_BIN}" -c "import quart" &> /dev/null; then
  echo "Quart is already installed."
else
  echo "Quart is not installed. Installing..."
  "${PYTHON_BIN}" -m pip install quart
fi

wait_for_server() {
  local port=$1
  echo "Waiting for server to start on port ${port}..."
  timeout 1200 bash -c "
    until curl -s http://127.0.0.1:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

echo "=== BFCL baseline run ==="
echo "BASELINE=${BASELINE}"
echo "MODEL_SIZE=${MODEL_SIZE}"
echo "QPS=${QPS}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TP=${TP}"
echo "PORT=${PORT}"
echo "INPUT_FILE=${INPUT_FILE}"
echo "OUTPUT_FILE=${OUTPUT_FILE}"
echo "RESULT_DIR=${RESULT_DIR}"
echo "SCHEDULE_LOG_PATH=${SCHEDULE_LOG_PATH}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} "${PYTHON_BIN}" \
  -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  -tp "${TP}" \
  --port "${PORT}" \
  --block-size "${BLOCK_SIZE}" \
  --dtype "${DTYPE}" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-partial-prefills "${MAX_NUM_PARTIAL_PREFILLS}" \
  --max-long-partial-prefills "${MAX_LONG_PARTIAL_PREFILLS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  > "${LOG_FILE}" 2>&1 &

sleep "${STARTUP_SLEEP}"
wait_for_server "${PORT}"

"${PYTHON_BIN}" "${AGENT_SCRIPT}" \
  -i "${INPUT_FILE}" \
  -o "${OUTPUT_FILE}" \
  --model "${MODEL_PATH}" \
  --api_url "http://127.0.0.1:${PORT}/v1/completions"

sleep infinity

ps -e | grep pt_main_thread | awk '{print $1}' | xargs kill -9 > /dev/null 2>&1
pkill -f python > /dev/null 2>&1
