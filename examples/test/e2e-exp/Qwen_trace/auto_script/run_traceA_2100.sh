#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then
  source "${SCRIPT_DIR}/env.sh"
fi

TRACEA_DIR=${TRACEA_DIR:-${SCRIPT_DIR}}
TRACEA_AGENT=${TRACEA_AGENT:-}
INPUT_FILE=${INPUT_FILE:-}
OUTPUT_FILE=${OUTPUT_FILE:-}
RESULT_ROOT=${RESULT_ROOT:-}
RUN_NAME=${RUN_NAME:-}
BASELINE=${BASELINE:-}
MODEL_SIZE=${MODEL_SIZE:-}
QPS=${QPS:-}
TIME_SCALE=${TIME_SCALE:-}
PORT=${PORT:-}
MODEL_PATH=${MODEL_PATH:-}
CONFIG_FILE=""
ALL_POLICIES=0
SINGLE_RUN=0
CONTINUE_ON_ERROR=0
DRY_RUN=0
COOLDOWN_SECONDS=${COOLDOWN_SECONDS:-60}
FAIL_ON_REQUEST_ERROR=1
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
RETRY_FAST_FAILURE_SECONDS=${RETRY_FAST_FAILURE_SECONDS:-2.0}
RETRY_BACKOFF_SECONDS=${RETRY_BACKOFF_SECONDS:-0.5}
MINEXP_B_OVERRIDE=${MINEXP_B:-}
MINEXP_ALPHA_OVERRIDE=${MINEXP_ALPHA:-}
MINEXP_BETA_OVERRIDE=${MINEXP_BETA:-}
AGING_INTERVAL_OVERRIDE=${AGING_INTERVAL:-}

usage() {
  cat <<'EOF'
Usage:
  # Run one configuration.
  bash run_traceA_2100.sh --baseline LRU --model-size 70B --qps 0.2

  # Run LRU, PENSIEVE, BS, MWF, and HOTPREFIX sequentially.
  bash run_traceA_2100.sh --all-policies --model-size 70B --qps 0.2

  # Run arbitrary configurations from a CSV file.
  bash run_traceA_2100.sh --config-file tracea_sweep.csv

Single-run and sweep options:
  --baseline NAME       LRU, PENSIEVE, BS, MWF, HOTPREFIX, CONTINUUM, or CONTINUUM_MWF
  --model-size SIZE     8B or 70B
  --qps VALUE           Scale the selected trace to this average request QPS
  --time-scale VALUE    Replay speed multiplier; mutually exclusive with QPS
  --all-policies        Run the five TraceA baselines
  --dry-run             Print the selected settings without starting a server
  --config-file PATH    Run one configuration per CSV row
  --continue-on-error   Continue the sweep after a failed configuration
  --cooldown-seconds N  Delay between configurations; default 60

Paths and overrides:
  --tracea-dir PATH
  --agent-script PATH
  --input-file PATH
  --output-file PATH    Single-run only
  --result-root PATH
  --run-name NAME       Single-run only
  --model-path PATH
  --port PORT
  --minexp-b VALUE
  --minexp-alpha VALUE
  --minexp-beta VALUE
  --aging-interval N    HOTPREFIX aging interval for a single run

Replay reliability:
  --max-attempts N
  --retry-fast-failure-seconds N
  --retry-backoff-seconds N
  --allow-request-errors  Do not fail a configuration when requests remain failed

CSV format (header is optional):
  baseline,model_size,timing_mode,timing_value,minexp_b,minexp_alpha,minexp_beta,run_name

Use '-' for optional values. Example:
  LRU,70B,qps,0.5,-,-,-,traceA_LRU_qps0.5
  PENSIEVE,70B,qps,0.5,-,-,-,traceA_PENSIEVE_qps0.5
  BS,70B,qps,0.5,-9.499324e+03,1.884532e+05,4.711331e+03,traceA_BS_qps0.5
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline) BASELINE="$2"; shift 2 ;;
    --model-size) MODEL_SIZE="$2"; shift 2 ;;
    --qps) QPS="$2"; shift 2 ;;
    --time-scale) TIME_SCALE="$2"; shift 2 ;;
    --all-policies) ALL_POLICIES=1; shift ;;
    --config-file) CONFIG_FILE="$2"; shift 2 ;;
    --continue-on-error) CONTINUE_ON_ERROR=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --cooldown-seconds) COOLDOWN_SECONDS="$2"; shift 2 ;;
    --tracea-dir) TRACEA_DIR="$2"; shift 2 ;;
    --agent-script) TRACEA_AGENT="$2"; shift 2 ;;
    --input-file) INPUT_FILE="$2"; shift 2 ;;
    --output-file) OUTPUT_FILE="$2"; shift 2 ;;
    --result-root) RESULT_ROOT="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --minexp-b) MINEXP_B_OVERRIDE="$2"; shift 2 ;;
    --minexp-alpha) MINEXP_ALPHA_OVERRIDE="$2"; shift 2 ;;
    --minexp-beta) MINEXP_BETA_OVERRIDE="$2"; shift 2 ;;
    --aging-interval) AGING_INTERVAL_OVERRIDE="$2"; shift 2 ;;
    --max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
    --retry-fast-failure-seconds)
      RETRY_FAST_FAILURE_SECONDS="$2"
      shift 2
      ;;
    --retry-backoff-seconds) RETRY_BACKOFF_SECONDS="$2"; shift 2 ;;
    --allow-request-errors) FAIL_ON_REQUEST_ERROR=0; shift ;;
    --single-run) SINGLE_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -n "${QPS}" && -n "${TIME_SCALE}" ]]; then
  echo "--qps and --time-scale are mutually exclusive" >&2
  exit 1
fi
if [[ -n "${CONFIG_FILE}" && "${ALL_POLICIES}" -eq 1 ]]; then
  echo "--config-file and --all-policies are mutually exclusive" >&2
  exit 1
fi

append_common_child_args() {
  local -n args_ref=$1
  args_ref+=(
    --tracea-dir "${TRACEA_DIR}"
    --max-attempts "${MAX_ATTEMPTS}"
    --retry-fast-failure-seconds "${RETRY_FAST_FAILURE_SECONDS}"
    --retry-backoff-seconds "${RETRY_BACKOFF_SECONDS}"
  )
  [[ -n "${TRACEA_AGENT}" ]] && args_ref+=(--agent-script "${TRACEA_AGENT}")
  [[ -n "${INPUT_FILE}" ]] && args_ref+=(--input-file "${INPUT_FILE}")
  [[ -n "${RESULT_ROOT}" ]] && args_ref+=(--result-root "${RESULT_ROOT}")
  [[ -n "${MODEL_PATH}" ]] && args_ref+=(--model-path "${MODEL_PATH}")
  [[ -n "${PORT}" ]] && args_ref+=(--port "${PORT}")
  [[ "${FAIL_ON_REQUEST_ERROR}" -eq 0 ]] && args_ref+=(--allow-request-errors)
  [[ "${DRY_RUN}" -eq 1 ]] && args_ref+=(--dry-run)
  return 0
}

run_child_configuration() {
  local baseline=$1
  local model_size=$2
  local timing_mode=$3
  local timing_value=$4
  local minexp_b=${5:-}
  local minexp_alpha=${6:-}
  local minexp_beta=${7:-}
  local run_name=${8:-}
  local aging_interval=${9:-}
  local args=(
    bash "${SCRIPT_PATH}"
    --single-run
    --baseline "${baseline}"
    --model-size "${model_size}"
  )
  append_common_child_args args

  case "${timing_mode}" in
    qps) args+=(--qps "${timing_value}") ;;
    time-scale|timescale) args+=(--time-scale "${timing_value}") ;;
    *) echo "Unsupported timing_mode=${timing_mode}" >&2; return 1 ;;
  esac
  [[ -n "${minexp_b}" && "${minexp_b}" != "-" ]] &&
    args+=(--minexp-b "${minexp_b}")
  [[ -n "${minexp_alpha}" && "${minexp_alpha}" != "-" ]] &&
    args+=(--minexp-alpha "${minexp_alpha}")
  [[ -n "${minexp_beta}" && "${minexp_beta}" != "-" ]] &&
    args+=(--minexp-beta "${minexp_beta}")
  [[ -n "${run_name}" && "${run_name}" != "-" ]] &&
    args+=(--run-name "${run_name}")
  [[ -n "${aging_interval}" && "${aging_interval}" != "-" ]] &&
    args+=(--aging-interval "${aging_interval}")

  echo "[SWEEP] baseline=${baseline} model=${model_size} ${timing_mode}=${timing_value}${aging_interval:+ aging_interval=${aging_interval}}"
  "${args[@]}"
}

run_batch_item() {
  if run_child_configuration "$@"; then
    echo "[SWEEP] configuration completed"
    return 0
  else
    local rc=$?
    echo "[SWEEP] configuration failed with rc=${rc}" >&2
    return "${rc}"
  fi
  return 0
}

run_config_file() {
  if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Missing config file: ${CONFIG_FILE}" >&2
    return 1
  fi

  local line line_number=0 executed=0 failed=0
  while IFS= read -r line || [[ -n "${line}" ]]; do
    ((line_number += 1))
    line=${line%$'\r'}
    [[ -z "${line//[[:space:]]/}" || "${line}" =~ ^[[:space:]]*# ]] && continue

    local baseline model_size timing_mode timing_value
    local minexp_b minexp_alpha minexp_beta run_name extra
    IFS=',' read -r baseline model_size timing_mode timing_value \
      minexp_b minexp_alpha minexp_beta run_name extra <<< "${line}"
    baseline=${baseline//[[:space:]]/}
    model_size=${model_size//[[:space:]]/}
    timing_mode=${timing_mode//[[:space:]]/}
    timing_value=${timing_value//[[:space:]]/}
    minexp_b=${minexp_b//[[:space:]]/}
    minexp_alpha=${minexp_alpha//[[:space:]]/}
    minexp_beta=${minexp_beta//[[:space:]]/}
    run_name=${run_name//[[:space:]]/}

    [[ "${baseline,,}" == "baseline" ]] && continue
    if [[ -z "${baseline}" || -z "${model_size}" ||
          -z "${timing_mode}" || -z "${timing_value}" ]]; then
      echo "Invalid config row ${line_number}: ${line}" >&2
      return 1
    fi

    if [[ "${DRY_RUN}" -eq 0 && "${executed}" -gt 0 && "${COOLDOWN_SECONDS}" != "0" ]]; then
      echo "[SWEEP] cooling down for ${COOLDOWN_SECONDS}s"
      sleep "${COOLDOWN_SECONDS}"
    fi
    run_batch_item "${baseline}" "${model_size}" "${timing_mode}" \
      "${timing_value}" "${minexp_b}" "${minexp_alpha}" \
      "${minexp_beta}" "${run_name}" || {
        [[ "${CONTINUE_ON_ERROR}" -eq 1 ]] || return 1
        failed=1
      }
    ((executed += 1))
  done < "${CONFIG_FILE}"

  if [[ "${executed}" -eq 0 ]]; then
    echo "No configurations found in ${CONFIG_FILE}" >&2
    return 1
  fi
  echo "[SWEEP] attempted ${executed} configured runs"
  return "${failed}"
}

run_all_policies() {
  if [[ -z "${MODEL_SIZE}" ]]; then
    echo "--all-policies requires --model-size" >&2
    return 1
  fi
  local timing_mode timing_value
  if [[ -n "${QPS}" ]]; then
    timing_mode=qps
    timing_value=${QPS}
  else
    timing_mode=time-scale
    timing_value=${TIME_SCALE:-1.0}
  fi

  local policies=(LRU PENSIEVE BS MWF HOTPREFIX)
  local index failed=0
  for index in "${!policies[@]}"; do
    if [[ "${DRY_RUN}" -eq 0 && "${index}" -gt 0 && "${COOLDOWN_SECONDS}" != "0" ]]; then
      echo "[SWEEP] cooling down for ${COOLDOWN_SECONDS}s"
      sleep "${COOLDOWN_SECONDS}"
    fi
    run_batch_item "${policies[${index}]}" "${MODEL_SIZE}" \
      "${timing_mode}" "${timing_value}" \
      "${MINEXP_B_OVERRIDE}" "${MINEXP_ALPHA_OVERRIDE}" \
      "${MINEXP_BETA_OVERRIDE}" "" "${AGING_INTERVAL_OVERRIDE}" || {
        [[ "${CONTINUE_ON_ERROR}" -eq 1 ]] || return 1
        failed=1
      }
  done
  return "${failed}"
}

if [[ -n "${CONFIG_FILE}" && "${SINGLE_RUN}" -eq 0 ]]; then
  run_config_file
  exit $?
fi
if [[ "${ALL_POLICIES}" -eq 1 && "${SINGLE_RUN}" -eq 0 ]]; then
  run_all_policies
  exit $?
fi

if [[ -z "${BASELINE}" || -z "${MODEL_SIZE}" ]]; then
  echo "Single run requires --baseline and --model-size" >&2
  usage >&2
  exit 1
fi

BASELINE=${BASELINE^^}
MODEL_SIZE=${MODEL_SIZE^^}
if [[ -n "${QPS}" ]]; then
  TIMING_TAG="qps${QPS}"
  REPLAY_TIMING_ARGS=(--target-qps "${QPS}")
  export TRACEA_TARGET_QPS="${QPS}"
else
  TIME_SCALE=${TIME_SCALE:-1.0}
  TIMING_TAG="timescale${TIME_SCALE}"
  REPLAY_TIMING_ARGS=(--time-scale "${TIME_SCALE}")
  unset TRACEA_TARGET_QPS
fi

TRACEA_AGENT=${TRACEA_AGENT:-${TRACEA_DIR}/replay_traceA.py}
INPUT_FILE=${INPUT_FILE:-${TRACEA_DIR}/traceA_2k_final_workloads/tracea_2k_cache_pressure_relaxed_2100.jsonl}

case "${MODEL_SIZE}" in
  70B)
    MODEL_PATH=${MODEL_PATH:-${MODEL_70B_PATH:-}}
    TP=${TP:-4}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
    STARTUP_SLEEP=${STARTUP_SLEEP:-500}
    DEFAULT_MINEXP_B=-5.536092e+02
    DEFAULT_MINEXP_ALPHA=1.973722e+03
    DEFAULT_MINEXP_BETA=4.934304e+01
    ;;
  8B)
    MODEL_PATH=${MODEL_PATH:-${MODEL_8B_PATH:-}}
    TP=${TP:-1}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
    STARTUP_SLEEP=${STARTUP_SLEEP:-90}
    DEFAULT_MINEXP_B=-5.536092e+02
    DEFAULT_MINEXP_ALPHA=1.973722e+03
    DEFAULT_MINEXP_BETA=4.934304e+01
    ;;
  *)
    echo "Unsupported MODEL_SIZE=${MODEL_SIZE}; expected 8B or 70B" >&2
    exit 1
    ;;
esac
if [[ "${DRY_RUN}" -eq 0 && -z "${MODEL_PATH}" ]]; then
  echo "Model path is empty; set MODEL_${MODEL_SIZE}_PATH or pass --model-path" >&2
  exit 1
fi

MINEXP_A=0
MINEXP_B=${MINEXP_B_OVERRIDE:-${DEFAULT_MINEXP_B}}
MINEXP_ALPHA=${MINEXP_ALPHA_OVERRIDE:-${DEFAULT_MINEXP_ALPHA}}
MINEXP_BETA=${MINEXP_BETA_OVERRIDE:-${DEFAULT_MINEXP_BETA}}
unset USE_CONTINUUM DISABLE_MULTIPLIER AGING_INTERVAL
export VLLM_SIMPLE_SHED_ALGO=0

case "${BASELINE}" in
  LRU)
    PORT=${PORT:-8052}
    export VLLM_EVICTOR_POLICY=lru
    export USE_MINEXP=0
    export DISABLE_MULTIPLIER=1
    ;;
  PENSIEVE)
    PORT=${PORT:-8053}
    export VLLM_EVICTOR_POLICY=pensieve
    export USE_MINEXP=0
    ;;
  HOTPREFIX)
    PORT=${PORT:-8062}
    export AGING_INTERVAL=${AGING_INTERVAL_OVERRIDE:-2}
    export VLLM_EVICTOR_POLICY=hotprefix
    export USE_MINEXP=0
    ;;
  BS|MAX_SCORE)
    BASELINE=BS
    PORT=${PORT:-8054}
    export VLLM_EVICTOR_POLICY=bs
    export USE_MINEXP=2
    ;;
  MWF)
    PORT=${PORT:-8066}
    export VLLM_EVICTOR_POLICY=mwf
    export USE_MINEXP=2
    ;;
  CONTINUUM)
    PORT=${PORT:-8166}
    export VLLM_EVICTOR_POLICY=lru
    export USE_MINEXP=0
    export USE_CONTINUUM=1
    ;;
  CONTINUUM_MWF)
    PORT=${PORT:-8153}
    export VLLM_EVICTOR_POLICY=mwf
    export USE_MINEXP=2
    export USE_CONTINUUM=1
    ;;
  *)
    echo "Unsupported BASELINE=${BASELINE}" >&2
    exit 1
    ;;
esac

RUN_NAME=${RUN_NAME:-traceA_2k_2100_${BASELINE}_${MODEL_SIZE}_${TIMING_TAG}_P75_max48}
RESULT_ROOT=${RESULT_ROOT:-${SCRIPT_DIR}/${MODEL_SIZE}_results}
RESULT_DIR="${RESULT_ROOT}/${RUN_NAME}"
OUTPUT_FILE=${OUTPUT_FILE:-${RESULT_DIR}/${RUN_NAME}.jsonl}
SCHEDULE_LOG_PATH="${RESULT_DIR}/scheduler.log"
LOG_FILE="${RESULT_DIR}/server_output.log"
REPLAY_SUMMARY_FILE="${RESULT_DIR}/replay_summary.json"
VLLM_REQUEST_OUTPUTS_FILE="${RESULT_DIR}/engine_request_outputs.jsonl"

PYTHON_BIN=${PYTHON_BIN:-python3}
DTYPE=${DTYPE:-float16}
BLOCK_SIZE=${BLOCK_SIZE:-64}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-450000}
MAX_NUM_PARTIAL_PREFILLS=${MAX_NUM_PARTIAL_PREFILLS:-4}
MAX_LONG_PARTIAL_PREFILLS=${MAX_LONG_PARTIAL_PREFILLS:-4}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}

export VLLM_USE_V1=0
export VLLM_USE_POD_ATTN=${VLLM_USE_POD_ATTN:-1}
export MINEXP_A MINEXP_B MINEXP_ALPHA MINEXP_BETA
export VLLM_EVICTOR_POLICY USE_MINEXP
export SCHEDULE_LOG_PATH VLLM_REQUEST_OUTPUTS_FILE
export no_proxy="127.0.0.1,localhost,::1${no_proxy:+,${no_proxy}}"
export NO_PROXY="${no_proxy}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf '%s\n' "BASELINE=${BASELINE} MODEL_SIZE=${MODEL_SIZE} TIMING=${TIMING_TAG}" \
    "VLLM_EVICTOR_POLICY=${VLLM_EVICTOR_POLICY} USE_MINEXP=${USE_MINEXP} USE_CONTINUUM=${USE_CONTINUUM:-0} AGING_INTERVAL=${AGING_INTERVAL:-}" \
    "MINEXP_B=${MINEXP_B} MINEXP_ALPHA=${MINEXP_ALPHA} MINEXP_BETA=${MINEXP_BETA}" \
    "INPUT_FILE=${INPUT_FILE}" "RESULT_DIR=${RESULT_DIR}"
  exit 0
fi

for required_file in "${TRACEA_AGENT}" "${INPUT_FILE}" "$(dirname "${INPUT_FILE}")/mapping_config.json"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing TraceA file: ${required_file}" >&2
    exit 1
  fi
done

mkdir -p "${RESULT_DIR}"
: > "${VLLM_REQUEST_OUTPUTS_FILE}"

wait_for_server() {
  local port=$1
  local server_pid=$2
  echo "[SERVER] waiting for health endpoint on port ${port}"
  local waited=0
  while (( waited < 1200 )); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "[SERVER] process ${server_pid} exited during startup" >&2
      tail -n 100 "${LOG_FILE}" >&2 || true
      return 1
    fi
    if curl --noproxy '*' --fail --silent \
      "http://127.0.0.1:${port}/health" > /dev/null; then
      echo "[SERVER] healthy on port ${port}"
      return 0
    fi
    sleep 1
    ((waited += 1))
  done
  echo "[SERVER] timed out on port ${port}" >&2
  tail -n 100 "${LOG_FILE}" >&2 || true
  return 1
}

SERVER_PID=""
SERVER_PGID=""
cleanup() {
  local rc=$?
  set +e
  trap - EXIT
  if [[ -n "${SERVER_PGID}" ]]; then
    echo "[CLEANUP] terminating vLLM process group ${SERVER_PGID}"
    kill -TERM -- "-${SERVER_PGID}" 2>/dev/null
    sleep 6
    kill -KILL -- "-${SERVER_PGID}" 2>/dev/null
  elif [[ -n "${SERVER_PID}" ]]; then
    echo "[CLEANUP] terminating vLLM pid ${SERVER_PID}"
    kill -TERM "${SERVER_PID}" 2>/dev/null
    sleep 6
    kill -KILL "${SERVER_PID}" 2>/dev/null
  fi
  wait "${SERVER_PID}" 2>/dev/null
  echo "[CLEANUP] finished"
  exit "${rc}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

echo "=== Trace A automated run ==="
echo "RUN_NAME=${RUN_NAME}"
echo "BASELINE=${BASELINE}"
echo "VLLM_EVICTOR_POLICY=${VLLM_EVICTOR_POLICY}"
echo "USE_MINEXP=${USE_MINEXP}"
[[ "${BASELINE}" == "HOTPREFIX" ]] &&
  echo "AGING_INTERVAL=${AGING_INTERVAL}"
echo "MODEL_SIZE=${MODEL_SIZE}"
echo "TIMING_MODE=${TIMING_TAG}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "PORT=${PORT}"
echo "OUTPUT_FILE=${OUTPUT_FILE}"
echo "RESULT_DIR=${RESULT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" setsid "${PYTHON_BIN}" \
  -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  -tp "${TP}" \
  --port "${PORT}" \
  --dtype "${DTYPE}" \
  --block-size "${BLOCK_SIZE}" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-seqs 48 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-partial-prefills "${MAX_NUM_PARTIAL_PREFILLS}" \
  --max-long-partial-prefills "${MAX_LONG_PARTIAL_PREFILLS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  > "${LOG_FILE}" 2>&1 &
SERVER_PID=$!
SERVER_PGID=$(ps -o pgid= -p "${SERVER_PID}" | tr -d ' ' || true)
SCRIPT_PGID=$(ps -o pgid= -p "$$" | tr -d ' ' || true)
if [[ -z "${SERVER_PGID}" || "${SERVER_PGID}" == "${SCRIPT_PGID}" ]]; then
  echo "[WARN] unsafe/empty server PGID; cleanup will use pid ${SERVER_PID}"
  SERVER_PGID=""
fi
echo "[SERVER] pid=${SERVER_PID} pgid=${SERVER_PGID:-<none>}"

sleep "${STARTUP_SLEEP}"
wait_for_server "${PORT}" "${SERVER_PID}"

REPLAY_ARGS=(
  "${PYTHON_BIN}" "${TRACEA_AGENT}"
  -i "${INPUT_FILE}"
  -o "${OUTPUT_FILE}"
  --summary-file "${REPLAY_SUMMARY_FILE}"
  --run-id "${RUN_NAME}"
  --baseline "${BASELINE}"
  --model-size "${MODEL_SIZE}"
  --model "${MODEL_PATH}"
  --api_url "http://127.0.0.1:${PORT}/v1/completions"
  --max-attempts "${MAX_ATTEMPTS}"
  --retry-fast-failure-seconds "${RETRY_FAST_FAILURE_SECONDS}"
  --retry-backoff-seconds "${RETRY_BACKOFF_SECONDS}"
  "${REPLAY_TIMING_ARGS[@]}"
)
[[ "${FAIL_ON_REQUEST_ERROR}" -eq 1 ]] &&
  REPLAY_ARGS+=(--fail-on-request-error)

"${REPLAY_ARGS[@]}"
echo "[DONE] Trace A experiment finished in ${RESULT_DIR}"
