#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXPERIMENTS=(
  "asymcache"
  "asymcache_no_scheduler"
)

run_sweep() {
  local experiment="$1"
  local workload="$2"
  local model_size="$3"
  local sweep_id="$4"
  local work_dir="${ROOT_DIR}/e2e/${experiment}/${workload}_${model_size}"
  local script="run_sweep${sweep_id}.sh"

  echo "[RUN] ${experiment}/${workload}_${model_size}/${script}"
  (
    cd "${work_dir}"
    bash "${script}"
  )
}

wait_for_group() {
  local label="$1"
  shift

  local rc=0
  local pid
  for pid in "$@"; do
    if ! wait "${pid}"; then
      rc=1
    fi
  done

  if [[ "${rc}" -ne 0 ]]; then
    echo "[ERROR] ${label} failed"
  fi
  return "${rc}"
}

run_workload() {
  local experiment="$1"
  local workload="$2"

  echo "[START] ${experiment}/${workload}: 8B sweeps"

  local pids=()
  local sweep_id
  for sweep_id in 1 2 3 4; do
    run_sweep "${experiment}" "${workload}" "8B" "${sweep_id}" &
    pids+=("$!")
  done

  wait_for_group "${experiment}/${workload}_8B" "${pids[@]}" || return 1

  echo "[START] ${experiment}/${workload}: 70B sweep 1"
  run_sweep "${experiment}" "${workload}" "70B" 1 || return 1

  echo "[START] ${experiment}/${workload}: 70B sweep 2"
  run_sweep "${experiment}" "${workload}" "70B" 2 || return 1

  echo "[DONE] ${experiment}/${workload}"
}

run_experiment() {
  local experiment="$1"

  echo "========== [EXPERIMENT START] ${experiment} =========="

  run_workload "${experiment}" "longbench" &
  local longbench_pid=$!

  run_workload "${experiment}" "loogle" &
  local loogle_pid=$!

  local rc=0
  wait "${longbench_pid}" || rc=1
  wait "${loogle_pid}" || rc=1

  if [[ "${rc}" -ne 0 ]]; then
    echo "========== [EXPERIMENT FAILED] ${experiment} =========="
    return "${rc}"
  fi

  echo "========== [EXPERIMENT DONE] ${experiment} =========="
}

main() {
  local experiment
  for experiment in "${EXPERIMENTS[@]}"; do
    run_experiment "${experiment}" || exit 1
  done

  echo "========== [ALL DONE] =========="
}

main "$@"
