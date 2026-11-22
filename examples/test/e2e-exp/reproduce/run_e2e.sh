#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
baseline=${BASELINE:-ALL}
workload=${WORKLOAD:-all}
model_size=${MODEL_SIZE:-all}
sweep=${SWEEP:-}
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline) baseline="${2:?Missing baseline}"; shift 2 ;;
    --workload) workload="${2:?Missing workload}"; shift 2 ;;
    --model-size) model_size="${2:?Missing model size}"; shift 2 ;;
    --sweep) sweep="${2:?Missing sweep number}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help)
      echo "Usage: bash run_e2e.sh [--baseline ALL|LRU|PENSIEVE|BS|ASYMCACHE|ASYMCACHE_NO_SCHEDULER]"
      echo "       [--workload all|longbench|loogle] [--model-size all|8B|70B] [--sweep N] [--dry-run]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "${baseline^^}" in
  ALL) experiments=(lru pensieve max_score asymcache asymcache_no_scheduler) ;;
  BS|MAX_SCORE) experiments=(max_score) ;;
  LRU|PENSIEVE|ASYMCACHE|ASYMCACHE_NO_SCHEDULER) experiments=("${baseline,,}") ;;
  *) echo "Unsupported baseline: ${baseline}" >&2; exit 2 ;;
esac
case "${workload,,}" in
  all) workloads=(longbench loogle) ;;
  longbench|loogle) workloads=("${workload,,}") ;;
  *) echo "Unsupported workload: ${workload}" >&2; exit 2 ;;
esac
case "${model_size^^}" in
  ALL) sizes=(8B 70B) ;;
  8B|70B) sizes=("${model_size^^}") ;;
  *) echo "Unsupported model size: ${model_size}" >&2; exit 2 ;;
esac

if [[ -n "${sweep}" ]]; then
  if [[ ! "${sweep}" =~ ^[1-4]$ ]] ||
     [[ "${model_size^^}" != 8B && "${sweep}" -gt 2 ]]; then
    echo "Use sweep 1–4 for 8B or 1–2 for 70B" >&2
    exit 2
  fi
fi

for experiment in "${experiments[@]}"; do
  for dataset in "${workloads[@]}"; do
    for size in "${sizes[@]}"; do
      sweeps=(1 2)
      [[ "${size}" == 8B ]] && sweeps+=(3 4)
      [[ -n "${sweep}" ]] && sweeps=("${sweep}")
      for id in "${sweeps[@]}"; do
        run_dir="${script_dir}/e2e/${experiment}/${dataset}_${size}"
        run_script="run_sweep${id}.sh"
        echo "[RUN] ${experiment}/${dataset}_${size}/${run_script}"
        if [[ "${dry_run}" -eq 0 ]]; then
          (cd "${run_dir}" && bash "${run_script}")
        fi
      done
    done
  done
done
