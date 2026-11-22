#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rc=0
while read -r qps set grad minexp_b minexp_alpha minexp_beta dataset frac; do
  [[ -z "${qps}" ]] && continue
  echo "[SWEEP] Running experiment: qps=${qps}, dataset=${dataset}, set=${set}, frac=${frac}"
  bash "${script_dir}/run_ab_one.sh" \
    --qps "${qps}" \
    --set "${set}" \
    --grad "${grad}" \
    --minexp-b "${minexp_b}" \
    --minexp-alpha "${minexp_alpha}" \
    --minexp-beta "${minexp_beta}" \
    --dataset "${dataset}" \
    --frac "${frac}" || rc=1
  sleep "${SWEEP_COOLDOWN_SECONDS:-60}"
  echo "[SWEEP] Finished experiment: qps=${qps}, dataset=${dataset}, set=${set}, frac=${frac}"
done < "${1:-${script_dir}/ablation_config.txt}"
exit "${rc}"
