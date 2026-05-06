#!/usr/bin/env bash
set -uo pipefail

while read -r qps set grad minexp_b minexp_alpha minexp_beta dataset frac; do
  [[ -z "${qps}" ]] && continue
  echo "[SWEEP] Running experiment: qps=${qps}, dataset=${dataset}, set=${set}, frac=${frac}"
  bash run_ab_one3.sh \
    --qps "${qps}" \
    --set "${set}" \
    --grad "${grad}" \
    --minexp-b "${minexp_b}" \
    --minexp-alpha "${minexp_alpha}" \
    --minexp-beta "${minexp_beta}" \
    --dataset "${dataset}" \
    --frac "${frac}" || true
  sleep 60
  echo "[SWEEP] Finished experiment: qps=${qps}, dataset=${dataset}, set=${set}, frac=${frac}"
done < ablation_config3.txt
