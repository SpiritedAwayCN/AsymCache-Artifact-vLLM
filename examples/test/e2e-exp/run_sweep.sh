#!/usr/bin/env bash
set -uo pipefail

while read -r qps p grad minexp_b minexp_alpha minexp_beta dataset model_size frac; do
  [[ -z "${qps}" ]] && continue
  echo "[SWEEP] Running experiment: qps=${qps}, dataset=${dataset}, model_size=${model_size}, frac=${frac}"
  bash run_one.sh \
    --qps "${qps}" \
    --p "${p}" \
    --grad "${grad}" \
    --minexp-b "${minexp_b}" \
    --minexp-alpha "${minexp_alpha}" \
    --minexp-beta "${minexp_beta}" \
    --dataset "${dataset}" \
    --model-size "${model_size}" \
    --frac "${frac}" || true
  sleep 60
  echo "[SWEEP] Finished experiment: qps=${qps}, dataset=${dataset}, model_size=${model_size}, frac=${frac}"
done < exp_config.txt