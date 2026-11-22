#!/usr/bin/env bash
set -uo pipefail
rc=0
while read -r qps p grad minexp_b minexp_alpha minexp_beta dataset model_size frac; do
  [[ -z "${qps}" ]] && continue
  echo "[SWEEP] Running experiment: qps=${qps}, dataset=${dataset}, model_size=${model_size}, frac=${frac}"
# check
  bash run_one3.sh \
    --qps "${qps}" \
    --p "${p}" \
    --grad "${grad}" \
    --minexp-b "${minexp_b}" \
    --minexp-alpha "${minexp_alpha}" \
    --minexp-beta "${minexp_beta}" \
    --dataset "${dataset}" \
    --model-size "${model_size}" \
    --frac "${frac}" || rc=1
  sleep "${SWEEP_COOLDOWN_SECONDS:-60}"
  echo "[SWEEP] Finished experiment: qps=${qps}, dataset=${dataset}, model_size=${model_size}, frac=${frac}"
done < ./loogle_8B_config3.txt
exit "${rc}"
# check
