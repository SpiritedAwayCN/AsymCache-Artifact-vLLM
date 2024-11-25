#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=4,5,6,7
export MKL_SERVICE_FORCE_INTEL=force
export VLLM_ALLOW_DEPRECATED_BLOCK_MANAGER_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

for seqlen in 1k 5k 10k 20k 40k 80k
do
    for starttoken in 32 96 160 112 128 192 144 48 176 64 80 16
    do
        python examples/test/profile_prefill_time.py $seqlen $starttoken
    done
done