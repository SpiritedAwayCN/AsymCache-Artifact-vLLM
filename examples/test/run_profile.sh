#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
# export CUDA_VISIBLE_DEVICES=1,2
export MKL_SERVICE_FORCE_INTEL=force
export VLLM_ALLOW_DEPRECATED_BLOCK_MANAGER_V1=1
export VLLM_USE_V1=0
export VLLM_USE_POD_ATTN=1
export TRANSFORMERS_OFFLINE=1

for seqlen in 1k 10k
do
    # python profile_cache_rate.py $seqlen 1
    # python profile_cache_rate_2seg.py $seqlen 1
    # python profile_cache_rate.py $seqlen 2
    # python profile_cache_rate_2seg.py $seqlen 2
    # python profile_cache_rate.py $seqlen 4
    # python profile_cache_rate_2seg.py $seqlen 4
    python profile_cache_rate.py $seqlen 8
    python profile_cache_rate_2seg.py $seqlen 8
done
