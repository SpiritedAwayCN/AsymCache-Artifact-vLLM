#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
# export CUDA_VISIBLE_DEVICES=1,2
export MKL_SERVICE_FORCE_INTEL=force
export VLLM_ALLOW_DEPRECATED_BLOCK_MANAGER_V1=1
export VLLM_USE_V1=0
export VLLM_USE_POD_ATTN=1
export TRANSFORMERS_OFFLINE=1

for tp in 2 4 8
do
    python benchmark_model_forward_time.py $tp
done
