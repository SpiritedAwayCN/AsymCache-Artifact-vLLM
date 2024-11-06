#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=2,3
export MKL_SERVICE_FORCE_INTEL=force
export VLLM_ALLOW_DEPRECATED_BLOCK_MANAGER_V1=1

python examples/test/profile_cache_rate_2seg.py 1k

python examples/test/profile_cache_rate.py 1k

python examples/test/profile_cache_rate_2seg.py 10k

python examples/test/profile_cache_rate.py 10k
