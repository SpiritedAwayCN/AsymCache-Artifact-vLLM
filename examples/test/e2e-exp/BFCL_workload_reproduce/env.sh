#!/bin/bash
# Common environment for all BFCL baseline runs.
# Baseline-specific variables such as USE_MINEXP, USE_CONTINUUM, and VLLM_EVICTOR_POLICY are set by run_bfcl_baseline_server.sh.

export NCCL_ALGO=${NCCL_ALGO:-Ring}
export VLLM_USE_V1=${VLLM_USE_V1:-0}
export VLLM_PORT=${VLLM_PORT:-12415}
export VLLM_USE_POD_ATTN=${VLLM_USE_POD_ATTN:-1}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export VLLM_HOST_IP=${VLLM_HOST_IP:-$(hostname -I | awk '{print $1}')}
export VLLM_ALLOW_DEPRECATED_BLOCK_MANAGER_V1=${VLLM_ALLOW_DEPRECATED_BLOCK_MANAGER_V1:-1}

export HF_HOME=${HF_HOME:-/jizhicfs/marvinhjia/sca/models/huggingface}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}
export CSET_DIR=${CSET_DIR:-/jizhicfs/marvinhjia/sca/yilin/vllm-dev/csrc/c_set}
export PYTHON_BIN=${PYTHON_BIN:-/jizhicfs/marvinhjia/miniconda3/envs/chenyilin-vllm/bin/python3}
export MODEL_8B_PATH=${MODEL_8B_PATH:-/jizhicfs/marvinhjia/sca/models/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}
export MODEL_70B_PATH=${MODEL_70B_PATH:-/jizhicfs/marvinhjia/sca/models/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b}

export BLOCK_SIZE=${BLOCK_SIZE:-64}
export DTYPE=${DTYPE:-float16}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-450000}
export MAX_NUM_PARTIAL_PREFILLS=${MAX_NUM_PARTIAL_PREFILLS:-4}
export MAX_LONG_PARTIAL_PREFILLS=${MAX_LONG_PARTIAL_PREFILLS:-4}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}
