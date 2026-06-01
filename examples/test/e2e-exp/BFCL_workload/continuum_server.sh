#!/bin/bash
export VLLM_USE_V1=0
export NCCL_ALGO=Ring
export VLLM_PORT=12413
export VLLM_USE_POD_ATTN=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=INFO
export HF_ENDPOINT=https://hf-mirror.com
export VLLM_ALLOW_DEPRECATED_BLOCK_MANAGER_V1=1
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
export HF_HOME=/jizhicfs/marvinhjia/sca/models/huggingface
export CSET_DIR=/jizhicfs/marvinhjia/sca/yilin/vllm-dev/csrc/c_set
export HUGGINGFACE_HUB_CACHE=/jizhicfs/marvinhjia/sca/models/huggingface/hub

export USE_CONTINUUM=1
export USE_MINEXP=0
export VLLM_EVICTOR_POLICY=lru
export MINEXP_A=0
# export MINEXP_B=-2.010611e+01
# export MINEXP_ALPHA=7.124777e+03
# export MINEXP_BETA=1.781194e+02
export MINEXP_B=-1.353967e+01
export MINEXP_ALPHA=4.797900e+03
export MINEXP_BETA=1.199475e+02
export SCHEDULE_LOG_PATH="./8B_results/2x_CONTINUUM_qps1.75/schedule_qps1.75.log"
# mkdir -p ./results/2x_CONTINUUM_qps1.75

if python -c "import quart" &> /dev/null; then
    echo "Quart is already installed."
else
    echo "Quart is not installed. Installing..."
    python -m pip install quart
fi

wait_for_server() {
  local port=$1
  echo "Waiting for server to start on port ${port}..."
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

LOG_FILE=./8B_results/2x_CONTINUUM_qps1.75/server_output_qps1.75.log
CUDA_VISIBLE_DEVICES=0 /jizhicfs/marvinhjia/miniconda3/envs/chenyilin-vllm/bin/python3 \
    -m vllm.entrypoints.openai.api_server \
    --model /jizhicfs/marvinhjia/sca/models/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
    -tp 1 \
    --port 8166 \
    --block-size 64 \
    --dtype float16 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-model-len 450000 \
    --max-num-partial-prefills 4 \
    --max-num-batched-tokens 8192 \
    --max-long-partial-prefills 4 \
    --gpu-memory-utilization 0.8 \
    > ${LOG_FILE} 2>&1 &

sleep 85
wait_for_server 8166
python agent_interface_continuum.py
sleep infinity
# Cleanup commands, suppressing their output
ps -e | grep pt_main_thread | awk '{print $1}' | xargs kill -9 > /dev/null 2>&1
pkill -f python > /dev/null 2>&1
