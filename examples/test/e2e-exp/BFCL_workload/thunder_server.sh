#!/bin/bash
export VLLM_USE_V1=0
export NCCL_ALGO=Ring
export VLLM_PORT=12355
export VLLM_USE_POD_ATTN=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=INFO
export HF_ENDPOINT=https://hf-mirror.com
export VLLM_ALLOW_DEPRECATED_BLOCK_MANAGER_V1=1
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
export HF_HOME=/jizhicfs/marvinhjia/sca/models/huggingface
export CSET_DIR=/jizhicfs/marvinhjia/sca/yilin/vllm-dev/csrc/c_set
export HUGGINGFACE_HUB_CACHE=/jizhicfs/marvinhjia/sca/models/huggingface/hub

export USE_MINEXP=0
export VLLM_EVICTOR_POLICY=lru
export MINEXP_A=0
export MINEXP_B=-2.010611e+01
export MINEXP_ALPHA=7.124777e+03
export MINEXP_BETA=1.781194e+02
export BASE_LOG_PATH="./results/3x_THUNDER_qps3.0"
export SCHEDULE_LOG_PATH="$BASE_LOG_PATH/schedule_qps3.0.log"

mkdir -p $BASE_LOG_PATH

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
    until curl -s ${VLLM_HOST_IP}:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

LOG_FILE=$BASE_LOG_PATH/server_output_qps3.0.log
# prefilling instance
CUDA_VISIBLE_DEVICES=4,5,6,7 /jizhicfs/marvinhjia/miniconda3/envs/sca-vllm-dev/bin/python3 \
    -m vllm.entrypoints.openai.api_server \
    --model /jizhicfs/marvinhjia/sca/models/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b \
    -tp 4 \
    --port 8243 \
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

# sleep 450
# wait_for_server 8243

# # set path to ../ and run ThunderAgent
# cd ..
# python -m ThunderAgent --backend-type vllm --backends http://localhost:8243 --port 9022 \
#     > $BASE_LOG_PATH/thunder.log 2>&1 &

# # return to current directory and run the agent interface
# cd BFCL_workload

# wait_for_server 9022
# python agent_interface_thunder.py

sleep infinity
# Cleanup commands, suppressing their output
ps -e | grep pt_main_thread | awk '{print $1}' | xargs kill -9 > /dev/null 2>&1
pkill -f python > /dev/null 2>&1
