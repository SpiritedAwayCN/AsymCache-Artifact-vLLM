#!/bin/bash
export VLLM_USE_V1=0
export NCCL_ALGO=Ring
export VLLM_PORT=12356
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
export MINEXP_B=-5.108343e+02
export MINEXP_ALPHA=1.821221e+03
export MINEXP_BETA=4.553053e+01
export SCHEDULE_LOG_PATH="3x_schedule.log"

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
    until curl -s http://127.0.0.1:${port}/health > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

LOG_FILE=3x_server_output.log
# prefilling instance
CUDA_VISIBLE_DEVICES=4,5,6,7 /jizhicfs/marvinhjia/miniconda3/envs/chenyilin-vllm/bin/python3 \
    -m vllm.entrypoints.openai.api_server \
    --model /jizhicfs/marvinhjia/sca/models/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b \
    -tp 4 \
    --port 8056 \
    --block-size 64 \
    --dtype float16 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-model-len 450000 \
    --max-num-partial-prefills 4 \
    --max-long-partial-prefills 4 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.8 \
    > ${LOG_FILE} 2>&1 &

sleep 405
wait_for_server 8056
python test.py -i ../data/3x_bfcl_web_search_prompts_qps3.0.csv --api_url http://127.0.0.1:8056/v1/completions --warmup 3 --num_runs 20 --use_nonce
sleep infinity
# Cleanup commands, suppressing their output
ps -e | grep pt_main_thread | awk '{print $1}' | xargs kill -9 > /dev/null 2>&1
pkill -f python > /dev/null 2>&1
