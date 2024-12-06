#!/bin/bash


# Model: neuralmagic/Meta-Llama-3-70B-Instruct-FP8-KV 
# Query: 2048 input tokens, 11 output tokens, QPS 4, 500 requests
# Resource: 8x H100
# Approaches:
# 1. Chunked prefill: 1 vllm instance with tp=8
# 2. Chunked prefill: 2 vllm instance with tp=4, equivalent to 1 tp=4 instance with QPS 4
# 3. Disaggregated prefill: 1 prefilling instance and 1 decoding instance
# Prefilling instance: max_output_token=1
# Decoding instance: force the input tokens be the same across requests to bypass prefilling

set -ex

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  local port=$1
  timeout 1200 bash -c "
    until curl -s ${VLLM_HOST_IP}:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

launch_disagg_prefill() {
  echo "launching disagg prefill"
  model="/home/tzl/huggingface/modelzoo/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" 
  # disagg prefill
  # optional for test:
  # --disable-async-output-proc
  # --enable-prefix-caching
  VLLM_PORT=12346 VLLM_DISTRIBUTED_KV_ROLE=producer CUDA_VISIBLE_DEVICES=0 python3 \
      -m vllm.entrypoints.openai.api_server \
      --model $model \
      --port 8100 \
      --max-model-len 10000 \
      --disable-log-stats \
      --disable-log-requests \
      --disable-async-output-proc \
      --gpu-memory-utilization 0.8 &> prefill.txt &
  VLLM_PORT=12346 VLLM_DISTRIBUTED_KV_ROLE=consumer CUDA_VISIBLE_DEVICES=1 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8200 \
    --max-model-len 10000 \
    --disable-log-stats \
    --disable-log-requests \
    --disable-async-output-proc \
    --gpu-memory-utilization 0.8 &> decode.txt &
  wait_for_server 8100
  wait_for_server 8200
  # python3 disagg_prefill_proxy_server.py &
  python3 disagg_independent_server.py &
  sleep 1
}



main() {
  default_output_len=20
  export VLLM_LOGGING_LEVEL=DEBUG
  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
  export VLLM_ALLOW_DEPRECATED_BLOCK_MANAGER_V1=1 
  # export CUDA_LAUNCH_BLOCKING=1
  launch_disagg_prefill
}

main "$@"