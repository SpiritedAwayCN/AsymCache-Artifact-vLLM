#!/bin/bash


# test for disagg

set -ex
benchmark() {
  results_folder="./results"
  model="/home/tzl/huggingface/modelzoo/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
  dataset_name="sonnet"
  dataset_path="../sonnet_4x.txt"
  num_prompts=500
  qps=8
  prefix_len=50
  input_len=400
  output_len=100
  tag=$3

  python3 ../benchmark_serving.py \
          --backend vllm \
          --model $model \
          --dataset-name $dataset_name \
          --dataset-path $dataset_path \
          --sonnet-input-len $input_len \
          --sonnet-output-len $output_len \
          --sonnet-prefix-len $prefix_len \
          --num-prompts $num_prompts \
          --port 8000 \
          --save-result \
          --result-dir $results_folder \
          --result-filename $tag-qps-$qps.json \
          --request-rate $qps

  sleep 2

}

main() {
  cd "$(dirname "$0")"
  cd ..
  # create sonnet-4x.txt so that we can sample 2048 tokens for input
  echo "" > sonnet_4x.txt
  for _ in {1..4}
  do
    cat sonnet.txt >> sonnet_4x.txt
  done
  cd disagg_benchmarks
  rm -rf results
  mkdir results
  default_output_len=20
  export VLLM_LOGGING_LEVEL=DEBUG
  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
  export VLLM_ALLOW_DEPRECATED_BLOCK_MANAGER_V1=1 
  for qps in 8; do
  benchmark $qps $default_output_len disagg_prefill
  done
}


main "$@"