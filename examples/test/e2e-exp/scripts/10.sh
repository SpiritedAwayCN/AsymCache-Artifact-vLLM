source envs.sh
export USE_MINEXP=1
CONFIG_FILE="config.txt"
OUTPUT_DIR=./Longbench-exp/minexp_7B_longbench_simulate_logs
mkdir -p $OUTPUT_DIR
while read -r div qps p alpha beta A B ; do
    for policy in mwf
    do
        echo "Simulating $policy with div=$div, qps=$qps, alpha=$alpha, beta=$beta, A=$A, B=$B"
        CUDA_VISIBLE_DEVICES=0,1,2,3 MINEXP_A=$A MINEXP_B=$B MINEXP_ALPHA=$alpha MINEXP_BETA=$beta /jizhicfs/marvinhjia/miniconda3/envs/chenyilin-vllm/bin/python3 ./minexp_simulator.py \
            --qps $qps \
            --policy $policy \
            --use_model_output \
            --model_output_file /jizhicfs/marvinhjia/sca/data_preprocess/two_longbench_deepseek_results.jsonl \
            --div $div > $OUTPUT_DIR/Longbench_simulator_${policy}_qps${qps}_div${div}_grad10_P${p}.log 2>&1
    done
done < "$CONFIG_FILE"