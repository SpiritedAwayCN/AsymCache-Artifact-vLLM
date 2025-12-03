source envs.sh
export USE_MINEXP=2

# multi-turn mode
OUTPUT_DIR=./Longbench-exp/minexp_7B_longbench_simulate_logs_v3
mkdir -p $OUTPUT_DIR
declare -A seen_qps=()
for grad in 2 5 10 20 
do
    CONFIG_FILE=$grad"_config.txt"
    while read -r div qps p alpha beta A B ; do
        # if [[ -z "${seen_qps[$qps]+x}" ]]; then
        #     seen_qps[$qps]=1
        #     policy=lru
        #     echo "Simulating $policy (multi-turn mode) with div=$div, qps=$qps."
        #     CUDA_VISIBLE_DEVICES=0,1,2,3 MINEXP_A=$A MINEXP_B=$B MINEXP_ALPHA=$alpha MINEXP_BETA=$beta /jizhicfs/marvinhjia/miniconda3/envs/chenyilin-vllm/bin/python3 ./minexp_simulator.py \
        #         --qps $qps \
        #         --policy $policy \
        #         --use_model_output \
        #         --model_output_file /jizhicfs/marvinhjia/sca/data_preprocess/two_longbench_deepseek_results.jsonl \
        #         --div $div > $OUTPUT_DIR/Longbench_simulator_${policy}_qps${qps}_div${div}.log 2>&1
        # fi
        
        for policy in mwf
        do
            echo "Simulating $policy (multi-turn mode) with div=$div, qps=$qps, grad=$grad: (alpha=$alpha, beta=$beta, A=$A, B=$B)."
            CUDA_VISIBLE_DEVICES=0,1,2,3 MINEXP_A=$A MINEXP_B=$B MINEXP_ALPHA=$alpha MINEXP_BETA=$beta /jizhicfs/marvinhjia/miniconda3/envs/chenyilin-vllm/bin/python3 ./minexp_simulator.py \
                --qps $qps \
                --policy $policy \
                --use_model_output \
                --model_output_file /jizhicfs/marvinhjia/sca/data_preprocess/two_longbench_deepseek_results.jsonl \
                --div $div > $OUTPUT_DIR/Longbench_simulator_${policy}_qps${qps}_div${div}_grad10_P${p}.log 2>&1
        done
    done < "$CONFIG_FILE"
done

# prefix mode
OUTPUT_DIR=./Longbench-exp/minexp_7B_longbench_simulate_logs_prefix_v3
mkdir -p $OUTPUT_DIR
declare -A seen_qps=()
for grad in 2 5 10 20 
do
    CONFIG_FILE=$grad"_config.txt"
    while read -r div qps p alpha beta A B ; do
        # if [[ -z "${seen_qps[$qps]+x}" ]]; then
        #     seen_qps[$qps]=1
        #     policy=lru
        #     echo "Simulating $policy (prefix mode) with div=$div, qps=$qps."
        #     CUDA_VISIBLE_DEVICES=0,1,2,3 MINEXP_A=$A MINEXP_B=$B MINEXP_ALPHA=$alpha MINEXP_BETA=$beta /jizhicfs/marvinhjia/miniconda3/envs/chenyilin-vllm/bin/python3 ./minexp_simulator.py \
        #         --qps $qps \
        #         --policy $policy \
        #         --use_model_output \
        #         --model_output_file /jizhicfs/marvinhjia/sca/data_preprocess/two_longbench_deepseek_results.jsonl \
        #         --prefix_mode \
        #         --div $div > $OUTPUT_DIR/Longbench_simulator_${policy}_qps${qps}_div${div}.log 2>&1
        # fi
        
        for policy in mwf
        do
            echo "Simulating $policy (prefix mode) with div=$div, qps=$qps, grad=$grad: (alpha=$alpha, beta=$beta, A=$A, B=$B)."
            CUDA_VISIBLE_DEVICES=0,1,2,3 MINEXP_A=$A MINEXP_B=$B MINEXP_ALPHA=$alpha MINEXP_BETA=$beta /jizhicfs/marvinhjia/miniconda3/envs/chenyilin-vllm/bin/python3 ./minexp_simulator.py \
                --qps $qps \
                --policy $policy \
                --use_model_output \
                --model_output_file /jizhicfs/marvinhjia/sca/data_preprocess/two_longbench_deepseek_results.jsonl \
                --prefix_mode \
                --div $div > $OUTPUT_DIR/Longbench_simulator_${policy}_qps${qps}_div${div}_grad10_P${p}.log 2>&1
        done
    done < "$CONFIG_FILE"
done