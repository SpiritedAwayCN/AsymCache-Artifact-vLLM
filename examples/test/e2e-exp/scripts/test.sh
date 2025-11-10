source envs.sh
OUTPUT_DIR=Longbench_exp/new_four_data_experiment/qps2.0_70B_leval_simulate_logs
mkdir -p $OUTPUT_DIR

for qps in 2.0
do
    for div in 10 20 40 80 160 320
    do
        echo "Simulating lru with qps=$qps, div=$div..."
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /jizhicfs/marvinhjia/miniconda3/envs/chenyilin-vllm/bin/python3 ./new_simulator.py \
            --policy lru \
            --qps $qps \
            --div $div > $OUTPUT_DIR/Longbench_simulator_lru_qps${qps}_div${div}_h0.log 2>&1
        
        for half_life in 20 40 60 80
        do
            for policy in mwf
            do
                echo "Simulating $policy with qps=$qps, div=$div, half_life=$half_life..."
                CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HALF_LIFE=$half_life /jizhicfs/marvinhjia/miniconda3/envs/chenyilin-vllm/bin/python3 ./new_simulator.py \
                    --qps $qps \
                    --policy $policy \
                    --div $div > $OUTPUT_DIR/Longbench_simulator_${policy}_qps${qps}_div${div}_h${half_life}.log 2>&1
            done
        done
    done
done