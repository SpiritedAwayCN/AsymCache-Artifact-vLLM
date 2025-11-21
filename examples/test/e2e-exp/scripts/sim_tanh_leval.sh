source envs.sh
export USE_TANH=1

OUTPUT_DIR=Leval_exp/new_four_data_experiment/70B_leval_simulate_logs
mkdir -p $OUTPUT_DIR

for qps in 0.02
do
    for div in 40
    do
        # echo "Simulating lru with qps=$qps, div=$div..."
        # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./new_simulator.py \
        #     --policy lru \
        #     --qps $qps \
        #     --div $div > $OUTPUT_DIR/Longbench_simulator_lru_qps${qps}_div${div}_h0.log 2>&1
        for life_span in 100
            for half_life in 20
            do
                for policy in mwf
                do
                    echo "Simulating $policy with qps=$qps, div=$div, life_span=$life_span, half_life=$half_life..."
                    TANH_DELTA=$life_span TANH_THETA=$half_life python ./tanh_simulator.py \
                        --dataset leval \
                        --qps $qps \
                        --policy $policy \
                        --div $div > $OUTPUT_DIR/Leval_simulator_${policy}_qps${qps}_div${div}_lf${life_span}_th${half_life}.log 2>&1
                done
            done
        done
    done
done