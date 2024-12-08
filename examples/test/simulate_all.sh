source examples/test/envs.sh

OUTPUT_DIR=examples/test/leval_simulate_all
mkdir -p $OUTPUT_DIR

for qps in 1.0
do
    for div in 10 20 40 80 160 320 640 1280 -1
    do
        echo "Simulating lru with qps=$qps, div=$div..."
        python examples/test/simulator_leval.py \
            --policy lru \
            --qps $qps \
            --div $div > $OUTPUT_DIR/LEval_simulator_lru_qps${qps}_div${div}_h0.log 2>&1
        
        for half_life in 40 20 10 5 2 1
        do
            for policy in mwf lfu
            do
                echo "Simulating $policy with qps=$qps, div=$div, half_life=$half_life..."
                HALF_LIFE=$half_life python examples/test/simulator_leval.py \
                    --policy $policy \
                    --qps $qps \
                    --div $div > $OUTPUT_DIR/LEval_simulator_${policy}_qps${qps}_div${div}_h${half_life}.log 2>&1
            done
        done
    done
done