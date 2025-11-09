source envs.sh

for qps in 0.5
do
    for div in 10 20 40 80 160 320
    do
        for policy in lru
        do
            echo "Simulating lru with qps=$qps, div=$div..."
            POLICY=$policy python ./leval_interface.py \
                --half_life 0 \
                --policy $policy \
                --qps $qps \
                --div $div
        done

        for half_life in 40 20 10 5 1
        do
            for policy in mwf lfu
            do
                echo "Simulating $policy with qps=$qps, div=$div, half_life=$half_life..."
                HALF_LIFE=$half_life POLICY=$policy python ./leval_interface.py \
                    --half_life $half_life \
                    --policy $policy \
                    --qps $qps \
                    --div $div
            done
        done
    done
done