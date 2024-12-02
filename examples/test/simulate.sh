source examples/test/envs.sh

python examples/test/simulator_leval.py lru > LEval_simulator_lru.log 2>&1
python examples/test/simulator_leval.py lfu > LEval_simulator_lfu.log 2>&1
