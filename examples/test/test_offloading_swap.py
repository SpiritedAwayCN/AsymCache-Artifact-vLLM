import torch

from vllm.engine.arg_utils import EngineArgs
from vllm.sequence import ExecuteModelRequest
from vllm.utils import get_distributed_init_method, get_ip, get_open_port
from vllm.worker.worker import Worker
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

# Configure the engine.
engine_args = EngineArgs(model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                            dtype="half",
                            load_format="dummy",
    num_gpu_blocks_override=100,
    max_model_len=100*16)
engine_config = engine_args.create_engine_config()
engine_config.cache_config.num_gpu_blocks = 100
engine_config.cache_config.num_cpu_blocks = 1000

# Create the worker.
distributed_init_method = get_distributed_init_method(
    get_ip(), get_open_port())
worker = Worker(
    model_config=engine_config.model_config,
    parallel_config=engine_config.parallel_config,
    scheduler_config=engine_config.scheduler_config,
    device_config=engine_config.device_config,
    cache_config=engine_config.cache_config,
    load_config=engine_config.load_config,
    local_rank=0,
    rank=0,
    distributed_init_method=distributed_init_method,
    is_driver_worker=True,
)

# Initialize the worker.
worker.init_device()
worker.load_model()
worker.initialize_cache(
    num_gpu_blocks=engine_config.cache_config.num_gpu_blocks,
    num_cpu_blocks=engine_config.cache_config.num_cpu_blocks)

# Randomly initialize the cache.
gpu_cache = worker.cache_engine[0].gpu_cache
cpu_cache = worker.cache_engine[0].cpu_cache
num_layers = len(gpu_cache)
for i in range(num_layers):
    gpu_key_cache, gpu_value_cache = gpu_cache[i]
    gpu_key_cache.random_()
    gpu_value_cache.random_()
    cpu_key_cache, cpu_value_cache = cpu_cache[i]
    cpu_key_cache.random_()
    cpu_value_cache.random_()

allclose = lambda a, b: torch.allclose(
    a.cuda(), b.cuda(), rtol=0.0, atol=0.0)

# Test offload swap
# blocks_to_offload = [(3, 72, True), (56, 35, False), (84, 34, True)]
blocks_to_offload = [(1, 2, True), (2, 3, False), (10, 11, False), (11, 12, True)]
execute_model_req = ExecuteModelRequest(
    seq_group_metadata_list=[],
    blocks_to_swap_in=[],
    blocks_to_swap_out=[],
    blocks_to_copy=[],
    blocks_to_offload=blocks_to_offload,
)
worker.execute_model(execute_model_req=execute_model_req)

for i in range(num_layers):
    gpu_key_cache, gpu_value_cache = gpu_cache[i]
    cpu_key_cache, cpu_value_cache = cpu_cache[i]
    for src, dst, flag in blocks_to_offload:
        if flag:
            # swap_out
            assert allclose(gpu_key_cache[src], cpu_key_cache[dst])
            assert allclose(gpu_value_cache[src], cpu_value_cache[dst])
        else:
            # swap_in
            assert allclose(cpu_key_cache[src], gpu_key_cache[dst])
            assert allclose(cpu_value_cache[src], gpu_value_cache[dst])
    
    # example
    assert allclose(gpu_key_cache[1], gpu_key_cache[3])
    assert allclose(gpu_value_cache[1], gpu_value_cache[3])

    assert allclose(cpu_key_cache[10], cpu_key_cache[12])
    assert allclose(cpu_value_cache[10], cpu_value_cache[12])



