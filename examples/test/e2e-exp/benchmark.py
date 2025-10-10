import numpy as np
from tqdm import trange
from vllm import LLM, SamplingParams
import torch
import sys
import os

tp = 2
chunk_sizes = [2**i for i in range(7,14)]
prefill_len = 2**14
d_len_list = [10]#[2**i for i in range(7, 16)]
d_batch_sizes = [1]#, 1, 2, 4, 8, 16, 32, 64]

# seqlen = int(sys.argv[1][:-1])
# tp = int(sys.argv[2])
# assert seqlen in range(1, 11), f'Invalid argument, got {sys.argv[1]}'
# OUTPUT_DIR = '/home/pkuhetu/sca/chenyilei/v0.8/vllm-dev/examples/test/output/'
# os.environ['EVCITOR_BLOCK_PROTECT_FLAG'] = '0'
llm = LLM(
    model='LargeWorldModel/LWM-Text-1M',
    enable_prefix_caching=True,
    # enable_chunked_prefill=True,
    tensor_parallel_size=tp,
    disable_log_stats=False,
    collect_detailed_traces=['all'],
    max_model_len=20000,
    block_size=64,
    dtype='float16'
)
scheduler = llm.llm_engine.scheduler[0]

sampling_params=SamplingParams(temperature=0, max_tokens=1)

# if sys.argv[1] == '1k':
#     TOTAL_LENGTH = 1024
#     BASE_TOKEN_ID = 5
#     MAX_TRY = 10
# elif sys.argv[1] == '10k':
#     TOTAL_LENGTH = 1024 * 10
#     BASE_TOKEN_ID = 5
#     MAX_TRY = 6
# else:
#     TOTAL_LENGTH = 1024 * seqlen
#     BASE_TOKEN_ID = 5
#     MAX_TRY = 6


target_input_ids = [5] * prefill_len
llm.generate(None, 
            sampling_params=sampling_params,
            prompt_token_ids=target_input_ids,
            use_tqdm=True)

def random_prompt_ids(len: int):
    ret = []
    import random
    for _ in range(len):
        ret.append(random.randint(10, 3000))
    return ret
for d_len in d_len_list:
    for bs in d_batch_sizes:
        for chunk_size in chunk_sizes:
            llm.reset_prefix_cache()
            
            for chunk_idx in range(prefill_len // chunk_size):
                if bs > 0:
                    # add decode request
                    os.environ['VLLM_USE_SIMPLE_SCHEDULE'] = '0'
                    for i in range(bs):
                        llm.llm_engine.add_request(
                            request_id=f"{d_len}_{bs}_{chunk_size}_{chunk_idx}_decode_{i}",
                            prompt={"prompt_token_ids": random_prompt_ids(d_len)},
                            params=sampling_params,
                        )
                    outputs = llm.llm_engine.step()  # Prefill
                    print(f'output={outputs}')

                os.environ['VLLM_USE_SIMPLE_SCHEDULE'] = '1'
                llm.llm_engine.add_request(
                    request_id=f"{d_len}_{bs}_{chunk_size}_{chunk_idx}_prefill",
                    prompt={"prompt_token_ids": target_input_ids[:(chunk_idx+1)*chunk_size]},
                    params=sampling_params,
                )
                outputs = llm.llm_engine.step()
                print(f'output={outputs}')
                outputs = llm.llm_engine.step()
                print(f'output={outputs}')
                forward_time = outputs[0].metrics.model_forward_time
                print(forward_time)
                exit(0)

