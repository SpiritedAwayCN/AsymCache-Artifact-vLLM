import sys
import os
import numpy as np
from tqdm import trange
from vllm import LLM, SamplingParams
import torch

assert sys.argv[1] in ['1k', '5k', '10k', '20k'], f'Invalid argument, got {sys.argv[1]}'
START_TOKEN = int(sys.argv[2])
assert START_TOKEN % 16 == 0, f'Invalid argument, got {START_TOKEN}'

OUTPUT_DIR = './examples/test/output/'

llm = LLM(
    model='meta-llama/Meta-Llama-3.1-8B-Instruct',
    enable_prefix_caching=True,
    tensor_parallel_size=4,
    disable_log_stats=False,
    collect_detailed_traces='all',
)
block_manager = llm.llm_engine.scheduler[0].block_manager
print(block_manager.__class__.__name__)

sampling_params=SamplingParams(temperature=0, max_tokens=1)

if sys.argv[1] == '1k':
    TOTAL_LENGTH = 1024
    BASE_TOKEN_ID = 5
    MAX_TRY = 6
    POST_TOKEN = 32 # for 1K
elif sys.argv[1] == '5k':
    TOTAL_LENGTH = 1024 * 5
    BASE_TOKEN_ID = 5
    MAX_TRY = 6
    POST_TOKEN = 128
elif sys.argv[1] == '10k':
    TOTAL_LENGTH = 1024 * 10
    BASE_TOKEN_ID = 5
    MAX_TRY = 6
    POST_TOKEN = 128
elif sys.argv[1] == '20k':
    TOTAL_LENGTH = 1024 * 20
    BASE_TOKEN_ID = 5
    MAX_TRY = 6
    POST_TOKEN = 128

target_input_ids = [BASE_TOKEN_ID] * (TOTAL_LENGTH - POST_TOKEN)

f = open(os.path.join(OUTPUT_DIR, f'profile_cache_2seg_{sys.argv[1]}-tp4.csv'), 'w')
f.write('cached_length,' + ','.join([f'p{i+1}' for i in range(MAX_TRY)]))
f.write(',AVG\n')
llm.generate(None, 
            sampling_params=sampling_params,
            prompt_token_ids=target_input_ids,
            use_tqdm=False,
            slo_urgents=2.0) # slo_urgent=2.0 means that the token is not evicted


block_manager.test_ignore_cache_block_start = START_TOKEN // block_manager.block_size - 1

trange_obj = trange(TOTAL_LENGTH - POST_TOKEN,  START_TOKEN, -16)
# output_lines = []
for i, cached_length in enumerate(trange_obj):
    # if cached_length >= 4608:
    #     continue
    output_stdout = str(cached_length) + ','
    
    results = []
    block_manager.test_ignore_cache_block_num = \
        (TOTAL_LENGTH - POST_TOKEN - cached_length) // block_manager.block_size
    print(f'block_manager.test_ignore_cache_block_num: {block_manager.test_ignore_cache_block_num}')
    print(f'block_manager.test_ignore_cache_block_start: {block_manager.test_ignore_cache_block_start}')
    for j in range(MAX_TRY):
        input_ids = target_input_ids + [(i * MAX_TRY + j) % 20000 + 5] * POST_TOKEN
        with torch.cuda.nvtx.range(f'cache_len={cached_length},try={j}'):
            outputs = llm.generate(None, 
                                sampling_params=sampling_params,
                                prompt_token_ids=input_ids,
                                use_tqdm=False)
        
        prefill_time = outputs[0].metrics.first_token_time - outputs[0].metrics.first_scheduled_time
        forward_time = outputs[0].metrics.model_forward_time
        results.append(forward_time)
        trange_obj.set_postfix({'forward_time': forward_time,
                                'cached_length': cached_length})
    
    output_stdout += ','.join([str(result) for result in results])
    results = np.array(results)
    results.sort()
    # ignore the max two and min two
    output_stdout += ',' + str(np.mean(results[1:-1])) + '\n'
    f.write(output_stdout)
    f.flush()