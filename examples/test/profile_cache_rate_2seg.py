import sys
import os
import numpy as np
from tqdm import trange
from vllm import LLM, SamplingParams
import torch

# tp = 8

seqlen = int(sys.argv[1][:-1])
tp = int(sys.argv[2])
assert seqlen in range(1, 11), f'Invalid argument, got {sys.argv[1]}'
OUTPUT_DIR = '/home/pkuhetu/sca/chenyilei/v0.8/vllm-dev/examples/test/output/'
os.environ['EVCITOR_BLOCK_PROTECT_FLAG'] = '0'
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
block_manager = llm.llm_engine.scheduler[0].block_manager
print(block_manager.__class__.__name__)

sampling_params=SamplingParams(temperature=0, max_tokens=1)

if sys.argv[1] == '1k':
    TOTAL_LENGTH = 1024
    BASE_TOKEN_ID = 5
    MAX_TRY = 10
    POST_TOKEN, START_TOKEN = 32, 32 # for 1K
elif sys.argv[1] == '10k':
    TOTAL_LENGTH = 1024 * 10
    BASE_TOKEN_ID = 5
    MAX_TRY = 6
    POST_TOKEN, START_TOKEN = 128, 32 # for 10K
else:
    TOTAL_LENGTH = 1024 * seqlen
    BASE_TOKEN_ID = 5
    MAX_TRY = 6
    POST_TOKEN, START_TOKEN = 128, 32 # for 10K

target_input_ids = [BASE_TOKEN_ID] * (TOTAL_LENGTH - POST_TOKEN)

f = open(os.path.join(OUTPUT_DIR, f'profile_tmp_{sys.argv[1]}-tp{tp}.csv'), 'w')
f.write('cached_length,' + ','.join([f'p{i+1}' for i in range(MAX_TRY)]))
f.write(',AVG\n')

os.environ['EVCITOR_BLOCK_PROTECT_FLAG'] = '1'

llm.generate(None, 
            sampling_params=sampling_params,
            prompt_token_ids=target_input_ids,
            use_tqdm=False,
            slo_urgents=2.0) # slo_urgent=2.0 means that the token is not evicted

os.environ['EVCITOR_BLOCK_PROTECT_FLAG'] = '0'
print("check")
# block_manager.test_ignore_cache_block_start = START_TOKEN // block_manager.block_size - 1
block_manager.test_force_nocache_begin = START_TOKEN

trange_obj = trange(TOTAL_LENGTH - POST_TOKEN,  START_TOKEN, -16)
# output_lines = []
for i, cached_length in enumerate(trange_obj):
    # if cached_length >= 4608:
    #     continue
    output_stdout = str(cached_length) + ','
    
    results = []
    # block_manager.test_ignore_cache_block_num = \
        # (TOTAL_LENGTH - POST_TOKEN - cached_length) // block_manager.block_size
    block_manager.test_force_nocache_end = TOTAL_LENGTH - POST_TOKEN - cached_length + START_TOKEN
    block_manager.test_set_force_nocache(START_TOKEN, TOTAL_LENGTH - POST_TOKEN - cached_length + START_TOKEN)
    print(f'block_manager.test_force_nocache_end: {block_manager.test_force_nocache_end}')
    print(f'block_manager.test_force_nocache_begin: {block_manager.test_force_nocache_begin}')
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