import numpy as np
from tqdm import trange
from vllm import LLM, SamplingParams
import sys
import os

assert sys.argv[1] in ['1k', '10k'], f'Invalid argument, got {sys.argv[1]}'
OUTPUT_DIR = './examples/test/output/'

llm = LLM(
    model='meta-llama/Meta-Llama-3.1-8B-Instruct',
    enable_prefix_caching=True,
    tensor_parallel_size=2,
    disable_log_stats=False,
    collect_detailed_traces='all',
)

sampling_params=SamplingParams(temperature=0, max_tokens=1)

if sys.argv[1] == '1k':
    TOTAL_LENGTH = 1024
    BASE_TOKEN_ID = 5
    MAX_TRY = 10
else:
    TOTAL_LENGTH = 1024 * 10
    BASE_TOKEN_ID = 5
    MAX_TRY = 6

target_input_ids = [BASE_TOKEN_ID] * TOTAL_LENGTH

f = open(os.path.join(OUTPUT_DIR, f'profile_cache_{sys.argv[1]}-tp2.csv'), 'w')
f.write('cached_length,' + ','.join([f'p{i+1}' for i in range(MAX_TRY)]))
f.write(',AVG\n')
llm.generate(None, 
            sampling_params=sampling_params,
            prompt_token_ids=target_input_ids,
            use_tqdm=False,
            slo_urgents=2.0) # slo_urgent=2.0 means that the token is not evicted

trange_obj = trange(0, TOTAL_LENGTH + 1, 16)
for cached_length in trange_obj:
    # trange_obj.set_postfix({'cached_length': cached_length})
    f.write(str(cached_length) + ',')
    results = []
    for i in range(MAX_TRY):
        input_ids = target_input_ids[:cached_length] + \
            [BASE_TOKEN_ID + i + 1] * (TOTAL_LENGTH - cached_length)
        outputs = llm.generate(None, 
                              sampling_params=sampling_params,
                              prompt_token_ids=input_ids,
                              use_tqdm=False)
        
        prefill_time = outputs[0].metrics.first_token_time - outputs[0].metrics.first_scheduled_time
        forward_time = outputs[0].metrics.model_forward_time
        results.append(forward_time)
        trange_obj.set_postfix({'forward_time': forward_time})
    
    f.write(','.join([str(result) for result in results]))
    results = np.array(results)
    results.sort()
    # ignore the max two and min two
    f.write(',' + str(np.mean(results[2:-2])) + '\n')
    f.flush()
f.close()