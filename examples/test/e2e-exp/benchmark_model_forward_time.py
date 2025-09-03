import numpy as np
from tqdm import trange
from vllm import LLM, SamplingParams
import torch
import sys
import pandas as pd
import os
from vllm.utils import Device
import csv

chunk_sizes = [2**i for i in range(7,14)]
prefill_len = 2**14 # 2**14
d_len_list = [2**i for i in range(7, 15)]
d_batch_sizes = [0, 1, 2, 4, 8, 16, 32, 64]
# seqlen = int(sys.argv[1][:-1])
tp = int(sys.argv[1])
OUTPUT_DIR = '/home/pkuhetu/sca/chenyilei/v0.8/vllm-dev/examples/test/output/'
os.environ['EVCITOR_BLOCK_PROTECT_FLAG'] = '0'
llm = LLM(
    model='LargeWorldModel/LWM-Text-1M',
    enable_prefix_caching=True,
    # enable_chunked_prefill=True,
    tensor_parallel_size=tp,
    disable_log_stats=False,
    collect_detailed_traces=['all'],
    max_model_len=8192*2+3,
    block_size=64,
    dtype='float16',
    gpu_memory_utilization=0.8,
)
scheduler = llm.llm_engine.scheduler[0]
block_manager = llm.llm_engine.scheduler[0].block_manager
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
repeat = 6

target_input_ids = [5] * prefill_len
def get_prompt_ids(id, last_id, len: int):
    ret = [id]*(len-1)+[last_id]
    return ret
count = 7
from tqdm import tqdm
for d_len in d_len_list:
    dir = f"d_len={d_len}"
    if not os.path.exists(dir):
        os.makedirs(dir)
    results = {}
    means = {}
    for bs in d_batch_sizes:
        with open("progress.txt", "a") as f:
            f.write(f'd_len={d_len}, bs={bs}\n')
        results[bs] = {}
        means[bs] = {}
        tot = []
        for chunk_size in chunk_sizes:
            results[bs][chunk_size] = {}
            means[bs][chunk_size] = {}
            for _ in range(repeat):
                block_manager.block_allocator._allocators[Device.GPU].evictor.protect_blocks.clear()
                llm.reset_prefix_cache()
                os.environ['EVCITOR_BLOCK_PROTECT_FLAG'] = '1'
                llm.generate(None, 
                    sampling_params=sampling_params,
                    prompt_token_ids=target_input_ids,)
                llm.generate(None,
                    sampling_params=sampling_params,
                    prompt_token_ids=get_prompt_ids(6, 6, d_len-64))
                os.environ['EVCITOR_BLOCK_PROTECT_FLAG'] = '0'
                for chunk_idx in range(prefill_len // chunk_size):
                    if bs > 0:
                        # add decode request
                        for i in range(bs):
                            llm.llm_engine.add_request(
                                request_id=f"{d_len}_{bs}_{chunk_size}_{chunk_idx}_{_}_decode_{i}",
                                prompt={"prompt_token_ids": get_prompt_ids(6, chunk_idx*bs+i+13, d_len)},
                                params=SamplingParams(temperature=0, max_tokens=2),
                            )
                        scheduler.use_simple_schedule = False
                        out = llm.llm_engine.step()  # Prefill
                        assert llm.llm_engine.has_unfinished_requests() == True
                    
                    scheduler.use_simple_schedule = True
                    count += 1
                    if count >= 32000:
                        count = 7
                    llm.llm_engine.add_request(
                        request_id=f"{d_len}_{bs}_{chunk_size}_{_}_{chunk_idx}_prefill",
                        prompt={"prompt_token_ids": target_input_ids[:chunk_idx*chunk_size]+[count]*chunk_size},
                        params=sampling_params,
                    )
                    outputs = llm.llm_engine.step()
                    # print("###:",len(outputs))
                    # print(outputs)
                    outputs = llm.llm_engine.step()
                    # print("$$$:", len(outputs))
                    assert "prefill" in outputs[0].request_id
                    forward_time = outputs[0].metrics.model_forward_time
                    # print(f"time={forward_time}")
                    if chunk_idx+1 not in results[bs][chunk_size]:
                        results[bs][chunk_size][chunk_idx+1] = []
                    results[bs][chunk_size][chunk_idx+1].append(forward_time)
                    tot.append(forward_time)
                    assert llm.llm_engine.has_unfinished_requests() == False
            
            for chunk_idx in range(prefill_len // chunk_size):
                tmp = np.array(results[bs][chunk_size][chunk_idx+1])
                tmp.sort()
                means[bs][chunk_size][chunk_idx+1] = np.mean(tmp[2:-2])

        with open('results_tot', 'a', newline='') as file:
            writer = csv.writer(file)
            for i in range(0, len(tot), repeat):
                writer.writerow(tot[i:i+repeat])  # 自动用逗号分隔
    all_data = []
    # 为每个bs创建并保存表格
    for bs, chunk_data in means.items():
        df = pd.DataFrame.from_dict(chunk_data, orient='index')
        df = df.sort_index(axis=0)  # 行按chunk_size排序
        df = df.sort_index(axis=1)  # 列按chunk_num排序
        df.index.name = 'chunk_size'
        df.columns.name = 'chunk_num'

        # filename = f"latency_bs{bs}.csv"
        # df.to_csv(filename)
        marker_row = pd.DataFrame([[bs] + [""] * (df.shape[1] - 1)])
        combined = pd.concat([marker_row, df], ignore_index=True)

        all_data.append(combined)
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(f"./{dir}/latency_16k_tp={tp}.csv", index=False, header=False)