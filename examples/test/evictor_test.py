import time
import json
import random
import logging
from vllm import LLM, SamplingParams


logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
# level=logging.INFO,
# format='%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s',
# datefmt='%m-%d %H:%M:%S

data_path = '/processing_data/infra/sca/sysbench/datas/system_benchmark_eval_datas.json'
data = json.load(open(data_path, 'r', encoding='utf-8'))

# LargeWorldModel/LWM-Text-1M
# meta-llama/Meta-Llama-3.1-8B-Instruct
llm = LLM(
    model='LargeWorldModel/LWM-Text-1M',
    enable_prefix_caching=True,
    tensor_parallel_size=2,
    disable_log_stats=False,
    enable_chunked_prefill=False,
    collect_detailed_traces='all',
    max_model_len=210854,
    gpu_memory_utilization=0.95
)

N_ENTRY = 500

request_arrive_order = list(range(N_ENTRY)) * 6
random.seed(0)
random.shuffle(request_arrive_order)
print('request_arrive_order:', request_arrive_order)

def conversation_generator(entry_id):
    messages = []
    data_entry = data[entry_id]
    
    for message in data_entry['messages']:
        if message['role'] != 'assistant':
            messages.append(message)
            continue
        
        # print(f'entry_id: {entry_id}, messages: {messages}')
        outputs = llm.chat(messages,
                           sampling_params=SamplingParams(
                               temperature=0, 
                               max_tokens=4096
                            ),
                           use_tqdm=False)
        yield outputs
        messages.append({'role': 'assistant', 'content': outputs[0].outputs[0].text})

generaters = [conversation_generator(entry_id) for entry_id in range(N_ENTRY)]
offset = 0
for i, current_id in enumerate(request_arrive_order):
    gen = generaters[current_id]
    try:
        outputs = next(gen)
    except StopIteration:
        offset += 1
        continue
    
    prefill_time = outputs[0].metrics.first_token_time - outputs[0].metrics.first_scheduled_time
    logger.info(f'[Step {i-offset}] entry_id: {current_id}, output_len: {len(outputs[0].outputs[0].token_ids)}, prefill time: {prefill_time*1000}ms, outputs: {outputs[0].metrics}')
    
