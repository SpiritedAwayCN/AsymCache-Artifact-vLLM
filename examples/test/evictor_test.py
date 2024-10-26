import time
import json
import random
from vllm import LLM, SamplingParams

llm = LLM(
    model='meta-llama/Meta-Llama-3.1-8B-Instruct',
    enable_prefix_caching=True,
    tensor_parallel_size=1,
    disable_log_stats=False,
    # collect_detailed_traces='all',
)

data_path = '/home/sca/sysbench/datas/system_benchmark_eval_datas.json'
data = json.load(open(data_path, 'r'))

N_ENTRY = 4

request_arrive_order = list(range(N_ENTRY)) * 5
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

for i, current_id in enumerate(request_arrive_order):
    gen = generaters[current_id]
    outputs = next(gen)
    
    prefill_time = outputs[0].metrics.first_token_time - outputs[0].metrics.first_scheduled_time
    print(f'[Step {i}] entry_id: {current_id}, prefill time: {prefill_time*1000}ms, outputs: {outputs[0].metrics}')
    