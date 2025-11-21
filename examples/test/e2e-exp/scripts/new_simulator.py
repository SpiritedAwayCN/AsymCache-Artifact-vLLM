import os
import glob
import json
import random
import argparse
from vllm.utils import Counter, is_list_of
from vllm.core.evictor import EvictionPolicy
from vllm.engine.arg_utils import EngineArgs
from vllm.core.block_manager import SelfAttnBlockSpaceManager
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.entrypoints.chat_utils import (apply_hf_chat_template, parse_chat_messages)

parser = argparse.ArgumentParser()
parser.add_argument('--policy', '-p', type=str, choices=['mwf', 'lru', 'lfu'], default='mwf')
parser.add_argument('--div', '-d', type=int, default=40)
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--qps', '-q', type=float, default=1)
arg = parser.parse_args()

if arg.policy == 'mwf':
    policy = EvictionPolicy.MWF
elif arg.policy == 'lru':
    policy = EvictionPolicy.LRU
elif arg.policy == 'lfu':
    policy = EvictionPolicy.LFU
else:
    raise ValueError(f"Invalid policy {arg.policy}")

TP = 4
block_size = 64
model = "LargeWorldModel/LWM-Text-1M"
params_dict = {4 : {'num_gpu_blocks': 7931, 'num_cpu_blocks': 819}, 8 : {'num_gpu_blocks': 21978, 'num_cpu_blocks': 1638}}

block_manager = SelfAttnBlockSpaceManager(
    block_size=block_size,
    num_gpu_blocks=7931,# params_dict[TP]['num_gpu_blocks'] * 64 // block_size,
    num_cpu_blocks=819, # params_dict[TP]['num_cpu_blocks'] * 64 // block_size,
    sliding_window=None,
    enable_caching=True,
    eviction_policy=policy,
)

seq_counter = Counter()
req_counter = Counter()

engine_args= EngineArgs(
            model=model,
            skip_tokenizer_init=False,
            enable_prefix_caching=True,
            tensor_parallel_size=TP,
            dtype='auto',
            disable_log_stats=False,
            enable_chunked_prefill=False,
            block_size=block_size,
        )

engine_config = engine_args.create_engine_config()
model_config = engine_config.model_config
tokenizer = init_tokenizer_from_configs(
                model_config=engine_config.model_config,
                scheduler_config=engine_config.scheduler_config,
                lora_config=engine_config.lora_config).tokenizer

def simulate_allocate_block(msgs, chat_template=None):
    if msgs[-1]['role'] == 'assistant':
        msgs = msgs[:-1]
    
    conversation, _ = parse_chat_messages(msgs, model_config, tokenizer, content_format="string")
   
    prompt_data = apply_hf_chat_template(
                    tokenizer,
                    conversation=conversation,
                    chat_template=chat_template,
                    add_generation_prompt=True,
                    continue_final_message=False,
                    model_config=model_config,
                    tools=None,
                )
    
    if is_list_of(prompt_data, int):
        raise ValueError("Tokenized prompt not supported")
    else:
        prompt = prompt_data
    
    seq_id = next(seq_counter)
    arrival_time = 0.5 + seq_id * arg.qps
    prompt_token_ids = tokenizer.encode(prompt)
    
    seq = Sequence(seq_id,
                   inputs={
                       "prompt": prompt,
                       "prompt_token_ids": prompt_token_ids,
                       "type": "token",
                   },
                   block_size=block_size,
                   eos_token_id=tokenizer.eos_token_id,
                   lora_request=None)
    
    seq_group = SequenceGroup(
        request_id=next(req_counter),
        seqs=[seq],
        arrival_time=arrival_time,
    )
    
    # Allocate
    block_manager.allocate(seq_group, blocks_to_offload=[])
    for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
        seq.status = SequenceStatus.RUNNING
    block_manager.access_all_blocks_in_seq(seq, arrival_time + 0.1)
   
    # Compute
    block_manager.mark_blocks_as_computed(seq_group, len(prompt_token_ids))
    block_manager.get_num_cached_tokens(seq)
    
    # Free
    for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
        seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
    block_manager.free(seq)

def shuffle_array_within_k(array, K, seed=None):
    if seed is not None:
        random.seed(seed)
    
    n = len(array)
    indexed_arr = []
    
    for i in range(n):
        new_index = i + random.randint(-K, K)
        indexed_arr.append((new_index, array[i]))
    
    indexed_arr.sort()
    shuffled_arr = [elem for _, elem in indexed_arr]
    
    return shuffled_arr
    
if __name__ == '__main__':
    import pandas as pd
    with open('/jizhicfs/marvinhjia/sca/llama-2-chat.jinja', 'r', encoding='utf-8') as file:
        chat_template = file.read()
    
    BASE_DIR = '/jizhicfs/marvinhjia/sca'
    df = pd.read_csv(f'{BASE_DIR}/data_preprocess/seqlen_stat_Longbench.csv')
    
    index_list = list(range(len(df)))
    if arg.div >= 0:
        index_list = shuffle_array_within_k(index_list, arg.div, seed=arg.seed)
    else:
        random.seed(arg.seed)
        random.shuffle(index_list)
    
    datasets = {}
    
    for jsonl_file in glob.glob(f"{BASE_DIR}/Longbench/*.jsonl", recursive=True):
        with open(jsonl_file, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file):
                data = json.loads(line)
                file_name = os.path.basename(jsonl_file)
                
                if file_name not in datasets:
                    datasets[file_name] = []
                datasets[file_name].append(data)
    
    for file_name in datasets:
        print(file_name, len(datasets[file_name]))
    
    total_session = sum(len(datasets[file_name]) for file_name in datasets)
    current_turn_map = {}    
    print(total_session)
    
    for idx in index_list:
        row = df.iloc[idx]
        file_name, line_number, turn_id = row['file'], row['line'], row['turn']
        
        current_turn_map_key = f'{file_name}:{line_number}'
        if current_turn_map_key in current_turn_map:
            current_turn_map[current_turn_map_key] += 1
        else:
            current_turn_map[current_turn_map_key] = 0
        
        data = datasets[file_name][line_number]
        msgs2 = [{'role': 'user', 'content': data['input'] + '\n\n'}]
        for idx, question in enumerate(data['instructions']):
            if idx > current_turn_map[current_turn_map_key]:
                break
            if idx == 0:
                msgs2[0]['content'] += question
            else:
                msgs2.append({'role': 'assistant', 'content': data['outputs'][idx-1]})
                msgs2.append({'role': 'user', 'content': question})
        
        simulate_allocate_block(msgs2, chat_template=chat_template)