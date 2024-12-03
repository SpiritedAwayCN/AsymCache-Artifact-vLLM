import time
import json
import random
import os
import glob
import sys

from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.transformers_utils.tokenizer_group import TokenizerGroup
from vllm.core.block_manager_v1 import BlockSpaceManagerV1
from vllm.core.evictor_v1 import EvictionPolicy

from vllm.utils import Counter, is_list_of
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.chat_utils import (apply_hf_chat_template,
                                         parse_chat_messages)
from vllm.inputs import TextPrompt, TokensPrompt
from vllm.transformers_utils.tokenizer_group import (
    BaseTokenizerGroup, init_tokenizer_from_configs)


policy = 'mwf' if len(sys.argv) < 2 else sys.argv[1]
if policy == 'mwf':
    policy = EvictionPolicy.MWF
elif policy == 'lru':
    policy = EvictionPolicy.LRU2
elif policy == 'lfu':
    policy = EvictionPolicy.LFU
else:
    raise ValueError(f"Invalid policy {policy}")

TP = 4
block_size = 64

# model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = "LargeWorldModel/LWM-Text-1M"
params_dict = { # TP -> num_blocks (with block_size=16), A800 80GB
    1 : {'num_gpu_blocks': 20621, 'num_cpu_blocks': 2048},
    # 2 : {'num_gpu_blocks': 53583, 'num_cpu_blocks': 4096},
    2 : {'num_gpu_blocks': 13188, 'num_cpu_blocks': 4096},
}

block_manager = BlockSpaceManagerV1(
    block_size=block_size,
    num_gpu_blocks=7204,# params_dict[TP]['num_gpu_blocks'] * 16 // block_size,
    num_cpu_blocks=512, # params_dict[TP]['num_cpu_blocks'] * 16 // block_size,
    sliding_window=None,
    enable_caching=True,
    evictor_policy=policy,
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
            # max_model_len=262144,
            # collect_detailed_traces='all',
        )
engine_config = engine_args.create_engine_config()

model_config = engine_config.model_config
tokenizer = init_tokenizer_from_configs(
                model_config=engine_config.model_config,
                scheduler_config=engine_config.scheduler_config,
                parallel_config=engine_config.parallel_config,
                enable_lora=bool(engine_config.lora_config)).tokenizer

def simulate_allocate_block(msgs, chat_template=None):
    if msgs[-1]['role'] == 'assistant':
        msgs = msgs[:-1]
    
    conversation, _ = parse_chat_messages(
                msgs, model_config, tokenizer)
    # print(msgs, conversation)
    prompt_data = apply_hf_chat_template(
                    tokenizer,
                    conversation=conversation,
                    chat_template=chat_template,
                    add_generation_prompt=True,
                    continue_final_message=False,
                )
    if is_list_of(prompt_data, int):
        raise ValueError("Tokenized prompt not supported")
    else:
        # prompt = TextPrompt(prompt=prompt_data).prompt
        prompt = prompt_data
    
    seq_id = next(seq_counter)
    arrival_time = time.time() + seq_id * 100
    prompt_token_ids = tokenizer.encode(prompt)
    # output_token_ids = tokenizer.encode(outputs)
    
    # print(len(prompt_token_ids), prompt_token_ids, sep='\t')
    
    seq = Sequence(seq_id,
                   inputs={
                       "prompt": prompt,
                       "prompt_token_ids": prompt_token_ids,
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
    block_manager.allocate(seq_group)
    for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
        seq.status = SequenceStatus.RUNNING
    block_manager.access_all_blocks_in_seq(seq, arrival_time + 10)
    
    # Compute
    block_manager.mark_blocks_as_computed(seq_group, len(prompt_token_ids))
    
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
    
    shuffled_arr = [elem for index, elem in indexed_arr]
    
    return shuffled_arr
    
if __name__ == '__main__':
    import pandas as pd
    with open('./examples/test/llama-2-chat.jinja', 'r', encoding='utf-8') as file:
        chat_template = file.read()
    
    BASE_DIR = '/processing_data/infra/sca/dataset/LEval'
    df = pd.read_csv(f'{BASE_DIR}/seqlen_stat_LEval.csv')
    
    index_list = list(range(len(df)))
    index_list = shuffle_array_within_k(index_list, 20, seed=0)
    # random.seed(0)
    # random.shuffle(index_list)
    
    datasets = {}
    
    for jsonl_file in glob.glob(f"{BASE_DIR}/LEval-data/**/*.jsonl", recursive=True):
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
                
    print(index_list, total_session)
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
        
        # if policy != EvictionPolicy.LRU2:
        #     time.sleep(1 + random.random() * 2)
    
    
    print('=' * 80)
    print(f'free table blocks at the end (len={len(block_manager.gpu_allocator.evictor.free_table)})')
    
    for block in block_manager.gpu_allocator.evictor.free_table.values():
        print(block)