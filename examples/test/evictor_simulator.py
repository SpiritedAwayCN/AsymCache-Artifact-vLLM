"""Test hashing of cache blocks.

Run `pytest tests/test_cache_block_hashing.py`.
"""
from typing import List, Optional
import time
import json
import random

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

TP = 4
block_size = 64

model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
params_dict = { # TP -> num_blocks (with block_size=16), A800 80GB
    1 : {'num_gpu_blocks': 20621, 'num_cpu_blocks': 2048},
    2 : {'num_gpu_blocks': 53583, 'num_cpu_blocks': 4096},
}

block_manager = BlockSpaceManagerV1(
    block_size=block_size,
    num_gpu_blocks=params_dict[TP]['num_gpu_blocks'] * 16 // block_size,
    num_cpu_blocks=params_dict[TP]['num_cpu_blocks'] * 16 // block_size,
    sliding_window=None,
    enable_caching=True,
    evictor_policy=EvictionPolicy.LRU3,
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


data_path = '/processing_data/infra/sca/sysbench/datas/system_benchmark_eval_datas.json'
data = json.load(open(data_path, 'r', encoding='utf-8'))
N_ENTRY = 500

request_arrive_order = list(range(N_ENTRY)) * 6
random.seed(0)
random.shuffle(request_arrive_order)
print('request_arrive_order:', request_arrive_order)

def simulate_allocate_block(msgs):
    assert msgs[-1]['role'] == 'assistant'
    
    conversation, _ = parse_chat_messages(
                msgs[:-1], model_config, tokenizer)
    # print(msgs, conversation)
    prompt_data = apply_hf_chat_template(
                    tokenizer,
                    conversation=conversation,
                    chat_template=None,
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

def conversation_generator(entry_id):
    messages = []
    data_entry = data[entry_id]
    
    for message in data_entry['messages']:
        messages.append(message)
        if message['role'] == 'assistant':
            simulate_allocate_block(messages)
            yield messages

generaters = [conversation_generator(entry_id) for entry_id in range(N_ENTRY)]
offset = 0
for i, current_id in enumerate(request_arrive_order):
    gen = generaters[current_id]
    try:
        outputs = next(gen)
    except StopIteration:
        offset += 1
        continue