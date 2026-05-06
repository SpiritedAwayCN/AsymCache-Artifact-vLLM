import math
import json
import hashlib
import pandas as pd
from transformers import AutoTokenizer


datasets = {}
BLOCK_SIZE = 64
unique_blocks = set()
model = "meta-llama/Llama-3.1-70B-Instruct"
df = pd.read_csv("seqlen_stat_sharegpt.csv")
tokenizer = AutoTokenizer.from_pretrained(model)

with open("/jizhicfs/marvinhjia/sca/sharegpt_500.jsonl", 'r', encoding='utf-8') as file: 
    file_name = "sharegpt"
    for line_number, line in enumerate(file): 
        data = json.loads(line)
        if file_name not in datasets:
            datasets[file_name] = []
        datasets[file_name].append(data)

for _, row in df.iterrows():
    file_name = row['file']
    line_number = int(row['line'])
    turn_id = int(row['turn'])
    data = datasets[file_name][line_number]
    msgs2 = []
    for idx, question in enumerate(data['instructions'][:turn_id + 1]):
        answer_text = data['outputs'][idx]
        msgs2.append({'role': 'user', 'content': question})
        msgs2.append({'role': 'assistant', 'content': answer_text})
    if msgs2[-1]['role'] != 'assistant':
        continue
    prompt = tokenizer.apply_chat_template(
        msgs2[:-1],
        tokenize=False,
        add_generation_prompt=True
    )

    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_len = len(tokens)
    num_blocks = math.ceil(prompt_len / BLOCK_SIZE)
    for i in range(num_blocks):
        end = min((i + 1) * BLOCK_SIZE, prompt_len)
        prefix_tokens = tokens[:end]
        hash_str = hashlib.md5(
            str(prefix_tokens).encode()
        ).hexdigest()
        unique_blocks.add(hash_str)

print("Total unique blocks:", len(unique_blocks))
if len(unique_blocks) >= 7893 * 2:
    print("✅")
else:
    print("❌")