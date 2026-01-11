import os
import glob
import json
import pandas as pd
from transformers import AutoTokenizer

use_model_output = True
model = "meta-llama/Llama-3.1-70B-Instruct"
output_jsonl = "leval_prompt_token_stat.jsonl"
tokenizer = AutoTokenizer.from_pretrained(model)
dataset_dir = "/jizhicfs/marvinhjia/sca/LEval-data"
df = pd.read_csv("/jizhicfs/marvinhjia/sca/data_preprocess/seqlen_stat_LEval.csv")
model_output_file = "/jizhicfs/marvinhjia/sca/data_preprocess/leval_deepseek_results.jsonl"

datasets = {}
for jsonl_file in glob.glob(f"{dataset_dir}/**/*.jsonl", recursive=True):
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file):
            data = json.loads(line)
            file_name = os.path.basename(jsonl_file)
            if file_name not in datasets:
                datasets[file_name] = []
            datasets[file_name].append(data)

model_outputs = {}
if use_model_output and model_output_file:
    with open(model_output_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            key = (item['file'], item['line'], item['turn'])
            model_outputs[key] = item['generate_text']
    print(f"[INFO] Loaded {len(model_outputs)} model-generated outputs from {model_output_file}")


def generate_prompt_and_answer(conversations):
    prompt, answer_token_ids = None, None
    if conversations[-1]['role'] == 'assistant':
        prompt = tokenizer.apply_chat_template(conversations[:-1], tokenize=False, add_generation_prompt=True)
        prompt_with_answer = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
        encoded_prompt = tokenizer.encode(prompt)
        encoded_prompt_with_answer = tokenizer.encode(prompt_with_answer)
        answer_token_ids = encoded_prompt_with_answer[len(encoded_prompt): ]
    return prompt, answer_token_ids


with open(output_jsonl, "w", encoding="utf-8") as fout:
    for _, row in df.iterrows():
        file_name, line_number, turn_id = row['file'], row['line'], row['turn']
        data = datasets[file_name][line_number]

        msgs2 = [{'role': 'user', 'content': data['input'] + '\n\n'}]
        for idx, question in enumerate(data['instructions'][:turn_id + 1]):
            if use_model_output:
                gen_key = (file_name, line_number, idx)
                answer_text = model_outputs.get(gen_key, data['outputs'][idx])
            else:
                answer_text = data['outputs'][idx]
            if idx == 0:
                msgs2[0]['content'] += question
                msgs2.append({'role': 'assistant', 'content': answer_text})            
            else:
                msgs2.append({'role': 'user', 'content': question})
                msgs2.append({'role': 'assistant', 'content': answer_text})

        prompt, answer = generate_prompt_and_answer(msgs2)
        prompt_len = len(tokenizer.encode(prompt))
        record = {
            "file": file_name,
            "line": int(line_number),
            "turn": int(turn_id),
            "prompt_tokens": int(prompt_len)
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")