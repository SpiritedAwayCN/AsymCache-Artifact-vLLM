import json
import pandas as pd
from transformers import AutoTokenizer


datasets = {}
model = "meta-llama/Llama-3.1-70B-Instruct"
output_jsonl = "gpt_prompt_token_stat.jsonl"
df = pd.read_csv("seqlen_stat_sharegpt.csv")
tokenizer = AutoTokenizer.from_pretrained(model)


with open("/jizhicfs/marvinhjia/sca/sharegpt_500.jsonl", 'r', encoding='utf-8') as file:
    file_name = "sharegpt"
    for line_number, line in enumerate(file):
        data = json.loads(line)    
        if file_name not in datasets:
            datasets[file_name] = []
        datasets[file_name].append(data)


def generate_prompt_and_answer(conversations):
    if conversations[-1]['role'] != 'assistant':
        return None
    prompt = tokenizer.apply_chat_template(
        conversations[:-1],
        tokenize=False,
        add_generation_prompt=True
    )
    prompt_token_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False
    )
    return len(prompt_token_ids)


with open(output_jsonl, "w", encoding="utf-8") as fout:
    for _, row in df.iterrows():
        file_name, line_number, turn_id = row['file'], row['line'], row['turn']
        data = datasets[file_name][line_number]
        msgs2 = []
        for idx, question in enumerate(data['instructions'][:turn_id + 1]):
            answer_text = data['outputs'][idx]
            msgs2.append({'role': 'user', 'content': question})
            msgs2.append({'role': 'assistant', 'content': answer_text})

        prompt_len = generate_prompt_and_answer(msgs2)
        record = {
            "file": file_name,
            "line": int(line_number),
            "turn": int(turn_id),
            "prompt_tokens": int(prompt_len)
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")