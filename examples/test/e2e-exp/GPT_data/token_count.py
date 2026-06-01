# import json
# import pandas as pd
# from transformers import AutoTokenizer


# datasets = {}
# model = "/jizhicfs/marvinhjia/sca/models/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b"
# df = pd.read_csv("seqlen_stat_sharegpt.csv")
# tokenizer = AutoTokenizer.from_pretrained(model)
# output_jsonl = "gpt_prompt_output_token_stat.jsonl"


# with open("/jizhicfs/marvinhjia/sca/sharegpt_500.jsonl", 'r', encoding='utf-8') as file:
#     file_name = "sharegpt"
#     for line_number, line in enumerate(file):
#         data = json.loads(line)    
#         if file_name not in datasets:
#             datasets[file_name] = []
#         datasets[file_name].append(data)


# def generate_prompt_and_answer(conversations):
#     if len(conversations) == 0:
#         return None, None
#     if conversations[-1]["role"] != "assistant":
#         return None, None
#     prompt = tokenizer.apply_chat_template(
#         conversations[:-1],
#         tokenize=False,
#         add_generation_prompt=True
#     )
#     prompt_token_ids = tokenizer.encode(
#         prompt,
#         add_special_tokens=False
#     )
#     output_text = conversations[-1]["content"]
#     output_token_ids = tokenizer.encode(
#         output_text,
#         add_special_tokens=False
#     )
#     return len(prompt_token_ids), len(output_token_ids)


# num_records = 0
# total_prompt_tokens = 0
# total_output_tokens = 0

# with open(output_jsonl, "w", encoding="utf-8") as fout:
#     for _, row in df.iterrows():
#         file_name, line_number, turn_id = row['file'], row['line'], row['turn']
#         data = datasets[file_name][line_number]
#         msgs2 = []
#         for idx, question in enumerate(data['instructions'][:turn_id + 1]):
#             answer_text = data['outputs'][idx]
#             msgs2.append({'role': 'user', 'content': question})
#             msgs2.append({'role': 'assistant', 'content': answer_text})

#         prompt_len, output_len = generate_prompt_and_answer(msgs2)
#         if prompt_len is None or output_len is None:
#             continue
#         record = {
#             "file": file_name,
#             "line": int(line_number),
#             "turn": int(turn_id),
#             "prompt_tokens": int(prompt_len),
#             "output_tokens": int(output_len),
#         }
#         fout.write(json.dumps(record, ensure_ascii=False) + "\n")
#         total_prompt_tokens += prompt_len
#         total_output_tokens += output_len
#         num_records += 1

# avg_prompt_tokens = total_prompt_tokens / num_records if num_records > 0 else 0
# avg_output_tokens = total_output_tokens / num_records if num_records > 0 else 0

# print(f"Total records: {num_records}")
# print(f"Average prompt_tokens: {avg_prompt_tokens:.2f}")
# print(f"Average output_tokens: {avg_output_tokens:.2f}")
# print(f"Results saved to: {output_jsonl}")

import json
input_jsonl = "gpt_prompt_token_stat.jsonl"

total_prompt_tokens = 0
num_records = 0
with open(input_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        record = json.loads(line)

        if "prompt_tokens" not in record:
            continue

        prompt_tokens = int(record["prompt_tokens"])

        total_prompt_tokens += prompt_tokens
        num_records += 1

avg_prompt_tokens = total_prompt_tokens / num_records if num_records > 0 else 0

print(f"Total records: {num_records}")
print(f"Total prompt_tokens: {total_prompt_tokens}")
print(f"Average prompt_tokens: {avg_prompt_tokens:.2f}")