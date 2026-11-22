import os
import json
import glob
import pandas as pd
from transformers import AutoTokenizer

# merged_dir = "/jizhicfs/marvinhjia/sca/LooGLE_merged"
# MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"

# # 加载 tokenizer
# tokenizer = AutoTokenizer.from_pretrained(
#     MODEL_NAME,
#     use_fast=True,
#     trust_remote_code=True
# )

# records = []

# for json_file in glob.glob(os.path.join(merged_dir, "*.jsonl")):
#     file_name = os.path.basename(json_file)
#     with open(json_file, "r", encoding="utf-8") as f:
#         for line_idx, line in enumerate(f):
#             doc = json.loads(line)
#             doc_id = doc["doc_id"]
#             outputs = doc.get("outputs", [])

#             for round_idx, answer in enumerate(outputs):
#                 if not isinstance(answer, str):
#                     print(
#                         f"[BAD ANSWER] doc_id={doc_id}, round={round_idx}, "
#                         f"type={type(answer)}, value={answer}, file={file_name}"
#                     )
#                     continue
#                 token_ids = tokenizer.encode(
#                     answer,
#                     add_special_tokens=False
#                 )
#                 token_len = len(token_ids)

#                 records.append({
#                     "file": file_name,
#                     "doc_id": doc_id,
#                     "round_id": round_idx,
#                     "answer_token_len": token_len,
#                 })

# # 转成 DataFrame
# df = pd.DataFrame(records)
# out_csv = "loogle_answer_token_stats.csv"
# df.to_csv(out_csv, index=False)
# print(f"Saved stats to {out_csv}")
# print(df.describe())

# import numpy as np

# token_lens = df["answer_token_len"].values

# stats = {
#     "count": len(token_lens),
#     "min": int(token_lens.min()),
#     "max": int(token_lens.max()),
#     "mean": float(token_lens.mean()),
#     "median": float(np.median(token_lens)),
#     "p90": float(np.percentile(token_lens, 90)),
#     "p95": float(np.percentile(token_lens, 95)),
#     "p99": float(np.percentile(token_lens, 99)),
# }

# for k, v in stats.items():
#     print(f"{k:>8}: {v:.2f}" if isinstance(v, float) else f"{k:>8}: {v}")

import math
import csv

BLOCK_SIZE = 64
output_csv = "loogle_prompt_blocks.csv"

records = []
dataset_dir = "/jizhicfs/marvinhjia/sca/LooGLE_merged"
df = pd.read_csv("/jizhicfs/marvinhjia/sca/data_preprocess/seqlen_stat_LooGLE.csv")
model = "meta-llama/Llama-3.1-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model)

# load dataset
datasets = {}
for jsonl_file in glob.glob(f"{dataset_dir}/**/*.jsonl", recursive=True):
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file):
            data = json.loads(line)
            file_name = os.path.basename(jsonl_file)
            if file_name not in datasets:
                datasets[file_name] = []
            datasets[file_name].append(data)

def generate_prompt_and_answer(conversations, tokenizer):
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


def prepare_requests(df, datasets, tokenizer):
    for _, row in df.iterrows():
        file_name = row['Dataset Name']
        line_number = row['Data Index']
        turn_id = row['Turn']

        data = datasets[file_name][line_number]
        msgs2 = [{'role': 'user', 'content': data['input'] + '\n\n'}]

        for idx, question in enumerate(data['instructions'][:turn_id + 1]):
            answer_text = data['outputs'][idx]
            if idx == 0:
                msgs2[0]['content'] += question
                msgs2.append({'role': 'assistant', 'content': answer_text})
            else:
                msgs2.append({'role': 'user', 'content': question})
                msgs2.append({'role': 'assistant', 'content': answer_text})

            prompt_tokens = generate_prompt_and_answer(msgs2, tokenizer)
            if prompt_tokens is None:
                continue

            blocks = math.ceil(prompt_tokens / BLOCK_SIZE)
            records.append({
                "Dataset Name": file_name,
                "Data Index": line_number,
                "Turn": idx,
                "Prompt Tokens": prompt_tokens,
                "Blocks": blocks,
            })


# 运行统计
prepare_requests(df, datasets, tokenizer)

# 写入 CSV
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Dataset Name",
            "Data Index",
            "Turn",
            "Prompt Tokens",
            "Blocks",
        ]
    )
    writer.writeheader()
    writer.writerows(records)

print(f"Saved prompt/block stats to {output_csv}")
