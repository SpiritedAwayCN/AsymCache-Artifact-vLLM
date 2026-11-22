import os
import glob
import json
import pandas as pd
from collections import defaultdict


input_dir = "/jizhicfs/marvinhjia/sca/LooGLE"
output_dir = "/jizhicfs/marvinhjia/sca/LooGLE_merged"
os.makedirs(output_dir, exist_ok=True)

for json_file in glob.glob(f"{input_dir}/*.jsonl"):
    file_name = os.path.basename(json_file)
    print(f"Processing {file_name} ...")

    # 临时容器：doc_id → 合并结构
    merged_data = {}

    with open(json_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            doc_id = data["doc_id"]
            # 初始化该 doc_id 的结构
            if doc_id not in merged_data:
                merged_data[doc_id] = {
                    "doc_id": doc_id,
                    "task": data.get("task"),
                    "context": data.get("context"),
                    "title": data.get("title"),
                    "metadata": data.get("metadata"),
                    "instructions": [],
                    "outputs": [],
                }

            # 添加question-answer
            merged_data[doc_id]["instructions"].append(data["question"])
            merged_data[doc_id]["outputs"].append(data["answer"])

    output_path = os.path.join(output_dir, file_name)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for doc in merged_data.values():
            json.dump(doc, out_f, ensure_ascii=False)
            out_f.write("\n")

    print(f"Saved merged file: {output_path} ({len(merged_data)} unique doc_ids)")
