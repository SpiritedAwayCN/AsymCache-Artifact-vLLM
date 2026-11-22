import os
import glob
import json

BASE_DIR = "/jizhicfs/marvinhjia/sca/Longbench"
OUTPUT_DIR = "/jizhicfs/marvinhjia/sca/Two-Longbench"

prefixes = ["K"]

for jsonl_file in glob.glob(f"{BASE_DIR}/*.jsonl", recursive=True):
    rel_path = os.path.relpath(jsonl_file, BASE_DIR)
    output_file = os.path.join(OUTPUT_DIR, rel_path)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 读取原始数据
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f if line.strip()]

    new_lines = []
    for data in lines:
        new_lines.append(json.dumps(data, ensure_ascii=False))

    for prefix in prefixes:
        for data in lines:
            new_data = data.copy()
            if "input" in new_data:
                new_data["input"] = prefix + new_data["input"]
            new_lines.append(json.dumps(new_data, ensure_ascii=False))

    # 写入新文件（每条数据一行）
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(new_lines) + "\n")
