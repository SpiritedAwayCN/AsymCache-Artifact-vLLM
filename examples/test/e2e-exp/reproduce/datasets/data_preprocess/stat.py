import glob
import os
import json

line_idx = []
datasets = {}
line_turns = []
line_filename = []
directory = '/jizhicfs/marvinhjia/sca/yilin/vllm-dev/examples/test/e2e-exp/scripts'
for json_file in glob.glob(f"{directory}/*.jsonl", recursive=True):
    total_conversations = 0
    with open(json_file, 'r', encoding='utf-8') as file:
        file_name = os.path.basename(json_file)
        print(file_name)
        for line_number, data in enumerate(file):
            data = json.loads(data)
            line_filename.append(file_name)
            line_idx.append(line_number)
            print(data.keys())
            line_turns.append(len(data['instructions']))
            if file_name not in datasets:
                datasets[file_name] = []
            datasets[file_name].append(data)

tot = len(line_idx)
print(tot)


column_file = []
column_line = []
column_turn = []
for i in range(tot):
    for turn_idx in range(line_turns[i]):
        column_file.append(line_filename[i])
        column_line.append(line_idx[i])
        column_turn.append(turn_idx)


import pandas as pd
df = pd.DataFrame({
    "file":column_file,
    "line":column_line,
    "turn":column_turn
})
df.to_csv("two_seqlen_stat_longbench.csv", index=False)
