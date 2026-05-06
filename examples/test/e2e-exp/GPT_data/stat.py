import glob
import os
import json

line_idx = []
line_turns = []
line_filename = []

total_conversations = 0
json_file = "/jizhicfs/marvinhjia/sca/sharegpt_500.jsonl"
with open(json_file, 'r', encoding='utf-8') as file:
    file_name = "sharegpt"
    for line_number, data in enumerate(file):
        data = json.loads(data)
        line_filename.append(file_name)
        line_idx.append(line_number)
        line_turns.append(len(data['instructions']))

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
df.to_csv("seqlen_stat_sharegpt.csv", index=False)