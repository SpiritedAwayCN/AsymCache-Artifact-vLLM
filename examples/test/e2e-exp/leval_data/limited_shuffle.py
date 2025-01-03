import random
import pandas as pd
import os
import argparse

BASE_DIR = '/processing_data/infra/sca/dataset/LEval'

parser = argparse.ArgumentParser()
parser.add_argument('--div', type=int, default=80)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--qps', type=float, default=0.5)
args = parser.parse_args()

def shuffle_array_within_k(array, K, seed=None):
    if seed is not None:
        random.seed(seed)
    
    n = len(array)
    indexed_arr = []
    
    for i in range(n):
        new_index = i + random.randint(-K, K)
        indexed_arr.append((new_index, array[i]))
    
    indexed_arr.sort()
    shuffled_arr = [elem for index, elem in indexed_arr]
    return shuffled_arr

df = pd.read_csv(os.path.join(BASE_DIR, 'seqlen_stat_LEval.csv'))
index_list = list(range(len(df)))
index_list = shuffle_array_within_k(index_list, args.div, seed=args.seed)

turn_indices_col = []
data_indices_col = []
dataset_names_col = []
# 根据打乱后的索引构建包含数据集名称, 数据行号和对话轮次的新列

current_turn_map = {}

for idx in index_list:
    row = df.iloc[idx]
    file_name, line_number, _ = row['file'], row['line'], row['turn']
    dataset_names_col.append(file_name)
    data_indices_col.append(line_number)
    
    current_turn_map_key = f'{file_name}:{line_number}'
    if current_turn_map_key in current_turn_map:
        current_turn_map[current_turn_map_key] += 1
    else:
        current_turn_map[current_turn_map_key] = 0
    
    turn_indices_col.append(current_turn_map[current_turn_map_key])

# 现有CSV文件路径, 仅含有一列时间戳数据
csv_file_path = f'time_qps{args.qps}.csv'
df = pd.read_csv(csv_file_path)

# 将新列添加到DataFrame
df['Dataset Name'] = dataset_names_col
df['Data Index'] = data_indices_col
df['Turn'] = turn_indices_col

# 保存至最终的CSV文件
output_file = f"outputs_q{args.qps}_d{args.div}.csv"
df.to_csv(output_file, index=False)
print(f"新文件已保存到 {output_file}")