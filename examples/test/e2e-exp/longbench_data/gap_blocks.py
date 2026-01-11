# import pandas as pd
# import json
# import math

jsonl_file_path = '/jizhicfs/marvinhjia/sca/yilin/vllm-dev/examples/test/e2e-exp/LongBench_simulator_minexp_full_qps0.007_0.0007_CV0.25_P99_grad40_gamma.jsonl'
csv_file_path = 'outputs_gamma_qps10_CV0.25_70B.csv'
BLOCK_SIZE = 64

# def calculate_interfering_blocks():
#     token_map = {}
#     try:
#         with open(jsonl_file_path, 'r', encoding='utf-8') as f:
#             for line_content in f:
#                 try:
#                     entry = json.loads(line_content)
#                     fname = entry.get('file')
#                     line_idx = entry.get('line')
#                     turn_idx = entry.get('turn')
#                     # 提取 prompt_tokens
#                     p_tokens = entry.get('usage', {}).get('prompt_tokens', 0)
                    
#                     if fname is not None and line_idx is not None and turn_idx is not None:
#                         token_map[(fname, line_idx, turn_idx)] = p_tokens
#                 except:
#                     continue
#     except FileNotFoundError:
#         print(f"错误: 找不到文件 {jsonl_file_path}")
#         return

#     df = pd.read_csv(csv_file_path)
#     df.columns = [c.strip() for c in df.columns]
#     def get_blocks(row):
#         key = (row['Dataset Name'], row['Data Index'], row['Turn'])
#         tokens = token_map.get(key, 0)
#         return math.ceil(tokens / BLOCK_SIZE)

#     df['blocks'] = df.apply(get_blocks, axis=1)
#     df = df.sort_values(by='Timestamp').reset_index(drop=True)
#     df['cumsum_blocks'] = df['blocks'].cumsum()
    
#     interfering_blocks_list = []
#     grouped = df.groupby(['Dataset Name', 'Data Index'])
#     for (dataset, data_idx), group in grouped:
#         sorted_indices = group.sort_values('Turn').index.tolist()
#         # 遍历该 Session 的连续两轮
#         for i in range(len(sorted_indices) - 1):
#             global_idx_start = sorted_indices[i]     # Turn N 的全局位置
#             global_idx_end = sorted_indices[i+1]     # Turn N+1 的全局位置
            
#             # 计算中间夹杂的 Block 总数
#             # 逻辑：全局 index 在 (global_idx_start, global_idx_end) 之间的所有请求 block 之和
#             if global_idx_end > global_idx_start + 1:
#                 blocks_end_prev = df.loc[global_idx_end - 1, 'cumsum_blocks']
#                 blocks_start = df.loc[global_idx_start, 'cumsum_blocks']
#                 interval_sum = blocks_end_prev - blocks_start
#             else:
#                 interval_sum = 0
            
#             interfering_blocks_list.append(interval_sum)

#     # 6. 输出结果
#     if interfering_blocks_list:
#         avg_interfering = sum(interfering_blocks_list) / len(interfering_blocks_list)
#         print("计算完成")
#         print(f"统计对象: 连续两轮 Session 对话之间插入的【其他请求】")
#         print(f"总样本对数: {len(interfering_blocks_list)}")
#         print(f"Block Size: {BLOCK_SIZE}")
#         print("-" * 40)
#         print(f"平均间隔 Block 总量: {avg_interfering:.4f}")
#         print("=" * 40)
        
#         print(f"最大间隔 Block: {max(interfering_blocks_list)}")
#         print(f"最小间隔 Block: {min(interfering_blocks_list)}")
#     else:
#         print("未找到有效的连续对话数据。")
        
# if __name__ == "__main__":
#     calculate_interfering_blocks()

# import json
# import math
# import pandas as pd
# import matplotlib.pyplot as plt


# CUDA_BLOCKS = 8837
# token_map = {}
# with open(jsonl_file_path, 'r') as f:
#     for line in f:
#         try:
#             e = json.loads(line)
#             key = (e.get('file'), e.get('line'), e.get('turn'))
#             tokens = e.get('usage', {}).get('prompt_tokens', 0)
#             if None not in key:
#                 token_map[key] = tokens
#         except:
#             pass

# df = pd.read_csv(csv_file_path)
# df.columns = [c.strip() for c in df.columns]

# def calc_blocks(row):
#     key = (row['Dataset Name'], row['Data Index'], row['Turn'])
#     tokens = token_map.get(key, 0)
#     return math.ceil(tokens / BLOCK_SIZE)

# df['blocks'] = df.apply(calc_blocks, axis=1)

# # Global arrival order
# df = df.sort_values('Timestamp').reset_index(drop=True)
# df['cumsum_blocks'] = df['blocks'].cumsum()

# gap_req = []
# gap_blocks = []
# grouped = df.groupby(['Dataset Name', 'Data Index'])

# for _, group in grouped:
#     idxs = group.sort_values('Turn').index.tolist()
#     for i in range(len(idxs) - 1):
#         s = idxs[i]
#         e = idxs[i + 1]

#         n_req = max(0, e - s - 1)
#         gap_req.append(n_req)

#         if n_req > 0:
#             b = df.loc[e - 1, 'cumsum_blocks'] - df.loc[s, 'cumsum_blocks']
#         else:
#             b = 0
#         gap_blocks.append(b)

# gap_df = pd.DataFrame({
#     'intervening_requests': gap_req,
#     'intervening_blocks': gap_blocks
# })


# plt.figure(figsize=(7,4))
# plt.hist(gap_df['intervening_blocks'], bins=100)
# plt.axvline(CUDA_BLOCKS, linestyle='--', linewidth=2)
# plt.xlabel('Intervening KV blocks between consecutive turns')
# plt.ylabel('Count')
# plt.title('Inter-session Interference vs KV Cache Capacity')
# plt.tight_layout()
# output_path = "const_qps10.png"
# plt.savefig(output_path, dpi=300)
# print(f"已保存到: {output_path}")
# print("Total samples:", len(gap_df))
# print("Mean blocks:", gap_df['intervening_blocks'].mean())
# print("Max blocks:", gap_df['intervening_blocks'].max())
# print("P90 blocks:", gap_df['intervening_blocks'].quantile(0.9))
# print("Fraction > cache:",(gap_df['intervening_blocks'] > CUDA_BLOCKS).mean())

import pandas as pd
import json
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 1. 构建 Token Map
token_map = {}
with open(jsonl_file_path, 'r') as f:
    for line in f:
        try:
            e = json.loads(line)
            key = (e.get('file'), e.get('line'), e.get('turn'))
            tokens = e.get('usage', {}).get('prompt_tokens', 0)
            if None not in key:
                token_map[key] = tokens
        except:
            pass

# 2. 读取并处理 DataFrame
df = pd.read_csv(csv_file_path)
df.columns = [c.strip() for c in df.columns]

def calc_blocks(row):
    key = (row['Dataset Name'], row['Data Index'], row['Turn'])
    tokens = token_map.get(key, 0)
    return math.ceil(tokens / BLOCK_SIZE)

df['blocks'] = df.apply(calc_blocks, axis=1)
if 'Timestamp' in df.columns:
    df = df.sort_values('Timestamp').reset_index(drop=True)
else:
    df = df.reset_index(drop=True)

df['cumsum_blocks'] = df['blocks'].cumsum()
df['gap_blocks'] = 0.0
grouped = df.groupby(['Dataset Name', 'Data Index'])

for _, group in grouped:
    # 获取该 Session 内按 Turn 排序后的全局索引列表
    idxs = group.sort_values('Turn').index.tolist()
    # 遍历该 Session 的每一对相邻轮次 (Turn i -> Turn i+1)
    for i in range(len(idxs) - 1):
        s = idxs[i]       # 上一轮 (source) 的全局索引
        e = idxs[i + 1]   # 当前轮 (end) 的全局索引  
        # 计算中间间隔的 Blocks
        # 逻辑：当前请求 e 之前的所有累积 blocks (e-1) 减去 上一轮请求 s 时的累积 blocks
        # 如果 e == s + 1，说明中间没有请求，gap 为 0
        if e > s + 1:
            gap = df.loc[e - 1, 'cumsum_blocks'] - df.loc[s, 'cumsum_blocks']
        else:
            gap = 0 
        # 将 gap 记录在当前轮次 e 上
        df.loc[e, 'gap_blocks'] = gap

plt.figure(figsize=(20, 8))
# X轴：1 到 1140
x_axis = range(1, len(df) + 1)
y_axis = df['gap_blocks']
# 绘制散点
plt.scatter(x_axis, y_axis, s=15, alpha=0.7, label='Gap Blocks')
plt.axhline(y=7893, color='red', linestyle='--', linewidth=2, label='Cache Capacity (7893 Blocks)')
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
plt.xticks(rotation=45, fontsize=8)
plt.xlabel('Request Arrival Order')
plt.ylabel('Gap Blocks (Blocks from Intervening Requests)')
plt.title('Gamma QPS10 CV0.25 70B: Gap Blocks Distribution Per Request')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlim(0, len(df) + 20)
plt.tight_layout()
output_path = "scatter_gamma_qps10_CV0.25_70B_0.023.png"
plt.savefig(output_path, dpi=300)
print(f"已保存到: {output_path}")