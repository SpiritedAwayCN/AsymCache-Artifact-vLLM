import math
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BLOCK_SIZE = 64
prompt_jsonl = "loogle_prompt_token_stat.jsonl"

token_map = {}
with open(prompt_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        key = (item["file"], item["line"], item["turn"])
        token_map[key] = item["prompt_tokens"]

def calc_blocks(row):
    key = (row['file'], row['line'], row['turn'])
    tokens = token_map.get(key, 0)
    return math.ceil(tokens / BLOCK_SIZE)

df = pd.read_csv("LooGLE_outputs_gamma_qps5_CV0.25_70B_0.01.csv")
df.columns = [c.strip() for c in df.columns]
df['blocks'] = df.apply(calc_blocks, axis=1)

if 'Timestamp' in df.columns:
    df = df.sort_values('Timestamp').reset_index(drop=True)
else:
    df = df.reset_index(drop=True)

df['cumsum_blocks'] = df['blocks'].cumsum()
df['gap_blocks'] = 0.0
grouped = df.groupby(['file', 'line'])

for _, group in grouped:
    idxs = group.sort_values('turn').index.tolist()
    for i in range(len(idxs) - 1):
        s = idxs[i]
        e = idxs[i + 1]
        if e > s + 1:
            gap = df.loc[e - 1, 'cumsum_blocks'] - df.loc[s, 'cumsum_blocks']
        else:
            gap = 0
        df.loc[e, 'gap_blocks'] = gap

plt.figure(figsize=(20, 8))
x_axis = range(1, len(df) + 1)
y_axis = df['gap_blocks']
plt.scatter(x_axis, y_axis, s=15, alpha=0.7, label='Gap Blocks')
plt.axhline(y=7893, color='red', linestyle='--', linewidth=2, label='Cache Capacity (7893 Blocks)')
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
plt.xticks(rotation=45, fontsize=8)
plt.xlabel('Request Arrival Order')
plt.ylabel('Gap Blocks (Blocks from Intervening Requests)')
plt.title('LEval Prompt-Based Gap Blocks Distribution (Gamma QPS5 CV0.25, 70B)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlim(0, len(df) + 20)
plt.tight_layout()
output_path = "LooGLE_scatter_gamma_qps5_CV0.25_70B_0.01.png"
plt.savefig(output_path, dpi=300)
print(f"已保存到: {output_path}")