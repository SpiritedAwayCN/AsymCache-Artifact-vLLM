import pandas as pd

df = pd.read_csv('ShareGPT_outputs_gamma_qps10_CV0.25_70B_0.25.csv')
print(f"数据总行数: {len(df)}")
df_subset = df.iloc[200:1000].copy()
base_timestamp = df_subset.iloc[0, 0]
df_subset.iloc[:, 0] = df_subset.iloc[:, 0] - base_timestamp

df_subset.iloc[0, 0] = 0
print(f"提取了 {len(df_subset)} 条数据")
print(f"时间戳基准值: {base_timestamp}")

# 保存为新的CSV文件
output_filename = 'outputs_gamma_qps10_CV0.25_70B_0.25_800.csv'
df_subset.to_csv(output_filename, index=False)
print(f"\n已保存为: {output_filename}")