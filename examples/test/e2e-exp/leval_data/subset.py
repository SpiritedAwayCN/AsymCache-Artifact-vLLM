import pandas as pd

df = pd.read_csv('/jizhicfs/marvinhjia/sca/yilin/vllm-dev/examples/test/e2e-exp/leval_data/LEval_outputs_gamma_qps10_CV0.25_70B_0.04.csv')
print(f"数据总行数: {len(df)}")
df_subset = df.iloc[400:700].copy()
base_timestamp = df_subset.iloc[0, 0]
df_subset.iloc[:, 0] = df_subset.iloc[:, 0] - base_timestamp

df_subset.iloc[0, 0] = 0
print(f"提取了 {len(df_subset)} 条数据")
print(f"时间戳基准值: {base_timestamp}")

# 保存为新的CSV文件
output_filename = 'outputs_gamma_qps10_CV0.25_70B_0.04_subset.csv'
df_subset.to_csv(output_filename, index=False)
print(f"\n已保存为: {output_filename}")