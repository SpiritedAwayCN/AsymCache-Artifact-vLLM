import pandas as pd

# 读取CSV文件
df = pd.read_csv('/jizhicfs/marvinhjia/sca/yilin/vllm-dev/examples/test/e2e-exp/longbench_data/outputs_gamma_qps10_CV0.25_8B_0.04.csv')

# 检查数据形状
print(f"数据总行数: {len(df)}")

# 提取第661到960条（注意：Python是0-based索引，所以第661条对应索引660）
df_subset = df.iloc[660:960].copy()  # 960是exclusive，所以取660:960

# 获取第661条的时间戳作为基准
base_timestamp = df_subset.iloc[0, 0]

# 调整时间戳：每条时间戳减去基准值
df_subset.iloc[:, 0] = df_subset.iloc[:, 0] - base_timestamp

# 确保第661条的时间戳为0
df_subset.iloc[0, 0] = 0

print(f"提取了 {len(df_subset)} 条数据")
print(f"时间戳基准值: {base_timestamp}")

# 保存为新的CSV文件
output_filename = 'outputs_gamma_qps10_CV0.25_8B_0.04_subset.csv'
df_subset.to_csv(output_filename, index=False)
print(f"\n已保存为: {output_filename}")
