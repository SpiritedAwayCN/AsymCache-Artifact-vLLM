import os
import numpy as np
import pandas as pd

file_name = 'BurstGPT_without_fails_1.csv'

def read_burstgpt_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    df = pd.read_csv(file_path)
    return df

burstgpt_data = read_burstgpt_data(file_name)
ts_lowerbound = 2 * 24 * 60 * 60  # 2 days in seconds
ts_upperbound = 3.7 * 24 * 60 * 60 + 1500  # 15 days in seconds

filtered_data = burstgpt_data[
    (burstgpt_data['Model'] == 'GPT-4') &
    (burstgpt_data['Log Type'] == 'Conversation log') &
    (burstgpt_data['Timestamp'].between(ts_lowerbound, ts_upperbound))
]

filtered_data = filtered_data.iloc[:-1]
print(len(filtered_data))
print(filtered_data.isna().any().any())

def rescale_to_qps(df: pd.DataFrame, qps: float, col='Timestamp', output_csv='time_qps0.02.csv'):
    # 转为数值并去掉NaN
    x = pd.to_numeric(df[col], errors='coerce').dropna()
    order = np.argsort(x.values)
    x_sorted = x.values[order]
    
    # 计算时间范围和当前QPS
    t0 = x_sorted[0]
    t1 = x_sorted[-1]
    duration = max(1e-12, t1 - t0)
    
    n = len(x_sorted)
    current_qps = n / duration
    print(f"当前样本数: {n}, 当前 QPS: {current_qps}")
    
    # 线性缩放生成新的时间戳
    scale = current_qps / qps
    x_rescaled = (df[col].values - t0) * scale
    
    rescaled_df = pd.DataFrame({f'{col}_rescaled': x_rescaled})
    nan_count = rescaled_df.isna().sum().iloc[0]
    if nan_count > 0:
        print(f"新生成列中有{nan_count}个NaN值")
    rescaled_df.to_csv(output_csv, index=False)
    print(f"已保存至: {output_csv}")
    
    vals = rescaled_df[f'{col}_rescaled'].dropna().values
    if len(vals) > 1:
        new_duration = vals[-1] - vals[0]
        new_qps = len(vals) / new_duration
        print(f"新 QPS: {new_qps:.6f}，目标 QPS: {qps:.6f}")

target_qps = 0.02
rescale_to_qps(filtered_data, qps=target_qps, col='Timestamp', output_csv='time_qps0.02.csv')