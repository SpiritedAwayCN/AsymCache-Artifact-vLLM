import pandas as pd

def rescale_qps(df, old_qps, new_qps, ts_col='Timestamp'):
    scale = old_qps / new_qps
    df_new = df.copy()
    df_new[ts_col] = df_new[ts_col] * scale
    df_new = df_new.sort_values(by=ts_col).reset_index(drop=True)
    return df_new

df = pd.read_csv('LEval_outputs_gamma_qps10_CV0.25_70B.csv')
df_new = rescale_qps(df, old_qps=0.02, new_qps=0.04)
overall_qps = len(df_new) / df_new['Timestamp'].iloc[-1]
print(f"Overall QPS: {overall_qps:.4f}")
output_file = f"LEval_outputs_gamma_qps10_CV0.25_70B_0.04.csv"
df_new.to_csv(output_file, index=False)
print(f"Output file saved to: {output_file}")
