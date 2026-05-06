import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_files = ["ShareGPT_outputs_gamma_qps10_CV0.25_70B_1.0.csv"]

titles = [
    "q1=0.017, q2=0.00017",
    "q1=0.0125, q2=0.00025",
    # "q1=0.01, q2=0.001",
    # "div=80, qps=0.1",
    # "div=80, qps=0.5",
]


def parse_qps_div(filename):
    m = re.search(r"q([0-9.]+)_d([0-9]+)\.csv$", filename)
    if not m:
        raise ValueError(f"无法从文件名解析 qps/div: {filename}")
    qps = float(m.group(1))
    div = int(m.group(2))
    return qps, div


def compute_deltas(csv_path):
    df = pd.read_csv(csv_path)
    df["Turn"] = df["Turn"].astype(int)
    df["Data Index"] = df["Data Index"].astype(int)
    df["Timestamp"] = df["Timestamp"].astype(float)
    df = df.sort_values(["Dataset Name", "Data Index", "Turn", "Timestamp"])
    all_deltas = []
    for _, group in df.groupby(["Dataset Name", "Data Index"]):
        if len(group) < 2:
            continue
        ts = group["Timestamp"].to_numpy()
        deltas = np.diff(ts)
        all_deltas.append(deltas)

    return np.concatenate(all_deltas)


config_lines = []
plt.figure(figsize=(12, 4))
percentiles = [99]

for i, csv_path in enumerate(csv_files):
    all_deltas = compute_deltas(csv_path)
    x = np.sort(all_deltas)
    y = np.arange(1, len(x) + 1) / len(x)
    
    print(f"Processing {csv_path} ...")
    p_values = {p: np.percentile(x, p) for p in percentiles}

    for p in percentiles:
        val = p_values[p]
        y_val = np.searchsorted(x, val, side="right") / len(x)
        print(val, y_val)
        A = 0.0
        B = -0.975 * val
        alpha = -val / np.log(y_val)
        beta = -val / (40 * np.log(y_val))
        line = f"{p} {B:.6e} {alpha:.6e} {beta:.6e} {A}"
        print(line)
        # config_lines.append(line)  # 160 0.02 90 1.039651e+05 2.599128e+03 0.0 -1.067997e+04
# output_config = "10_trace_config.txt"
# with open(output_config, "a") as f:
#     for line in config_lines:
#         # f.write(line + "\n")      
#         plt.axvline(val, linestyle="--", alpha=0.6)
#         plt.scatter([val], [y_val], s=30)
#         plt.text(
#             val + 0.01 * x.max(), 
#             y_val,
#             f"P{p}={val:.2f}s",
#             fontsize=8,
#             verticalalignment="bottom",
#             horizontalalignment="left"
#         )
#     plt.title(titles[i])
#     plt.xlabel("inter-turn time (seconds)")
#     if i == 0:
#         plt.ylabel("CDF")
#     plt.grid(True, alpha=0.3)
# plt.tight_layout()
# output_path = "sim_CDF.png"
# plt.savefig(output_path, dpi=300)
# print(f"已保存 CDF 图到: {output_path}")