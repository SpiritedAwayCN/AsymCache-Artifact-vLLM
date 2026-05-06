import os
import random
import argparse
import pandas as pd
from timestamp import GammaRequestIntervalGenerator


# BASE_DIR = '/jizhicfs/marvinhjia/sca/data_preprocess'
# parser = argparse.ArgumentParser()
# parser.add_argument('--seed', type=int, default=42)
# parser.add_argument('--q1', type=float, default=0.0035)
# parser.add_argument('--q2', type=float, default=0.00035)
# parser.add_argument('--k1', type=float, default=16, help="gamma shape for inter-session intervals")
# parser.add_argument('--k2', type=float, default=16, help="gamma shape for intra-session intervals")
# args = parser.parse_args()


# def _count_inversions(arr):
#     """Count inversions in the given array using merge sort (O(n log n)).
#     The array should contain comparable items; returns the number of pairs (i < j)
#     with arr[i] > arr[j].
#     """
#     def sort_count(a):
#         n = len(a)
#         if n <= 1:
#             return a, 0
#         mid = n // 2
#         left, c_left = sort_count(a[:mid])
#         right, c_right = sort_count(a[mid:])
#         merged = []
#         i = j = 0
#         c_split = 0
#         while i < len(left) and j < len(right):
#             if left[i] <= right[j]:
#                 merged.append(left[i])
#                 i += 1
#             else:
#                 merged.append(right[j])
#                 j += 1
#                 c_split += len(left) - i
#         if i < len(left):
#             merged.extend(left[i:])
#         if j < len(right):
#             merged.extend(right[j:])
#         return merged, c_left + c_right + c_split

#     _, inv_count = sort_count(list(arr))
#     return inv_count


# def inversion_count_of_timestamp_sort(df):
#     """Compute the inversion count after sorting the DataFrame by 'Timestamp',
#     relative to the original order.

#     This function does not modify the input DataFrame. It creates a copy with an
#     original index marker, sorts by 'Timestamp', then counts inversions in the
#     sequence of original indices in the sorted order.
#     """
#     tmp = df.copy()
#     tmp['__orig_idx__'] = range(len(tmp))
#     tmp = tmp.sort_values(by='Timestamp')
#     return _count_inversions(tmp['__orig_idx__'].to_list())


# class ConstantIntervalGenerator:
#     def __init__(self, qps: float):
#         if qps <= 0:
#             raise ValueError(f"qps must be > 0, got {qps}")
#         self.interval = 1.0 / float(qps)

#     def get_next_inter_request_time(self) -> float:
#         return self.interval


# generator_inter = GammaRequestIntervalGenerator(qps=args.q1, seed=args.seed,   shape=args.k1)
# generator_intra = GammaRequestIntervalGenerator(qps=args.q2, seed=args.seed+1, shape=args.k2)
# # generator_inter = PoissonRequestIntervalGenerator(qps=args.q1, seed=args.seed)
# # generator_intra = PoissonRequestIntervalGenerator(qps=args.q2, seed=args.seed+1)
# # 固定间隔替代泊松
# # generator_inter = ConstantIntervalGenerator(qps=args.q1)
# # generator_intra = ConstantIntervalGenerator(qps=args.q2)

# df = pd.read_csv(os.path.join(BASE_DIR, 'seqlen_stat_Longbench.csv'))
# timestamp_col = []
# inter_time_acc = 0.0
# intra_time_acc = 0.0
# last_session = (None, None)
# for row in df.itertuples():
#     file_name = row.file
#     line_number = row.line
#     turn = row.turn
    
#     current_session = (file_name, line_number)
    
#     if current_session != last_session:
#         inter_interval = generator_inter.get_next_inter_request_time()
#         inter_time_acc += inter_interval
#         intra_time_acc = 0.0
#         last_session = current_session
#     else:
#         intra_interval = generator_intra.get_next_inter_request_time()
#         intra_time_acc += intra_interval
        
#     timestamp_col.append(inter_time_acc + intra_time_acc)

# df['Timestamp'] = timestamp_col
# cols = df.columns.tolist()
# cols = cols[-1:] + cols[:-1]
# df = df[cols]

# # Keep a copy before sorting to compute inversion count relative to original order
# df_before_sort = df.copy()
# df = df.sort_values(by='Timestamp')
# # Compute and report inversion count after sorting by Timestamp
# inv_count = inversion_count_of_timestamp_sort(df_before_sort)
# print(f"Inversion count after Timestamp sort: {inv_count}")
# overall_qps = len(df) / df['Timestamp'].iloc[-1]
# print(f"Overall QPS: {overall_qps:.4f}")
# # 保存至最终的CSV文件
# output_file = f"outputs_gamma_qps10_CV0.25_8B_0.01.csv"
# df.to_csv(output_file, index=False)
# print(f"Output file saved to: {output_file}")


def rescale_qps(df, old_qps, new_qps, ts_col='Timestamp'):
    scale = old_qps / new_qps
    df_new = df.copy()
    df_new[ts_col] = df_new[ts_col] * scale
    df_new = df_new.sort_values(by=ts_col).reset_index(drop=True)
    return df_new

df = pd.read_csv('outputs_gamma_qps10_CV0.25_8B_0.01.csv')
df_new = rescale_qps(df, old_qps=0.01, new_qps=0.04)
overall_qps = len(df_new) / df_new['Timestamp'].iloc[-1]
print(f"Overall QPS: {overall_qps:.4f}")
output_file = f"outputs_gamma_qps10_CV0.25_8B_0.04.csv"
df_new.to_csv(output_file, index=False)
print(f"Output file saved to: {output_file}")