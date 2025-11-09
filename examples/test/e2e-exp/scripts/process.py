import os
import re
import csv
from collections import defaultdict

log_dir = "Leval_exp/new_four_data_experiment/qps2.0_70B_leval_simulate_logs"
out_csv = "leval_four_qps2.0_cache_stats.csv"

fname_re = re.compile(r"Leval_simulator_(?P<policy>mwf|lru|lfu)_qps[\d.]+_div(?P<div>\d+)_h(?P<h>\d+)\.log", re.IGNORECASE)
line_pattern = re.compile(r"num_blocks=(\d+).*?suffix_computed_block_locs:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\].*?first_compute_start:\s*(\d+)")

agg = defaultdict(
    lambda: {
        "single_count": 0,
        "true_double_count": 0,
        "false_double_count": 0,
        "true_triple_count": 0,
        "false_triple_count": 0,
        "one_blocksize": 0,
        "total_matches": 0,
        "total_hit_rate": 0.0,
    }
)

for fn in sorted(os.listdir(log_dir)):
    mfn = fname_re.match(fn)
    if not mfn:
        continue
    div = int(mfn.group("div"))
    half_time = int(mfn.group("h"))
    policy = mfn.group("policy").lower()
    key = (policy, div, half_time)
    path = os.path.join(log_dir, fn)
    if div <= 320 and half_time <= 100:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    mo = line_pattern.search(line)
                    if not mo:
                        continue
                    
                    num_blocks = int(mo.group(1))
                    s0, s1, s2, s3 = int(mo.group(2)), int(mo.group(3)), int(mo.group(4)), int(mo.group(5))
                    first_compute_start = int(mo.group(6))
                    
                    agg[key]["total_matches"] += 1
                    
                    hit_blocks = 0
                    if s1 > s0:
                        hit_blocks += (s1 - s0) // 64
                    if s3 > s2:
                        hit_blocks += (s3 - s2) // 64
                    if first_compute_start > 0:
                        hit_blocks += first_compute_start // 64
                    hit_rate = hit_blocks / num_blocks if num_blocks > 0 else 0.0
                    agg[key]["total_hit_rate"] += hit_rate
                    # 优先判三段 -> 两段 -> 单段
                    if s2 != 0:
                        if s0 != 0 and first_compute_start != 0:
                            agg[key]["true_triple_count"] += 1
                            # print(s0, s1, s2, s3)
                            # assert 0
                        else:
                            agg[key]["false_triple_count"] += 1
                        
                    elif s0 != 0:
                        if first_compute_start != 0:
                            agg[key]["true_double_count"] += 1
                        else:
                            agg[key]["false_double_count"] += 1
                    else:
                        if first_compute_start != 0:
                            agg[key]["single_count"] += 1
                    if (s1-s0 == 64) or (s3-s2 == 64):
                        agg[key]["one_blocksize"] += 1
         
        except Exception as e:
            print(f"读取文件失败 {path}: {e}")
            continue

with open(out_csv, "w", newline="", encoding="utf-8") as csvf:
    writer = csv.writer(csvf)
    writer.writerow([
        "policy", "div", "half_time",
        "avg_hit_rate",
        "single_ratio",              
        "true_double_ratio",
        "false_double_ratio",     
        "true_triple_ratio",
        "false_triple_ratio",     
        "one_block_ratio",
        "total_items"
    ])

    for key in sorted(agg.keys(), key=lambda x: (x[0], x[1], x[2])):
        policy, div, half_time = key
        rec = agg[key]

        total = rec["total_matches"]
        if total == 0:
            continue

        # ---- 取出统计字段 ----
        single = rec["single_count"]
        one_block = rec["one_blocksize"]
        true_double = rec["true_double_count"]
        true_triple = rec["true_triple_count"]
        false_triple = rec["false_triple_count"]
        false_double = rec["false_double_count"]
        avg_hit_rate = rec["total_hit_rate"] / total if total > 0 else 0.0
    
        # ---- 各项比例 ----
        single_ratio = single / total
        true_double_ratio = true_double / total
        true_triple_ratio = true_triple / total
        false_double_ratio = false_double / total
        false_triple_ratio = false_triple / total
        one_block_ratio = one_block / (true_double + false_double + true_triple + false_triple) if (true_double + false_double + true_triple + false_triple) > 0 else 0.0

        # ---- 输出到 CSV ----
        writer.writerow([
            policy, div, half_time,
            f"{avg_hit_rate:.6f}",
            f"{single_ratio:.6f}",
            f"{true_double_ratio:.6f}",
            f"{false_double_ratio:.6f}",
            f"{true_triple_ratio:.6f}",
            f"{false_triple_ratio:.6f}",
            f"{one_block_ratio:.6f}",
            f"{total}"
        ])