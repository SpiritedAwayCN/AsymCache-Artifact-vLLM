import os
import re
import csv
from collections import defaultdict

log_dir = "Longbench-exp/minexp_7B_longbench_simulate_logs_bugfixed"
out_csv = "two_longbench_7B_cache_stats_bugfixed.csv"
fname_re = re.compile(
    r"Longbench_simulator_"
    r"(?P<policy>mwf|lru|lfu)_"
    r"qps(?P<qps>[\d.]+)_"
    r"div(?P<div>\d+)"
    r"(?:_grad(?P<grad>\d+))?"
    r"_P(?P<percent>\d+)\.log",
    re.IGNORECASE
)
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
    qps = float(mfn.group("qps"))
    percent = int(mfn.group("percent"))
    grad = int(mfn.group("grad")) if mfn.group("grad") else 10
    policy = mfn.group("policy").lower()
    key = (policy, div, qps, grad, percent)
    path = os.path.join(log_dir, fn)
    if div <= 320:
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
                    if (s1 - s0 == 64) or (s3 - s2 == 64):
                        agg[key]["one_blocksize"] += 1
         
        except Exception as e:
            print(f"读取文件失败 {path}: {e}")
            continue

with open(out_csv, "w", newline="", encoding="utf-8") as csvf:
    writer = csv.writer(csvf)
    writer.writerow([
        "policy", "div", "qps", "grad", "percent",
        "avg_hit_rate",
        "single_ratio",              
        "true_double_ratio",
        "false_double_ratio",     
        "true_triple_ratio",
        "false_triple_ratio",     
        "one_block_ratio",
        "total_items"
    ])

    for key in sorted(agg.keys(), key=lambda x: (x[0], x[1], x[2], x[3], x[4])):
        policy, div, qps, grad, percent = key
        rec = agg[key]

        total = rec["total_matches"]
        if total == 0:
            continue

        # 取出统计字段
        single = rec["single_count"]
        one_block = rec["one_blocksize"]
        true_double = rec["true_double_count"]
        true_triple = rec["true_triple_count"]
        false_triple = rec["false_triple_count"]
        false_double = rec["false_double_count"]
        avg_hit_rate = rec["total_hit_rate"] / total if total > 0 else 0.0
    
        # 各项比例
        single_ratio = single / total
        true_double_ratio = true_double / total
        true_triple_ratio = true_triple / total
        false_double_ratio = false_double / total
        false_triple_ratio = false_triple / total
        one_block_ratio = one_block / (true_double + false_double + true_triple + false_triple) if (true_double + false_double + true_triple + false_triple) > 0 else 0.0

        writer.writerow([
            policy, div, qps, grad, percent,
            f"{avg_hit_rate:.6f}",
            f"{single_ratio:.6f}",
            f"{true_double_ratio:.6f}",
            f"{false_double_ratio:.6f}",
            f"{true_triple_ratio:.6f}",
            f"{false_triple_ratio:.6f}",
            f"{one_block_ratio:.6f}",
            f"{total}"
        ])