import json
import os


def is_cache_hit(record):
    choices = record.get("choices", [])
    metrics = {}
    if choices and isinstance(choices[0], dict):
        metrics = choices[0].get("metrics", {})
    suffix_locs = metrics.get("suffix_computed_block_locs", [])
    if all(x == 0 for x in suffix_locs):
        return False
    else:
        return True


def get_ttft(record):
    choices = record.get("choices", [])
    metrics = {}
    if choices and isinstance(choices[0], dict):
        metrics = choices[0].get("metrics", {})
    arrival_time = metrics.get("arrival_time")
    first_token_time = metrics.get("first_token_time")
    if arrival_time is None or first_token_time is None:
        return None
    return first_token_time - arrival_time


def get_new_ttft(record):
    choices = record.get("choices", [])
    metrics = {}
    if choices and isinstance(choices[0], dict):
        metrics = choices[0].get("metrics", {})
    first_token_time = metrics.get("first_token_time")
    schedule_time = metrics.get("first_scheduled_time")
    if schedule_time is None or first_token_time is None:
        return None
    return first_token_time - schedule_time


def build_info_map(file_path):
    info_map = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)
            key = (record.get("file"), record.get("line"), record.get("turn"))
            ttft = get_ttft(record)
            hit = is_cache_hit(record)
            new_ttft = get_new_ttft(record)
            info_map[key] = {"hit": hit, "ttft": ttft, "new_ttft": new_ttft}
    return info_map


def compute_stats(info_a, info_b):
    stats = {
        "both_hit":   {"count": 0, "ttft_sum_a": 0.0, "ttft_sum_b": 0.0, "new_ttft_sum_a": 0.0, "new_ttft_sum_b": 0.0, "rows": []},
        "only_a_hit": {"count": 0, "ttft_sum_a": 0.0, "ttft_sum_b": 0.0, "new_ttft_sum_a": 0.0, "new_ttft_sum_b": 0.0, "rows": []},
        "only_b_hit": {"count": 0, "ttft_sum_a": 0.0, "ttft_sum_b": 0.0, "new_ttft_sum_a": 0.0, "new_ttft_sum_b": 0.0, "rows": []},
        "both_miss":  {"count": 0, "ttft_sum_a": 0.0, "ttft_sum_b": 0.0, "new_ttft_sum_a": 0.0, "new_ttft_sum_b": 0.0, "rows": []},
    }

    all_keys = set(info_a.keys()) | set(info_b.keys())

    for key in all_keys:
        rec_a = info_a.get(key, {"hit": False, "ttft": None, "new_ttft": None})
        rec_b = info_b.get(key, {"hit": False, "ttft": None, "new_ttft": None})

        hit_a = rec_a["hit"]
        hit_b = rec_b["hit"]
        ttft_a = rec_a["ttft"]
        ttft_b = rec_b["ttft"]
        new_ttft_a = rec_a["new_ttft"]
        new_ttft_b = rec_b["new_ttft"]

        if hit_a and hit_b:
            cat = "both_hit"
        elif hit_a and not hit_b:
            cat = "only_a_hit"
        elif (not hit_a) and hit_b:
            cat = "only_b_hit"
        else:
            cat = "both_miss"

        stats[cat]["count"] += 1
        stats[cat]["rows"].append(key) 
        
        if ttft_a is not None:
            stats[cat]["ttft_sum_a"] += ttft_a
        if ttft_b is not None:
            stats[cat]["ttft_sum_b"] += ttft_b
        if new_ttft_a is not None:
            stats[cat]["new_ttft_sum_a"] += new_ttft_a
        if new_ttft_b is not None:
            stats[cat]["new_ttft_sum_b"] += new_ttft_b

    for cat, v in stats.items():
        v["gain"] = v["ttft_sum_a"] - v["ttft_sum_b"]
        v["new_gain"] = v["new_ttft_sum_a"] - v["new_ttft_sum_b"]
    return stats


def print_row_details(stats):
    for cat in ["both_hit", "only_a_hit", "only_b_hit", "both_miss"]:
        rows = stats[cat].get("rows", [])
        print(f"\n[{cat}] 行数 = {len(rows)}")
        for file_name, line_no, turn_no in rows[:10]:
            print(f"  file={file_name}, line={line_no}, turn={turn_no}")

            
def print_tables(stats, name_a="A", name_b="B"):
    both_hit   = stats["both_hit"]["count"]
    both_miss  = stats["both_miss"]["count"]
    only_a_hit = stats["only_a_hit"]["count"]
    only_b_hit = stats["only_b_hit"]["count"]
    
    gain_both_hit   = stats["both_hit"]["gain"]
    gain_both_miss  = stats["both_miss"]["gain"]
    gain_only_a_hit = stats["only_a_hit"]["gain"]
    gain_only_b_hit = stats["only_b_hit"]["gain"]
    
    new_gain_both_hit   = stats["both_hit"]["new_gain"]
    new_gain_both_miss  = stats["both_miss"]["new_gain"]
    new_gain_only_a_hit = stats["only_a_hit"]["new_gain"]
    new_gain_only_b_hit = stats["only_b_hit"]["new_gain"]
    
    total = both_hit + only_a_hit + only_b_hit
    print("\n[Percent] excluding both_miss")
    pct_both_hit   = both_hit / total
    pct_only_a_hit = only_a_hit / total
    pct_only_b_hit = only_b_hit / total

    print(f"both_hit: {pct_both_hit*100:8.3f}%")
    print(f"only_lru_hit: {pct_only_a_hit*100:8.3f}%")
    print(f"only_minexp_hit: {pct_only_b_hit*100:8.3f}%")

    def avg_gain(cat, key_gain):
        c = stats[cat]["count"]
        if c <= 0:
            return 0.0
        return stats[cat][key_gain] / c
    
    print("\ncount: 行数")
    print(f"            {name_b}_hit    {name_b}_miss")
    print("--------------------------------")
    print(f"{name_a}_hit   {both_hit:8d} {only_a_hit:11d}")
    print(f"{name_a}_miss  {only_b_hit:8d} {both_miss:11d}")

    print("\nTTFT 总和差值(A-B)")
    print(f"          {name_b}_hit        {name_b}_miss")
    print("----------------------------------------")
    print(f"{name_a}_hit   {gain_both_hit:10.6f} {gain_only_a_hit:14.6f}")
    print(f"{name_a}_miss  {gain_only_b_hit:10.6f} {gain_both_miss:14.6f}")

    print("\nNew TTFT 总和差值(A-B)")
    print(f"          {name_b}_hit        {name_b}_miss")
    print("----------------------------------------")
    print(f"{name_a}_hit   {new_gain_both_hit:10.6f} {new_gain_only_a_hit:14.6f}")
    print(f"{name_a}_miss  {new_gain_only_b_hit:10.6f} {new_gain_both_miss:14.6f}")
    
    # New TTFT gain
    avg_both_hit_new   = avg_gain("both_hit", "new_gain")
    avg_only_a_hit_new = avg_gain("only_a_hit", "new_gain")
    avg_only_b_hit_new = avg_gain("only_b_hit", "new_gain")

    weighted_avg_new_gain = (
        pct_both_hit   * avg_both_hit_new +
        pct_only_a_hit * avg_only_a_hit_new +
        pct_only_b_hit * avg_only_b_hit_new
    )

    weighted_total_new_gain = total * weighted_avg_new_gain
    print(f"weighted_total_new_gain (pct-weighted) = {weighted_total_new_gain:.6f}")
    print(f"check(sum of sums) = {(stats['both_hit']['new_gain'] + stats['only_a_hit']['new_gain'] + stats['only_b_hit']['new_gain']):.6f}")

    avg_both_hit   = avg_gain("both_hit", "gain")
    avg_only_a_hit = avg_gain("only_a_hit", "gain")
    avg_only_b_hit = avg_gain("only_b_hit", "gain")

    weighted_avg_gain = (
        pct_both_hit   * avg_both_hit +
        pct_only_a_hit * avg_only_a_hit +
        pct_only_b_hit * avg_only_b_hit
    )

    weighted_total_gain = total * weighted_avg_gain
    print(f"weighted_total_gain (pct-weighted) = {weighted_total_gain:.6f}")
    print(f"check(sum of sums) = {(stats['both_hit']['gain'] + stats['only_a_hit']['gain'] + stats['only_b_hit']['gain']):.6f}")

if __name__ == "__main__":
    baseline_path = "/jizhicfs/marvinhjia/sca/yilin/vllm-dev/examples/test/e2e-exp/LongBench_simulator_lru_full_qps0.007_0.0007_h0_const.jsonl"
    info_lru = build_info_map(baseline_path)
    minexp_files = ["/jizhicfs/marvinhjia/sca/yilin/vllm-dev/examples/test/e2e-exp/LongBench_simulator_minexp_full_qps0.007_0.0007_P90_grad40_const.jsonl"]

    for path in minexp_files:
        exp_name = "qps0.007_0.0007_minexp"
        info_exp = build_info_map(path)
        stats = compute_stats(info_lru, info_exp)
        print_tables(stats, name_a="LRU", name_b=exp_name)