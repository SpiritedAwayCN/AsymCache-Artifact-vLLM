import os
import csv
import json
import math


def extract_metrics(record):
    choices = record.get("choices", [])
    metrics = {}
    if choices and isinstance(choices[0], dict):
        metrics = choices[0].get("metrics", {})
    return metrics


def is_cache_hit(metrics):
    suffix_locs = metrics.get("suffix_computed_block_locs", [])
    if not suffix_locs:
        return False
    return not all(x == 0 for x in suffix_locs)


def load_jsonl_as_dict(file_path):
    data = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)
            key = (record.get("file"), record.get("line"), record.get("turn"))
            if key in data:
                pass
            data[key] = record
    return data


def analyze_records(records, name="[unknown]"):
    tbt_list = []
    ttft_list = []
    tpot_list = []
    new_ttft_list = []
    arrival_times = []
    last_token_times = []
    total = 0
    multi_cache = 0
    single_cache = 0
    total_hit_rate = 0.0
    total_output_tokens = 0

    for i, record in enumerate(records):
        usage = record.get("usage", {})
        metrics = extract_metrics(record)

        arrival_time = metrics.get("arrival_time")
        prompt_tokens = usage.get("prompt_tokens", 0)
        last_token_time = metrics.get("last_token_time")
        first_token_time = metrics.get("first_token_time")
        schedule_time = metrics.get("first_scheduled_time")
        completion_tokens = usage.get("completion_tokens", 0)
        suffix_locs = metrics.get("suffix_computed_block_locs", [0, 0, 0, 0, 0])

        if arrival_time is None or first_token_time is None:
            print(f"[Error][{name}] Missing field in item {i}:")
            print(json.dumps(metrics, indent=2, ensure_ascii=False))
            raise ValueError(f"Item {i} is missing arrival_time or first_token_time")

        if arrival_time is not None:
            arrival_times.append(arrival_time)
        if last_token_time is not None:
            last_token_times.append(last_token_time)

        ttft = first_token_time - arrival_time
        total_output_tokens += completion_tokens
        new_ttft = first_token_time - schedule_time if schedule_time is not None else None

        if completion_tokens > 0 and last_token_time is not None:
            tpot = (last_token_time - arrival_time) / completion_tokens
        else:
            tpot = None

        if completion_tokens > 1 and last_token_time is not None:
            tbt = (last_token_time - first_token_time) / (completion_tokens - 1)
        else:
            tbt = None

        ttft_list.append(ttft)
        new_ttft_list.append(new_ttft)
        
        if tbt is not None:
            tbt_list.append(tbt)
        if tpot is not None:
            tpot_list.append(tpot)

        # 缓存触发(增加block命中率) —— 延续你原来这块逻辑
        total += 1
        s0, s1, s2, s3 = suffix_locs[:4]
        first_compute_start = suffix_locs[-1]

        if any(x != 0 for x in suffix_locs[:4]):
            multi_cache += 1
        elif all(x == 0 for x in suffix_locs[:4]) and first_compute_start != 0:
            single_cache += 1

        hit_blocks = 0
        num_blocks = math.ceil(prompt_tokens / 64) if prompt_tokens else 0
        if s1 > s0:
            hit_blocks += (s1 - s0) // 64
        if s3 > s2:
            hit_blocks += (s3 - s2) // 64
        if first_compute_start > 0:
            hit_blocks += first_compute_start // 64
        hit_rate = hit_blocks / num_blocks if num_blocks > 0 else 0.0
        total_hit_rate += hit_rate

    # 平均指标
    avg_tbt = sum(tbt_list) / len(tbt_list) if tbt_list else 0.0
    avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0.0
    avg_tpot = sum(tpot_list) / len(tpot_list) if tpot_list else 0.0
    avg_new_ttft = sum(new_ttft_list) / len(new_ttft_list) if new_ttft_list else 0.0

    multi_rate = multi_cache / total if total > 0 else 0.0
    single_rate = single_cache / total if total > 0 else 0.0
    avg_hit_rate = total_hit_rate / total if total > 0 else 0.0

    if arrival_times and last_token_times:
        tpt = max(last_token_times) - min(arrival_times)
        throughput = total_output_tokens / tpt if tpt > 0 else 0.0
    else:
        tpt = 0.0
        throughput = 0.0

    return {
        "name": name,
        "total": total,
        "avg_tbt": avg_tbt,
        "avg_ttft": avg_ttft,
        "avg_tpot": avg_tpot,
        "avg_new_ttft": avg_new_ttft,
        "avg_hit_rate": avg_hit_rate,
        "multi_cache_rate": multi_rate,
        "single_cache_rate": single_rate,
        "total_time": tpt,
        "throughput": throughput,
    }


def analyze_pair_with_lru(lru_dict, minexp_dict, exp_name):
    lru_records_filtered = []
    minexp_records_filtered = []

    all_keys = set(lru_dict.keys()) | set(minexp_dict.keys())
    for key in all_keys:
        lru_rec = lru_dict.get(key)
        exp_rec = minexp_dict.get(key)
        if lru_rec is None or exp_rec is None:
            continue

        lru_metrics = extract_metrics(lru_rec)
        exp_metrics = extract_metrics(exp_rec)

        lru_hit = is_cache_hit(lru_metrics)
        exp_hit = is_cache_hit(exp_metrics)

        # 两边都没命中 -> 丢弃
        if (not lru_hit) and (not exp_hit):
            continue

        lru_records_filtered.append(lru_rec)
        minexp_records_filtered.append(exp_rec)

    lru_stats = analyze_records(lru_records_filtered, name="LRU")
    exp_stats = analyze_records(minexp_records_filtered, name=exp_name)

    return {
        "file": exp_name,
        "total_filtered": exp_stats["total"],  # 使用筛选后的总行数
        "lru_avg_tbt": lru_stats["avg_tbt"],
        "exp_avg_tbt": exp_stats["avg_tbt"],
        "lru_avg_ttft": lru_stats["avg_ttft"],
        "exp_avg_ttft": exp_stats["avg_ttft"],
        "lru_avg_tpot": lru_stats["avg_tpot"],
        "exp_avg_tpot": exp_stats["avg_tpot"],
        "lru_avg_new_ttft": lru_stats["avg_new_ttft"],
        "exp_avg_new_ttft": exp_stats["avg_new_ttft"],
        "lru_avg_hit_rate": lru_stats["avg_hit_rate"],
        "exp_avg_hit_rate": exp_stats["avg_hit_rate"],
        "lru_multi_cache_rate": lru_stats["multi_cache_rate"],
        "exp_multi_cache_rate": exp_stats["multi_cache_rate"],
        "lru_single_cache_rate": lru_stats["single_cache_rate"],
        "exp_single_cache_rate": exp_stats["single_cache_rate"],
        "lru_total_time": lru_stats["total_time"],
        "exp_total_time": exp_stats["total_time"],
        "lru_throughput": lru_stats["throughput"],
        "exp_throughput": exp_stats["throughput"],
    }


def summarize_folder(folder_path, lru_path, output_csv="summary.csv"):
    print(f"Loading LRU baseline from: {lru_path}")
    lru_dict = load_jsonl_as_dict(lru_path)

    results = []
    for filename in os.listdir(folder_path):
        if filename.startswith("LongBench_simulator_minexp_300_qps0.013_0.00013_P90_grad40_const.jsonl") and filename.endswith(".jsonl"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing {filename} ...")
            minexp_dict = load_jsonl_as_dict(file_path)
            stats = analyze_pair_with_lru(lru_dict, minexp_dict, exp_name=filename)
            results.append(stats)

    # 写入 CSV：一行代表「一组 MinExp vs LRU」的对比结果
    with open(output_csv, "a", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=[
            "file",
            "total_filtered",
            "lru_avg_tbt", "exp_avg_tbt",
            "lru_avg_ttft", "exp_avg_ttft",
            "lru_avg_tpot", "exp_avg_tpot",
            "lru_avg_new_ttft", "exp_avg_new_ttft",
            "lru_avg_hit_rate", "exp_avg_hit_rate",
            "lru_multi_cache_rate", "exp_multi_cache_rate",
            "lru_single_cache_rate", "exp_single_cache_rate",
            "lru_total_time", "exp_total_time",
            "lru_throughput", "exp_throughput",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "file": r["file"],
                "total_filtered": r["total_filtered"],
                "lru_avg_tbt": f"{r['lru_avg_tbt']:.6f}",
                "exp_avg_tbt": f"{r['exp_avg_tbt']:.6f}",
                "lru_avg_ttft": f"{r['lru_avg_ttft']:.6f}",
                "exp_avg_ttft": f"{r['exp_avg_ttft']:.6f}",
                "lru_avg_tpot": f"{r['lru_avg_tpot']:.6f}",
                "exp_avg_tpot": f"{r['exp_avg_tpot']:.6f}",
                "lru_avg_new_ttft": f"{r['lru_avg_new_ttft']:.6f}",
                "exp_avg_new_ttft": f"{r['exp_avg_new_ttft']:.6f}",
                "lru_avg_hit_rate": f"{r['lru_avg_hit_rate']:.4%}",
                "exp_avg_hit_rate": f"{r['exp_avg_hit_rate']:.4%}",
                "lru_multi_cache_rate": f"{r['lru_multi_cache_rate']:.4%}",
                "exp_multi_cache_rate": f"{r['exp_multi_cache_rate']:.4%}",
                "lru_single_cache_rate": f"{r['lru_single_cache_rate']:.4%}",
                "exp_single_cache_rate": f"{r['exp_single_cache_rate']:.4%}",
                "lru_total_time": f"{r['lru_total_time']:.6f}",
                "exp_total_time": f"{r['exp_total_time']:.6f}",
                "lru_throughput": f"{r['lru_throughput']:.6f}",
                "exp_throughput": f"{r['exp_throughput']:.6f}",
            })

    print(f"\nSummary saved to {output_csv}")


if __name__ == "__main__":
    folder = "/jizhicfs/marvinhjia/sca/yilin/vllm-dev/examples/test/e2e-exp"
    lru_file = "/jizhicfs/marvinhjia/sca/yilin/vllm-dev/examples/test/e2e-exp/LongBench_simulator_lru_300_qps0.013_0.00013_h0_const.jsonl"
    summarize_folder(folder, lru_path=lru_file, output_csv="summary2.csv")
