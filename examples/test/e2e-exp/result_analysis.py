import os
import csv
import json
import math

def analyze_jsonl(file_path):
    tbt_list = []
    ttft_list = []
    tpot_list = []
    arrival_times = []
    last_token_times= []
    total = 0
    multi_cache = 0
    single_cache = 0
    total_hit_rate = 0.0
    total_output_tokens = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            record = json.loads(line)
            usage = record.get("usage", {})
            choices = record.get("choices", [])
            
            metrics = {}
            if choices and isinstance(choices[0], dict):
                metrics = choices[0].get("metrics", {}) 

            arrival_time = metrics.get("arrival_time")
            prompt_tokens = usage.get("prompt_tokens")
            last_token_time = metrics.get("last_token_time")
            first_token_time = metrics.get("first_token_time")
            completion_tokens = usage.get("completion_tokens")
            suffix_locs = metrics.get("suffix_computed_block_locs", [0, 0, 0, 0, 0])

            if arrival_time is None or first_token_time is None:
                print(f"[Error] Missing field in line {i + 1}:")
                print(json.dumps(metrics, indent=2, ensure_ascii=False))
                raise ValueError(f"Line {i + 1} is missing arrival_time or first_token_time")
            
            if arrival_time is not None:
                arrival_times.append(arrival_time)
            if last_token_time is not None:
                last_token_times.append(last_token_time)
            
            ttft = first_token_time - arrival_time
            total_output_tokens += completion_tokens
            
            if completion_tokens > 0:
                tpot = (last_token_time - arrival_time) / completion_tokens
            else:
                tpot = None
            
            if completion_tokens > 1:
                tbt = (last_token_time - first_token_time) / (completion_tokens - 1)
            else:
                tbt = None

            ttft_list.append(ttft)
            if tbt is not None:
                tbt_list.append(tbt)
            if tpot is not None:
                tpot_list.append(tpot)
           
            # 缓存触发(增加block命中率)
            total += 1
            s0, s1, s2, s3 = suffix_locs[:4]
            first_compute_start = suffix_locs[-1]

            if any(x != 0 for x in suffix_locs[:4]):
                multi_cache += 1
            elif all(x == 0 for x in suffix_locs[:4]) and first_compute_start != 0:
                single_cache += 1
            
            hit_blocks = 0
            num_blocks = math.ceil(prompt_tokens / 64)
            if s1 > s0:
                hit_blocks += (s1 - s0) // 64
            if s3 > s2:
                hit_blocks += (s3 - s2) // 64
            if first_compute_start > 0:
                hit_blocks += first_compute_start // 64
            hit_rate = hit_blocks / num_blocks if num_blocks > 0 else 0.0
            total_hit_rate += hit_rate

    # 平均指标
    avg_tbt = sum(tbt_list) / len(tbt_list) if tbt_list else 0
    avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0
    avg_tpot = sum(tpot_list) / len(tpot_list) if tpot_list else 0
  
    multi_rate = multi_cache / total if total > 0 else 0
    single_rate = single_cache / total if total > 0 else 0
    avg_hit_rate = total_hit_rate / total if total > 0 else 0.0
    
    if arrival_times and last_token_times:
        tpt = max(last_token_times) - min(arrival_times)
        throughput = total_output_tokens / tpt if tpt > 0 else 0
    else:
        tpt = 0
        throughput = 0
    
    return {
        "file": os.path.basename(file_path),
        "total": total,
        "avg_tbt": avg_tbt,
        "avg_ttft": avg_ttft,
        "avg_tpot": avg_tpot,
        "avg_hit_rate": avg_hit_rate,
        "multi_cache_rate": multi_rate,
        "single_cache_rate": single_rate,
        "total_time": tpt,
        "throughput": throughput
    }


def summarize_folder(folder_path, output_csv="summary.csv"):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith("P90_slope100.jsonl"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing {filename} ...")
            stats = analyze_jsonl(file_path)
            results.append(stats)

    # 写入 CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=[
            "file", "total_items", "avg_tbt", "avg_ttft", "avg_tpot", "avg_hit_rate",
            "multi_cache_rate", "single_cache_rate", "total_time", "throughput"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "file": r["file"],
                "total_items": r["total"],
                "avg_tbt": f"{r['avg_tbt']:.6f}",
                "avg_ttft": f"{r['avg_ttft']:.6f}",
                "avg_tpot": f"{r['avg_tpot']:.6f}",
                "avg_hit_rate": f"{r['avg_hit_rate']:.4%}",
                "multi_cache_rate": f"{r['multi_cache_rate']:.4%}",
                "single_cache_rate": f"{r['single_cache_rate']:.4%}",
                "total_time": f"{r['total_time']:.6f}",
                "throughput": f"{r['throughput']:.6f}",
            })

    print(f"\nSummary saved to {output_csv}")


if __name__ == "__main__":
    folder = "/jizhicfs/marvinhjia/sca/yilin/vllm-dev/examples/test/e2e-exp"
    summarize_folder(folder, output_csv="summary.csv")