import os
import csv
import json
import math
from pathlib import Path


def analyze_jsonl(file_path):
    tbt_list = []
    ttft_list = []
    tpot_list = []
    arrival_times = []
    new_ttft_list = []
    new_tpot_list = []
    last_token_times = []
    total = 0
    multi_cache = 0
    single_cache = 0
    total_hit_rate = 0.0
    total_output_tokens = 0
    session_bounds = {}

    hit_total = 0
    ttft_hit_list = []
    new_ttft_hit_list = []

    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            usage = record.get("usage", {})
            choices = record.get("choices", [])
            metrics = {}
            if choices and isinstance(choices[0], dict):
                metrics = choices[0].get("metrics", {})
            arrival_time = metrics.get("arrival_time")
            if arrival_time is None:
                print(f"Missing field in line {line_no}: arrival_time")
                continue
            rows.append((float(arrival_time), line_no, record))

    rows.sort(key=lambda x: (x[0], x[1]))
    for _, line_no, record in rows:
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
        scheduler_time = metrics.get("first_scheduled_time")
        suffix_locs = metrics.get("suffix_computed_block_locs", [0, 0, 0, 0, 0])
        session_id = record.get("instance_id")
        if (
            arrival_time is None
            or first_token_time is None
            or scheduler_time is None
            or last_token_time is None
            or prompt_tokens is None
            or completion_tokens is None
        ):
            print(f"Missing field in line {line_no}:")
            print(
                f"Line {line_no} is missing arrival_time, first_token_time, "
                "last_token_time, scheduler_time, prompt_tokens, or completion_tokens"
            )
            continue
        if arrival_time is not None:
            arrival_times.append(arrival_time)
        if last_token_time is not None:
            last_token_times.append(last_token_time)
        if session_id is not None and arrival_time is not None and last_token_time is not None:
            if session_id not in session_bounds:
                session_bounds[session_id] = [arrival_time, last_token_time]
            else:
                session_bounds[session_id][0] = min(session_bounds[session_id][0], arrival_time)
                session_bounds[session_id][1] = max(session_bounds[session_id][1], last_token_time)
        ttft = first_token_time - arrival_time
        total_output_tokens += completion_tokens
        new_ttft = first_token_time - scheduler_time
        if completion_tokens > 0:
            tpot = (last_token_time - arrival_time) / completion_tokens
            new_tpot = (last_token_time - scheduler_time) / completion_tokens
        else:
            tpot = None
            new_tpot = None
        if completion_tokens > 1:
            tbt = (last_token_time - first_token_time) / (completion_tokens - 1)
        else:
            tbt = None
        ttft_list.append(ttft)
        if tbt is not None:
            tbt_list.append(tbt)
        if tpot is not None:
            tpot_list.append(tpot)
        if new_ttft is not None:
            new_ttft_list.append(new_ttft)
        if new_tpot is not None:
            new_tpot_list.append(new_tpot)
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
        is_hit = any(x != 0 for x in suffix_locs)
        if is_hit:
            hit_total += 1
            ttft_hit_list.append(ttft)
            new_ttft_hit_list.append(new_ttft)

    avg_tbt = sum(tbt_list) / len(tbt_list) if tbt_list else 0
    avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0
    avg_tpot = sum(tpot_list) / len(tpot_list) if tpot_list else 0
    avg_new_ttft = sum(new_ttft_list) / len(new_ttft_list) if new_ttft_list else 0
    avg_new_tpot = sum(new_tpot_list) / len(new_tpot_list) if new_tpot_list else 0
    avg_ttft_hit = sum(ttft_hit_list) / len(ttft_hit_list) if ttft_hit_list else 0
    avg_new_ttft_hit = (
        sum(new_ttft_hit_list) / len(new_ttft_hit_list)
        if new_ttft_hit_list else 0
    )
    job_latencies = [
        end_time - start_time
        for start_time, end_time in session_bounds.values()
        if end_time >= start_time
    ]
    session_count = len(job_latencies)
    avg_job_latency = sum(job_latencies) / session_count if session_count else 0
    max_job_latency = max(job_latencies) if job_latencies else 0
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
        "avg_new_ttft": avg_new_ttft,
        "avg_new_tpot": avg_new_tpot,
        "hit_total": hit_total,
        "avg_ttft_hit": avg_ttft_hit,
        "avg_new_ttft_hit": avg_new_ttft_hit,
        "avg_hit_rate": avg_hit_rate,
        "multi_cache_rate": multi_rate,
        "single_cache_rate": single_rate,
        "session_count": session_count,
        "avg_job_latency": avg_job_latency,
        "max_job_latency": max_job_latency,
        "total_time": tpt,
        "throughput": throughput,
    }


def summarize_folder(folder_path):
    results = []
    folder = Path(folder_path)
    jsonl_files = sorted(folder.rglob("*.jsonl"))
    print(f"Found {len(jsonl_files)} .jsonl files to process\n")
    for file_path in jsonl_files:
        print(f"Processing {file_path.relative_to(folder)} ...")
        try:
            stats = analyze_jsonl(file_path)
            stats["file"] = folder.name
            results.append(stats)
            print(f"Successfully processed ({stats['total']} items)")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    print(f"\nProcessed {len(results)} files successfully")
    return results


def write_summary_csv(results, output_csv="summary.csv"):
    if not results:
        return
    with open(output_csv, "a", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=[
            "file", "total_items", "avg_tbt", "avg_ttft", "avg_tpot",
            "avg_new_ttft", "avg_new_tpot", "hit_items", "avg_ttft_hit",
            "avg_new_ttft_hit", "avg_hit_rate", "multi_cache_rate",
            "single_cache_rate", "session_count", "avg_job_latency",
            "max_job_latency", "total_time", "throughput"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "file": r["file"],
                "total_items": r["total"],
                "avg_tbt": f"{r['avg_tbt']:.6f}",
                "avg_ttft": f"{r['avg_ttft']:.6f}",
                "avg_tpot": f"{r['avg_tpot']:.6f}",
                "avg_new_ttft": f"{r['avg_new_ttft']:.6f}",
                "avg_new_tpot": f"{r['avg_new_tpot']:.6f}",
                "hit_items": r["hit_total"],
                "avg_ttft_hit": f"{r['avg_ttft_hit']:.6f}",
                "avg_new_ttft_hit": f"{r['avg_new_ttft_hit']:.6f}",
                "avg_hit_rate": f"{r['avg_hit_rate']:.4%}",
                "multi_cache_rate": f"{r['multi_cache_rate']:.4%}",
                "single_cache_rate": f"{r['single_cache_rate']:.4%}",
                "session_count": r["session_count"],
                "avg_job_latency": f"{r['avg_job_latency']:.6f}",
                "max_job_latency": f"{r['max_job_latency']:.6f}",
                "total_time": f"{r['total_time']:.6f}",
                "throughput": f"{r['throughput']:.6f}",
            })
    print(f"\nSummary saved to {output_csv}")


if __name__ == "__main__":
    folders = [
        # "./100_LRU_qps2.0",
        # "./100_MWF_qps2.0",
        # "./120_LRU_qps2.0",
        # "./120_MWF_qps2.0",
        "./100_CONTINUUM_qps2.0",
        "./100_CONTINUUM_MWF_qps2.0"
        # "./CONTINUUM_qps3.0",
        # "./CONTINUUM_MWF_qps3.0",
        # "./LRU_qps2.0",
        # "./MWF_qps2.0",
        # "./CONTINUUM_qps2.0",
        # "./CONTINUUM_MWF_qps2.0",
    ]
    all_results = []
    for folder in folders:
        all_results.extend(summarize_folder(folder))
    write_summary_csv(all_results, output_csv="SWE_summary.csv")