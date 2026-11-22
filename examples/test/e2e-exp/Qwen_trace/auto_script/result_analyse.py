import os
import csv
import json
import math
import argparse
from pathlib import Path
from typing import Any, Optional


DEFAULT_RESULTS_DIR = (
    "/jizhicfs/pkuhetu/sca/yilin/vllm-dev/examples/test/e2e-exp/"
    "Qwen_trace/auto_script/70B_results"
)


def as_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def average(values):
    return sum(values) / len(values) if values else 0.0


def percentile(values, percent):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return (
        ordered[lower_index] * (1 - weight)
        + ordered[upper_index] * weight
    )


def normalize_cache_metrics(metrics, block_size, prompt_tokens):
    raw_locs = metrics.get("suffix_computed_block_locs")
    if not isinstance(raw_locs, (list, tuple)):
        raw_locs = []

    suffix_locs = []
    for value in raw_locs[:4]:
        parsed = as_number(value)
        suffix_locs.append(max(0, int(parsed)) if parsed is not None else 0)
    suffix_locs.extend([0] * (4 - len(suffix_locs)))
    s0, s1, s2, s3 = suffix_locs

    explicit_first_compute = as_number(metrics.get("first_compute_start"))
    if explicit_first_compute is not None:
        first_compute_start = max(0, int(explicit_first_compute))
    elif len(raw_locs) >= 5:
        packed_first_compute = as_number(raw_locs[4])
        first_compute_start = (
            max(0, int(packed_first_compute))
            if packed_first_compute is not None
            else 0
        )
    else:
        first_compute_start = 0

    suffix_hit = (s1 > s0) or (s3 > s2)
    prefix_hit = first_compute_start > 0
    hit_blocks = first_compute_start // block_size
    if s1 > s0:
        hit_blocks += (s1 - s0) // block_size
    if s3 > s2:
        hit_blocks += (s3 - s2) // block_size

    prompt_blocks = (
        math.ceil(prompt_tokens / block_size) if prompt_tokens > 0 else 0
    )
    # Cache locations should be disjoint and block aligned. The cap prevents
    # malformed or overlapping metadata from producing a rate above 100%.
    hit_blocks = min(hit_blocks, prompt_blocks)
    return {
        "is_hit": prefix_hit or suffix_hit,
        "is_multi": suffix_hit,
        "is_single": prefix_hit and not suffix_hit,
        "hit_blocks": hit_blocks,
        "prompt_blocks": prompt_blocks,
    }


def analyze_jsonl(file_path, block_size=64):
    tbt_list = []
    ttft_list = []
    tpot_list = []
    new_ttft_list = []
    new_tpot_list = []
    arrival_times = []
    last_token_times = []
    ttft_hit_list = []
    new_ttft_hit_list = []

    total = 0
    multi_cache = 0
    single_cache = 0
    hit_total = 0
    total_hit_blocks = 0
    total_prompt_blocks = 0
    total_output_tokens = 0

    with open(file_path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"Invalid JSON at line {line_number}: {error}")
                continue
            if record.get("error"):
                print(f"Request failed in line {line_number}; skipped")
                continue

            usage = record.get("usage")
            choices = record.get("choices")
            if not isinstance(usage, dict) or not isinstance(choices, list) or not choices:
                print(f"Missing usage/choices in line {line_number}")
                continue
            first_choice = choices[0] if isinstance(choices[0], dict) else {}
            metrics = first_choice.get("metrics")
            if not isinstance(metrics, dict):
                print(f"Missing choices[0].metrics in line {line_number}")
                continue

            arrival_time = as_number(metrics.get("arrival_time"))
            first_token_time = as_number(metrics.get("first_token_time"))
            first_scheduled_time = as_number(metrics.get("first_scheduled_time"))
            last_token_time = as_number(metrics.get("last_token_time"))
            prompt_tokens = as_number(usage.get("prompt_tokens"))
            completion_tokens = as_number(usage.get("completion_tokens"))

            if prompt_tokens is None:
                prompt_tokens = as_number(record.get("input_length"))
            if completion_tokens is None:
                completion_tokens = as_number(record.get("output_length"))
            if (
                arrival_time is None
                or first_token_time is None
                or first_scheduled_time is None
                or prompt_tokens is None
                or completion_tokens is None
            ):
                print(
                    f"Missing timing/token field in line {line_number}: "
                    "arrival_time, first_token_time, first_scheduled_time, "
                    "prompt_tokens, or completion_tokens"
                )
                continue

            prompt_tokens = max(0, int(prompt_tokens))
            completion_tokens = max(0, int(completion_tokens))
            total += 1
            total_output_tokens += completion_tokens
            arrival_times.append(arrival_time)
            if last_token_time is not None:
                last_token_times.append(last_token_time)

            ttft = first_token_time - arrival_time
            new_ttft = first_token_time - first_scheduled_time
            ttft_list.append(ttft)
            new_ttft_list.append(new_ttft)

            if last_token_time is not None and completion_tokens > 0:
                tpot_list.append(
                    (last_token_time - arrival_time) / completion_tokens
                )
                new_tpot_list.append(
                    (last_token_time - first_scheduled_time) / completion_tokens
                )
            if last_token_time is not None and completion_tokens > 1:
                tbt_list.append(
                    (last_token_time - first_token_time)
                    / (completion_tokens - 1)
                )

            cache = normalize_cache_metrics(
                metrics, block_size, prompt_tokens
            )
            total_hit_blocks += cache["hit_blocks"]
            total_prompt_blocks += cache["prompt_blocks"]
            if cache["is_multi"]:
                multi_cache += 1
            elif cache["is_single"]:
                single_cache += 1
            if cache["is_hit"]:
                hit_total += 1
                ttft_hit_list.append(ttft)
                new_ttft_hit_list.append(new_ttft)

    avg_hit_rate = (
        total_hit_blocks / total_prompt_blocks
        if total_prompt_blocks
        else 0.0
    )
    multi_rate = multi_cache / total if total else 0.0
    single_rate = single_cache / total if total else 0.0

    if arrival_times and last_token_times:
        total_time = max(last_token_times) - min(arrival_times)
        throughput = total_output_tokens / total_time if total_time > 0 else 0.0
    else:
        total_time = 0.0
        throughput = 0.0

    return {
        "file": os.path.basename(file_path),
        "total": total,
        "avg_tbt": average(tbt_list),
        "avg_ttft": average(ttft_list),
        "p99_ttft": percentile(ttft_list, 99),
        "p95_ttft": percentile(ttft_list, 95),
        "p90_ttft": percentile(ttft_list, 90),
        "avg_hit_rate": avg_hit_rate,
        "multi_cache_rate": multi_rate,
        "single_cache_rate": single_rate,
        "avg_tpot": average(tpot_list),
        "avg_new_ttft": average(new_ttft_list),
        "avg_new_tpot": average(new_tpot_list),
        "hit_total": hit_total,
        "avg_ttft_hit": average(ttft_hit_list),
        "avg_new_ttft_hit": average(new_ttft_hit_list),
        "total_time": total_time,
        "throughput": throughput,
    }


def summarize_folder(folder_path, output_csv="summary.csv", block_size=64):
    results = []
    folder = Path(folder_path)
    jsonl_files = sorted(folder.rglob("traceA_*.jsonl"))
    print(f"Found {len(jsonl_files)} traceA_*.jsonl files to process\n")
    for file_path in jsonl_files:
        print(f"Processing {file_path.relative_to(folder)} ...")
        try:
            stats = analyze_jsonl(file_path, block_size)
            stats["file"] = str(file_path.relative_to(folder))
            results.append(stats)
            print(f"Successfully processed ({stats['total']} items)")
        except Exception as error:
            print(f"Error processing {file_path}: {error}")
    print(f"\nProcessed {len(results)} files successfully")
    if not results:
        return

    fieldnames = [
        "file",
        "total_items",
        "avg_tbt",
        "avg_ttft",
        "p99_ttft",
        "p95_ttft",
        "p90_ttft",
        "avg_hit_rate",
        "multi_cache_rate",
        "single_cache_rate",
        "avg_tpot",
        "avg_new_ttft",
        "avg_new_tpot",
        "hit_items",
        "avg_ttft_hit",
        "avg_new_ttft_hit",
        "total_time",
        "throughput",
    ]
    output_path = Path(output_csv)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "file": result["file"],
                    "total_items": result["total"],
                    "avg_tbt": f"{result['avg_tbt']:.6f}",
                    "avg_ttft": f"{result['avg_ttft']:.6f}",
                    "p99_ttft": f"{result['p99_ttft']:.6f}",
                    "p95_ttft": f"{result['p95_ttft']:.6f}",
                    "p90_ttft": f"{result['p90_ttft']:.6f}",
                    "avg_hit_rate": f"{result['avg_hit_rate']:.4%}",
                    "multi_cache_rate": f"{result['multi_cache_rate']:.4%}",
                    "single_cache_rate": f"{result['single_cache_rate']:.4%}",
                    "avg_tpot": f"{result['avg_tpot']:.6f}",
                    "avg_new_ttft": f"{result['avg_new_ttft']:.6f}",
                    "avg_new_tpot": f"{result['avg_new_tpot']:.6f}",
                    "hit_items": result["hit_total"],
                    "avg_ttft_hit": f"{result['avg_ttft_hit']:.6f}",
                    "avg_new_ttft_hit": f"{result['avg_new_ttft_hit']:.6f}",
                    "total_time": f"{result['total_time']:.6f}",
                    "throughput": f"{result['throughput']:.6f}",
                }
            )
    print(f"\nSummary saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output_csv", type=str, default="final_summary.csv")
    parser.add_argument("--block-size", type=int, default=64, help="vLLM KV-cache block size used by the experiment")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")
    summarize_folder(args.folder, args.output_csv, args.block_size)
