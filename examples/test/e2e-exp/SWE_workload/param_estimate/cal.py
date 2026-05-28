import csv
import sys
import json
import math
import argparse
import numpy as np
from pathlib import Path


max_field_size = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_field_size)
        break
    except OverflowError:
        max_field_size = int(max_field_size / 10)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--instance_col", default="instance_id")
    parser.add_argument("--iteration_col", default="iteration")
    parser.add_argument("--prompt_tokens_col", default="prompt_tokens")
    parser.add_argument("--completion_tokens_col", default="completion_tokens")
    parser.add_argument("--tool_sleep_col", default="tool_sleep_seconds")
    parser.add_argument("--input_percentile", type=float, default=90.0)
    parser.add_argument("--lifespan_percentiles", default="95,99")
    parser.add_argument("--measured_no_cache_latency", type=float, required=True)
    parser.add_argument("--beta_divisor", type=float, default=40.0)
    parser.add_argument("--b_scale", type=float, default=-0.975)
    parser.add_argument("--A", type=float, default=0.0)
    return parser.parse_args()


def as_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default):
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def percentile_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def empirical_cdf_at(sorted_values, value):
    return np.searchsorted(sorted_values, value, side="right") / len(sorted_values)


def detect_input_format(path: Path, input_format: str):
    if input_format != "auto":
        return input_format
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    return "jsonl"


def load_from_csv(path: Path, args):
    records = []
    sessions = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = [
            args.instance_col,
            args.iteration_col,
            args.prompt_tokens_col,
            args.completion_tokens_col,
            args.tool_sleep_col,
        ]
        for col in required:
            if col not in reader.fieldnames:
                raise ValueError(f"Missing CSV column: {col}")
        for row_no, row in enumerate(reader, start=2):
            instance_id = row[args.instance_col]
            sessions.add(instance_id)
            records.append(
                {
                    "instance_id": instance_id,
                    "iteration": row[args.iteration_col],
                    "prompt_tokens": as_float(row[args.prompt_tokens_col]),
                    "completion_tokens": as_float(row[args.completion_tokens_col]),
                    "tool_latency": max(0.0, as_float(row[args.tool_sleep_col])),
                    "tool_call_count": as_int(row.get("tool_call_count"), 0),
                    "source_line_no": row_no,
                }
            )
    
    records.sort(
        key=lambda r: (
            r["instance_id"],
            as_int(r["iteration"], 10**18),
            r["source_line_no"],
        )
    )
    return records, len(sessions)


def solve_params(lifespan_samples, percentiles, args):
    sorted_lifespans = np.sort(np.asarray(lifespan_samples, dtype=float))
    rows = []
    for p in percentiles:
        val = float(np.percentile(sorted_lifespans, p))
        y_val = empirical_cdf_at(sorted_lifespans, val)
        if not 0.0 < y_val < 1.0:
            y_val = min(max(p / 100.0, 1e-12), 1.0 - 1e-12)
        alpha = -val / math.log(y_val)
        beta = alpha / args.beta_divisor
        b_value = args.b_scale * val
        rows.append(
            {
                "percentile": p,
                "lifespan": val,
                "cdf": y_val,
                "B": b_value,
                "alpha": alpha,
                "beta": beta,
                "A": args.A,
                "config_line": (
                    f"{b_value:.6e} {alpha:.6e} {beta:.6e} {args.A}"
                ),
            }
        )
    return rows


def main():
    args = parse_args()
    input_path = Path(args.input)
    records, num_sessions = load_from_csv(input_path, args)
    if not records:
        raise ValueError(f"No SWE llm_call records loaded from {input_path}")
    prompt_tokens = np.asarray([r["prompt_tokens"] for r in records], dtype=float)
    completion_tokens = np.asarray([r["completion_tokens"] for r in records], dtype=float)
    tool_latencies = np.asarray([r["tool_latency"] for r in records], dtype=float)
    tool_call_counts = np.asarray([r["tool_call_count"] for r in records], dtype=float)
    lifespan_samples = args.measured_no_cache_latency + tool_latencies
    rows = solve_params(
        lifespan_samples=lifespan_samples,
        percentiles=percentile_list(args.lifespan_percentiles),
        args=args,
    )
    print("=== Dataset Profile ===")
    print(f"sessions: {num_sessions}")
    print(f"llm_calls: {len(records)}")
    print(
        f"input_tokens_p{args.input_percentile:g}: "
        f"{np.percentile(prompt_tokens, args.input_percentile):.3f}"
    )
    print(f"output_tokens_avg: {np.mean(completion_tokens):.3f}")
    print(f"tool_calls_avg_per_llm: {np.mean(tool_call_counts):.6f}")
    print("=== Lifespan Estimate ===")
    print(f"measured_vllm_no_cache_latency: {args.measured_no_cache_latency:.6f}")
    print(f"tool_latency_avg: {np.mean(tool_latencies):.6f}")
    print(f"tool_latency_p50: {np.percentile(tool_latencies, 50):.6f}")
    print(f"tool_latency_p90: {np.percentile(tool_latencies, 90):.6f}")
    print(f"tool_latency_p95: {np.percentile(tool_latencies, 95):.6f}")
    print(f"tool_latency_p99: {np.percentile(tool_latencies, 99):.6f}")
    print(f"lifespan_avg: {np.mean(lifespan_samples):.6f}")
    print(f"lifespan_p95: {np.percentile(lifespan_samples, 95):.6f}")
    print(f"lifespan_p99: {np.percentile(lifespan_samples, 99):.6f}")
    print("=== Config Lines ===")
    for row in rows:
        print(
            f"p{row['percentile']:g}: "
            f"lifespan={row['lifespan']:.6f}, cdf={row['cdf']:.6f}, "
            f"B={row['B']:.6e}, alpha={row['alpha']:.6e}, beta={row['beta']:.6e}"
        )
        print(row["config_line"])


if __name__ == "__main__":
    main()