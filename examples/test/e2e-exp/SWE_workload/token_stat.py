import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


max_field_size = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_field_size)
        break
    except OverflowError:
        max_field_size = int(max_field_size / 10)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute prompt/output token statistics for SWE jsonl or CSV data."
    )
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument(
        "--input_format",
        choices=["auto", "jsonl", "csv"],
        default="auto",
    )
    parser.add_argument("--prompt_tokens_col", default="prompt_tokens")
    parser.add_argument("--completion_tokens_col", default="completion_tokens")
    return parser.parse_args()


def detect_input_format(path, input_format):
    if input_format != "auto":
        return input_format
    return "csv" if Path(path).suffix.lower() == ".csv" else "jsonl"


def as_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_jsonl(path, args):
    prompt_tokens = []
    output_tokens = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") not in (None, "llm_call"):
                continue

            usage = record.get("usage") or {}
            prompt = record.get(args.prompt_tokens_col, usage.get("prompt_tokens"))
            output = record.get(
                args.completion_tokens_col,
                usage.get("completion_tokens"),
            )
            prompt = as_float(prompt)
            output = as_float(output)
            if prompt is None or output is None:
                continue
            prompt_tokens.append(prompt)
            output_tokens.append(output)
    return prompt_tokens, output_tokens


def load_csv(path, args):
    prompt_tokens = []
    output_tokens = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for col in [args.prompt_tokens_col, args.completion_tokens_col]:
            if col not in reader.fieldnames:
                raise ValueError(f"Missing CSV column: {col}")
        for row in reader:
            prompt = as_float(row.get(args.prompt_tokens_col))
            output = as_float(row.get(args.completion_tokens_col))
            if prompt is None or output is None:
                continue
            prompt_tokens.append(prompt)
            output_tokens.append(output)
    return prompt_tokens, output_tokens


def print_stats(name, values):
    arr = np.asarray(values, dtype=float)
    print(f"{name}_count: {len(arr)}")
    print(f"{name}_mean: {np.mean(arr):.3f}")
    print(f"{name}_p50: {np.percentile(arr, 50):.3f}")
    print(f"{name}_p90: {np.percentile(arr, 90):.3f}")
    print(f"{name}_p95: {np.percentile(arr, 95):.3f}")
    print(f"{name}_p99: {np.percentile(arr, 99):.3f}")
    print(f"{name}_min: {np.min(arr):.3f}")
    print(f"{name}_max: {np.max(arr):.3f}")


def main():
    args = parse_args()
    input_format = detect_input_format(args.input, args.input_format)
    if input_format == "csv":
        prompt_tokens, output_tokens = load_csv(args.input, args)
    else:
        prompt_tokens, output_tokens = load_jsonl(args.input, args)

    if not prompt_tokens:
        raise ValueError(f"No token records found in {args.input}")

    print(f"input: {args.input}")
    print(f"input_format: {input_format}")
    print_stats("prompt_tokens", prompt_tokens)
    print_stats("output_tokens", output_tokens)


if __name__ == "__main__":
    main()
