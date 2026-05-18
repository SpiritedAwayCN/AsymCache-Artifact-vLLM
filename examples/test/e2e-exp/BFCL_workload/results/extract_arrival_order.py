import csv
import json
import math
import argparse
from pathlib import Path


DEFAULT_FIELDS = [
    "line_no",
    "sample_id",
    "iteration",
    "arrival_time",
    "first_scheduled_time",
    "time_in_queue",
    "first_token_time",
    "finished_time",
    "prompt_blocks",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract request arrival order from vLLM replay jsonl files and "
            "write CSV files sorted by metrics.arrival_time."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input jsonl file or directory containing jsonl files.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        default="arrival_order_csv",
        help="Directory for per-jsonl CSV outputs.",
    )
    parser.add_argument(
        "--pattern",
        default="*.jsonl",
        help="Glob pattern used when --input is a directory.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for jsonl files when --input is a directory.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=64,
        help="Block size used to compute prompt_blocks and hit_rate.",
    )
    return parser.parse_args()


def iter_input_files(input_path: Path, pattern: str, recursive: bool):
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    globber = input_path.rglob if recursive else input_path.glob
    return sorted(p for p in globber(pattern) if p.is_file())


def get_metrics(record):
    choices = record.get("choices") or []
    if choices and isinstance(choices[0], dict):
        return choices[0].get("metrics") or {}
    return {}


def extract_rows(jsonl_path: Path, block_size: int):
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] bad JSON: {jsonl_path}:{line_no}: {exc}")
                continue
            usage = record.get("usage") or {}
            metrics = get_metrics(record)
            arrival_time = metrics.get("arrival_time")
            if arrival_time is None:
                continue
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            prompt_blocks = math.ceil(prompt_tokens / block_size) if prompt_tokens else 0
            rows.append(
                {
                    "line_no": line_no,
                    "sample_id": record.get("sample_id"),
                    "iteration": record.get("local_iteration"),
                    "arrival_time": arrival_time,
                    "first_scheduled_time": metrics.get("first_scheduled_time"),
                    "time_in_queue": metrics.get("time_in_queue"),
                    "first_token_time": metrics.get("first_token_time"),
                    "finished_time": metrics.get("finished_time"),
                    "prompt_blocks": prompt_blocks,
                }
            )
    rows.sort(
        key=lambda r: (
            float("inf") if r["arrival_time"] is None else float(r["arrival_time"]),
            int(r["line_no"]),
        )
    )
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DEFAULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    jsonl_files = iter_input_files(input_path, args.pattern, args.recursive)
    for jsonl_path in jsonl_files:
        rows = extract_rows(
            jsonl_path=jsonl_path,
            block_size=args.block_size,
        )
        output_path = output_dir / f"{jsonl_path.stem}_arrival_order.csv"
        write_csv(output_path, rows)
        print(f"[OK] {jsonl_path} -> {output_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()