import csv
import argparse
import pandas as pd
from pathlib import Path


OUTPUT_FIELDS = [
    "iteration",
    "count",
    "first_arrival_rank",
    "last_arrival_rank",
    "mean_prompt_blocks",
    "median_prompt_blocks",
    "p90_prompt_blocks",
    "p95_prompt_blocks",
    "max_prompt_blocks",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize prompt_blocks by iteration from arrival-order CSV files.")
    parser.add_argument("--input", "-i", required=True, help="CSV file or directory.")
    parser.add_argument("--output_csv", "-o", default="iteration_prompt_blocks_summary.csv")
    parser.add_argument("--pattern", default="*_arrival_order.csv")
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def iter_input_files(input_path: Path, pattern: str, recursive: bool):
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    globber = input_path.rglob if recursive else input_path.glob
    return sorted(p for p in globber(pattern) if p.is_file())


def summarize_file(path: Path):
    df = pd.read_csv(path)
    required = ["iteration", "arrival_time", "prompt_blocks"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    df = df.copy()
    df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")
    df["arrival_time"] = pd.to_numeric(df["arrival_time"], errors="coerce")
    df["prompt_blocks"] = pd.to_numeric(df["prompt_blocks"], errors="coerce")
    df = df.dropna(subset=["iteration", "arrival_time", "prompt_blocks"])
    df["iteration"] = df["iteration"].astype(int)

    sort_cols = ["arrival_time"]
    if "line_no" in df.columns:
        sort_cols.append("line_no")
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    df["arrival_rank"] = range(1, len(df) + 1)

    rows = []
    for iteration, group in df.groupby("iteration", sort=True):
        blocks = group["prompt_blocks"]
        rows.append(
            {
                "iteration": int(iteration),
                "count": int(len(group)),
                "first_arrival_rank": int(group["arrival_rank"].min()),
                "last_arrival_rank": int(group["arrival_rank"].max()),
                "mean_prompt_blocks": float(blocks.mean()),
                "median_prompt_blocks": float(blocks.median()),
                "p90_prompt_blocks": float(blocks.quantile(0.90)),
                "p95_prompt_blocks": float(blocks.quantile(0.95)),
                "max_prompt_blocks": int(blocks.max()),
            }
        )
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows):
    for row in rows:
        print(
            f"{row['iteration']},{row['count']},"
            f"{row['mean_prompt_blocks']:.2f},"
            f"{row['median_prompt_blocks']:.2f},"
            f"{row['p95_prompt_blocks']:.2f}"
        )


def main():
    args = parse_args()
    files = iter_input_files(Path(args.input), args.pattern, args.recursive)
    if not files:
        raise ValueError(f"No files matched: {args.input} / {args.pattern}")
    all_rows = []
    for path in files:
        rows = summarize_file(path)
        all_rows.extend(rows)
        print(f"[OK] {path}: {len(rows)} iterations")
    write_csv(Path(args.output_csv), all_rows)
    print(f"[OK] wrote {args.output_csv}")
    print_table(all_rows)


if __name__ == "__main__":
    main()