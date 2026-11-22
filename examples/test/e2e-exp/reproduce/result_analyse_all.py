import argparse
import csv
from pathlib import Path

from result_analyse import analyze_jsonl


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[5] / "results"
CSV_FIELDS = [
    "top_group",
    "experiment",
    "relative_path",
    "file",
    "total_items",
    "avg_tbt",
    "avg_ttft",
    "avg_tpot",
    "avg_new_ttft",
    "avg_new_tpot",
    "hit_items",
    "avg_ttft_hit",
    "avg_new_ttft_hit",
    "avg_hit_rate",
    "multi_cache_rate",
    "single_cache_rate",
    "total_time",
    "throughput",
]


def format_row(stats):
    return {
        "top_group": stats["top_group"],
        "experiment": stats["experiment"],
        "relative_path": stats["relative_path"],
        "file": stats["file"],
        "total_items": stats["total"],
        "avg_tbt": f"{stats['avg_tbt']:.6f}",
        "avg_ttft": f"{stats['avg_ttft']:.6f}",
        "avg_tpot": f"{stats['avg_tpot']:.6f}",
        "avg_new_ttft": f"{stats['avg_new_ttft']:.6f}",
        "avg_new_tpot": f"{stats['avg_new_tpot']:.6f}",
        "hit_items": stats["hit_total"],
        "avg_ttft_hit": f"{stats['avg_ttft_hit']:.6f}",
        "avg_new_ttft_hit": f"{stats['avg_new_ttft_hit']:.6f}",
        "avg_hit_rate": f"{stats['avg_hit_rate']:.4%}",
        "multi_cache_rate": f"{stats['multi_cache_rate']:.4%}",
        "single_cache_rate": f"{stats['single_cache_rate']:.4%}",
        "total_time": f"{stats['total_time']:.6f}",
        "throughput": f"{stats['throughput']:.6f}",
    }


def infer_labels(relative_path):
    parts = relative_path.parts
    top_group = parts[0] if parts else ""
    if top_group == "e2e" and len(parts) > 1:
        experiment = parts[1]
    elif top_group == "ablation" and len(parts) > 1:
        experiment = f"ablation/{parts[1]}"
    else:
        experiment = top_group
    return top_group, experiment


def collect_results(results_dir):
    results_root = Path(results_dir)
    jsonl_files = sorted(results_root.rglob("*.jsonl"))
    rows = []

    print(f"Found {len(jsonl_files)} .jsonl files under {results_root}")
    for file_path in jsonl_files:
        relative_path = file_path.relative_to(results_root)
        print(f"Processing {relative_path} ...")
        try:
            stats = analyze_jsonl(file_path)
        except Exception as exc:
            print(f"Error processing {file_path}: {exc}")
            continue

        top_group, experiment = infer_labels(relative_path)
        stats["top_group"] = top_group
        stats["experiment"] = experiment
        stats["relative_path"] = str(relative_path)
        rows.append(stats)
        print(f"Successfully processed ({stats['total']} items)")

    return rows


def write_csv(rows, output_csv):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(format_row(row))

    print(f"Summary saved to {output_path}")


def summarize_all(results_dir=DEFAULT_RESULTS_DIR, output_csv=None):
    results_root = Path(results_dir)
    if not results_root.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_root}")

    output_path = Path(output_csv) if output_csv else results_root / "e2e_results.csv"
    rows = collect_results(results_root)
    write_csv(rows, output_path)
    print(f"Processed {len(rows)} files successfully")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze all jsonl files under results")
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output_csv", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summarize_all(args.results_dir, args.output_csv)
