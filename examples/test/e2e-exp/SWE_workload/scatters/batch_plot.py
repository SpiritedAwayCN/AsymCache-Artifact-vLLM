import json
import math
import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="../results/120_LRU_qps2.0")
    parser.add_argument("--output","-o", default="gap_blocks_outputs")
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--cache_capacity_blocks", type=float, default=7893)
    parser.add_argument("--session_col", default="instance_id")
    parser.add_argument("--iteration_col", default="iteration")
    parser.add_argument("--local_iteration_col", default="local_iteration")
    parser.add_argument("--order_col", default=None)
    parser.add_argument("--title", default="Gap Blocks Distribution")
    return parser.parse_args()


def load_jsonl(path, args):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            usage = item.get("usage") or {}
            choices = item.get("choices") or []
            metrics = {}
            if choices:
                metrics = choices[0].get("metrics") or {}
            prompt_tokens = usage.get("prompt_tokens")
            if prompt_tokens is None:
                continue
            session_id = item.get(args.session_col)
            if session_id is None:
                continue
            rows.append({
                "line_no": line_no,
                "id": item.get("id"),
                "session_id": session_id,
                "iteration": item.get("iteration"),
                "local_iteration": item.get("local_iteration"),
                "created": item.get("created"),
                "arrival_time": metrics.get("arrival_time"),
                "first_scheduled_time": metrics.get("first_scheduled_time"),
                "time_in_queue": metrics.get("time_in_queue"),
                "finished_time": metrics.get("finished_time"),
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
            })
    return pd.DataFrame(rows)


def sort_requests(df, order_col):
    if order_col:
        if order_col not in df.columns:
            raise ValueError(f"Missing order column: {order_col}")
        return df.sort_values(order_col, kind="mergesort").reset_index(drop=True)
    for candidate in ["arrival_time", "created", "line_no"]:
        if candidate in df.columns and df[candidate].notna().any():
            return df.sort_values(candidate, kind="mergesort").reset_index(drop=True)
    return df.reset_index(drop=True)


def make_batch_output_path(input_file: Path, output_root: Path):
    return output_root / f"{input_file.stem}_gap_blocks_scatter.png"


def plot_one_file(input_file, output_file, args, title=None):
    df = load_jsonl(input_file, args)
    if df.empty:
        raise ValueError(f"No usable rows with usage.prompt_tokens found in {input_file}")
    df = sort_requests(df, args.order_col)
    df["blocks"] = df["prompt_tokens"].apply(lambda x: math.ceil(x / args.block_size))
    df["cumsum_blocks"] = df["blocks"].cumsum()
    df["gap_blocks"] = 0.0
    backward_pairs = 0
    grouped = df.groupby("session_id", sort=False)
    
    for _, group in grouped:
        sort_col = "local_iteration" if group["local_iteration"].notna().all() else "iteration"
        group = group.sort_values(sort_col, kind="mergesort")
        idxs = group.index.tolist()
        for i in range(len(idxs) - 1):
            cur_idx = idxs[i]
            next_idx = idxs[i + 1]
            if next_idx > cur_idx + 1:
                gap = df.loc[next_idx - 1, "cumsum_blocks"] - df.loc[cur_idx, "cumsum_blocks"]
            elif next_idx <= cur_idx:
                backward_pairs += 1
                gap = 0
            else:
                gap = 0
            df.loc[next_idx, "gap_blocks"] = gap
    
    plt.figure(figsize=(20, 8))
    x_axis = range(1, len(df) + 1)
    y_axis = df["gap_blocks"]
    plt.scatter(x_axis, y_axis, s=15, alpha=0.7, label="Gap Blocks")
    plt.axhline(
        y=args.cache_capacity_blocks,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Cache Capacity ({args.cache_capacity_blocks:g} Blocks)",
    )
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(max(1, len(df) // 40)))
    plt.xticks(rotation=45, fontsize=8)
    plt.xlabel("Request Arrival Order")
    plt.ylabel("Gap Blocks (Blocks from Intervening Requests)")
    plt.title(title or args.title)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xlim(0, len(df) + max(20, len(df) // 20))
    plt.tight_layout()
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    above_capacity = int((df["gap_blocks"] > args.cache_capacity_blocks).sum())
    stats = {
        "input": str(input_file),
        "output": str(output_file),
        "rows": len(df),
        "sessions": grouped.ngroups,
        "pairs_not_in_arrival_order": backward_pairs,
        "gap_blocks_gt_cache_capacity": above_capacity,
        "gap_blocks_mean": float(df["gap_blocks"].mean()),
        "gap_blocks_p95": float(df["gap_blocks"].quantile(0.95)),
        "gap_blocks_p99": float(df["gap_blocks"].quantile(0.99)),
    }
    print(f"input: {input_file}")
    print(f"rows: {len(df)}")
    print(f"sessions: {grouped.ngroups}")
    print(f"pairs_not_in_arrival_order: {backward_pairs}")
    print(f"gap_blocks_gt_cache_capacity: {above_capacity}")
    print(f"gap_blocks_mean: {df['gap_blocks'].mean():.3f}")
    print(f"gap_blocks_p95: {df['gap_blocks'].quantile(0.95):.3f}")
    print(f"gap_blocks_p99: {df['gap_blocks'].quantile(0.99):.3f}")
    print(f"saved to: {output_file}")
    print()
    return stats


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_dir():
        raise ValueError(f"Input path does not exist: {input_path}")
    jsonl_files = sorted(input_path.rglob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No jsonl files found under {input_path}")
    print(f"Found {len(jsonl_files)} jsonl files under {input_path}")
    ok = 0
    failed = 0
    for jsonl_file in jsonl_files:
        out_file = make_batch_output_path(jsonl_file, output_path)
        title = f"{args.title} - {jsonl_file.stem}"
        try:
            plot_one_file(jsonl_file, out_file, args, title=title)
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"[WARN] failed: {jsonl_file}: {exc}")
    print("=== Batch Summary ===")
    print(f"processed: {ok}")
    print(f"failed: {failed}")
    print(f"output_dir: {output_path}")


if __name__ == "__main__":
    main()