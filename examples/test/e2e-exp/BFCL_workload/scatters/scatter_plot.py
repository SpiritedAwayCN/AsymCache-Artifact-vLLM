import argparse
import json
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="../results/MWF_qps3.0/MWF_replay_output_qps3.0.jsonl")
    parser.add_argument("--output", "-o", default="qps3.0_gap_blocks_scatter.png")
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--cache_capacity_blocks", type=float, default=7893)
    parser.add_argument("--sample_col", default="sample_id")
    parser.add_argument("--turn_col", default="turn_id")
    parser.add_argument("--iteration_col", default="iteration")
    parser.add_argument("--order_col", default=None)
    parser.add_argument("--title", default="Gap Blocks Distribution(qps3.0)")
    return parser.parse_args()


def load_jsonl(path):
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

            rows.append({
                "line_no": line_no,
                "id": item.get("id"),
                "sample_id": item.get("sample_id"),
                "turn_id": item.get("turn_id", 0),
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


def main():
    args = parse_args()
    df = load_jsonl(args.input)
    if df.empty:
        raise ValueError(f"No usable rows with usage.prompt_tokens found in {args.input}")

    df = sort_requests(df, args.order_col)
    df["blocks"] = df["prompt_tokens"].apply(lambda x: math.ceil(x / args.block_size))
    df["cumsum_blocks"] = df["blocks"].cumsum()
    df["gap_blocks"] = 0.0

    grouped = df.groupby([args.sample_col, args.turn_col], sort=False)
    for _, group in grouped:
        group = group.sort_values(args.iteration_col)
        idxs = group.index.tolist()
        for i in range(len(idxs) - 1):
            cur_idx = idxs[i]
            next_idx = idxs[i + 1]
            if next_idx > cur_idx + 1:
                gap = df.loc[next_idx - 1, "cumsum_blocks"] - df.loc[cur_idx, "cumsum_blocks"]
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
    plt.title(args.title)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xlim(0, len(df) + max(20, len(df) // 20))
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)

    print(f"rows: {len(df)}")
    print(f"sessions: {grouped.ngroups}")
    print(f"gap_blocks_mean: {df['gap_blocks'].mean():.3f}")
    print(f"gap_blocks_p95: {df['gap_blocks'].quantile(0.95):.3f}")
    print(f"gap_blocks_p99: {df['gap_blocks'].quantile(0.99):.3f}")
    print(f"saved to: {args.output}")


if __name__ == "__main__":
    main()
