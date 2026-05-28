import re
import json
import math
import argparse
import numpy as np
from datetime import datetime
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turn_col", default="turn_id")
    parser.add_argument("--sample_col", default="sample_id")
    parser.add_argument("--iteration_col", default="iteration")
    parser.add_argument("--timestamp_col", default="timestamp")
    parser.add_argument("--lifespan_percentiles", default="95,99")
    parser.add_argument("--latency_col", default="latency_seconds")
    parser.add_argument("--prompt_tokens_col", default="prompt_tokens")
    parser.add_argument("--input_percentile", type=float, default=90.0)
    parser.add_argument("--input", "-i", default="../data/web_search_trace_2x.jsonl")
    parser.add_argument("--completion_tokens_col", default="completion_tokens")
    parser.add_argument("--measured_no_cache_latency", type=float, default=5.429580)
    parser.add_argument("--fetch_mean", type=float, default=1.80425)
    parser.add_argument("--search_mean", type=float, default=4.22863)
    parser.add_argument("--q1", type=float, default=0.007)
    parser.add_argument("--q2", type=float, default=0.0007)
    parser.add_argument("--beta_divisor", type=float, default=40.0)
    parser.add_argument("--b_scale", type=float, default=-0.975)
    parser.add_argument("--A", type=float, default=0.0)
    return parser.parse_args()


def parse_timestamp(value):
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        raise ValueError("empty timestamp")
    try:
        return float(text)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            continue
    return datetime.fromisoformat(text).timestamp()


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_line_no"] = line_no
            records.append(record)
    return records


def session_key(record, args):
    return (record.get(args.sample_col), record.get(args.turn_col, 0))


def as_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_session_pairs(records, args):
    sessions = defaultdict(list)
    for record in records:
        sessions[session_key(record, args)].append(record)
    pairs = []
    for key, group in sessions.items():
        group.sort(
            key=lambda r: (
                as_int(r.get(args.iteration_col), 10**18),
                r.get("_line_no", 10**18),
            )
        )
        for cur, nxt in zip(group, group[1:]):
            cur_ts = parse_timestamp(cur[args.timestamp_col])
            next_ts = parse_timestamp(nxt[args.timestamp_col])
            delta = next_ts - cur_ts
            if delta < 0:
                raise ValueError(
                    f"Negative timestamp delta for session={key}, "
                    f"line={cur['_line_no']} -> {nxt['_line_no']}"
                )
            llm_latency = float(cur.get(args.latency_col, 0.0))
            tool_latency = max(0.0, delta - llm_latency)
            pairs.append({
                "session": key,
                "iteration": cur.get(args.iteration_col),
                "next_iteration": nxt.get(args.iteration_col),
                "delta_start_to_start": delta,
                "llm_latency": llm_latency,
                "estimated_tool_latency": tool_latency,
            })
    return pairs


def normalize_tool_name(tool_name):
    if tool_name is None:
        return None
    tool_name = str(tool_name).strip()
    if not tool_name:
        return None
    if "search_engine_query" in tool_name:
        return "search_engine_query"
    if "fetch_url_content" in tool_name:
        return "fetch_url_content"
    return tool_name


def parse_tool_names(output_text):
    if output_text is None:
        return []
    output_text = str(output_text)
    tool_names = []
    for match in re.finditer(r"Action:\s*([A-Za-z0-9_]+)", output_text):
        tool_name = normalize_tool_name(match.group(1))
        if tool_name in ("fetch_url_content", "search_engine_query"):
            tool_names.append(tool_name)
    if not tool_names:
        if "fetch_url_content" in output_text:
            tool_names.append("fetch_url_content")
        if "search_engine_query" in output_text:
            tool_names.append("search_engine_query")
    return tool_names


def estimate_tool_latency(output_text, args):
    latency = 0.0
    for tool_name in parse_tool_names(output_text):
        if tool_name == "fetch_url_content":
            latency += args.fetch_mean
        elif tool_name == "search_engine_query":
            latency += args.search_mean
    return latency


def percentile_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def empirical_cdf_at(sorted_values, value):
    return np.searchsorted(sorted_values, value, side="right") / len(sorted_values)


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
        rows.append({
            "percentile": p,
            "lifespan": val,
            "cdf": y_val,
            "B": b_value,
            "alpha": alpha,
            "beta": beta,
            "A": args.A,
            "config_line": (
                f"{args.q1} {args.q2} {p:g} "
                f"{b_value:.6e} {alpha:.6e} {beta:.6e} {args.A}"
            ),
        })
    return rows


def main():
    args = parse_args()
    records = load_records(args.input)
    if not records:
        raise ValueError(f"No records loaded from {args.input}")
    prompt_tokens = np.asarray(
        [float(r[args.prompt_tokens_col]) for r in records if args.prompt_tokens_col in r],
        dtype=float,
    )
    completion_tokens = np.asarray(
        [float(r[args.completion_tokens_col]) for r in records if args.completion_tokens_col in r],
        dtype=float,
    )
    latencies = np.asarray(
        [float(r[args.latency_col]) for r in records if args.latency_col in r],
        dtype=float,
    )
    pairs = build_session_pairs(records, args)
    if not pairs:
        raise ValueError("No adjacent LLM calls found; cannot estimate lifespan")
    start_to_start = np.asarray([x["delta_start_to_start"] for x in pairs], dtype=float)
    gpt_api_latencies = np.asarray([x["llm_latency"] for x in pairs], dtype=float)
    gpt_api_tool_latencies = np.asarray([x["estimated_tool_latency"] for x in pairs], dtype=float)

    records_by_session_iter = {
        (session_key(r, args), as_int(r.get(args.iteration_col), 10**18)): r
        for r in records
    }
    tool_latencies = []
    for pair in pairs:
        record = records_by_session_iter.get(
            (pair["session"], as_int(pair["iteration"], 10**18))
        )
        tool_latencies.append(estimate_tool_latency(record.get("output_text", ""), args))
    tool_latencies = np.asarray(tool_latencies, dtype=float)
    lifespan_samples = args.measured_no_cache_latency + tool_latencies
    lifespan_source = "vllm_no_cache_latency_plus_tool_mean_from_output_text"
    percentiles = percentile_list(args.lifespan_percentiles)
    rows = solve_params(lifespan_samples, percentiles, args)

    print("=== Dataset Profile ===")
    print(f"records: {len(records)}")
    print(f"adjacent_call_pairs: {len(pairs)}")
    print(f"input_tokens_p{args.input_percentile:g}: {np.percentile(prompt_tokens, args.input_percentile):.3f}")
    print(f"output_tokens_avg: {np.mean(completion_tokens):.3f}")
    print(f"gpt_api_llm_latency_avg_not_used: {np.mean(latencies):.6f}")
    print(f"gpt_api_llm_latency_p95_not_used: {np.percentile(latencies, 95):.6f}")
    print(f"gpt_api_llm_latency_p99_not_used: {np.percentile(latencies, 99):.6f}")
    print()
    print("=== Lifespan Estimate ===")
    print(f"source: {lifespan_source}")
    print(f"measured_vllm_no_cache_latency: {args.measured_no_cache_latency:.6f}")
    print(f"fetch_mean: {args.fetch_mean:.6f}")
    print(f"search_mean: {args.search_mean:.6f}")
    print(f"estimated_tool_latency_avg: {np.mean(tool_latencies):.6f}")
    print(f"estimated_tool_latency_p95: {np.percentile(tool_latencies, 95):.6f}")
    print(f"estimated_tool_latency_p99: {np.percentile(tool_latencies, 99):.6f}")
    print(f"lifespan_avg: {np.mean(lifespan_samples):.6f}")
    print(f"lifespan_p95: {np.percentile(lifespan_samples, 95):.6f}")
    print(f"lifespan_p99: {np.percentile(lifespan_samples, 99):.6f}")
    print()
    print("=== Trace Timing Diagnostics Not Used ===")
    print(f"gpt_api_start_to_start_avg: {np.mean(start_to_start):.6f}")
    print(f"gpt_api_estimated_tool_latency_avg: {np.mean(gpt_api_tool_latencies):.6f}")
    print(f"gpt_api_pair_llm_latency_avg: {np.mean(gpt_api_latencies):.6f}")
    print()
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