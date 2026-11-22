#!/usr/bin/env python3
"""Compute MWF inter-turn CDF parameters for a Trace A replay manifest."""

from __future__ import annotations
import json
import math
import argparse
import numpy as np
from typing import Any
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict


DEFAULT_INPUT = Path(
    r"C:\Users\15085\Desktop\Qwen\traceA_vllm_replay"
    r"\traceA_prefix_3393_replay_data\tracea_prefix_3393_replay.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Trace A inter-turn deltas and the MWF parameters used by "
            "CDF.py. By default, requests are grouped by root_chat_id and "
            "adjacent timestamps within each session are differenced."
        )
    )
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--percentile", type=float, default=75.0)
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Replay speed multiplier; must match replay_tracea_vllm.py",
    )
    timing.add_argument(
        "--target-qps",
        type=float,
        default=0.3,
        help="Derive time scale using the same formula as replay_tracea_vllm.py",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="Output path without extension; defaults beside the input manifest",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            missing = {
                "chat_id",
                "root_chat_id",
                "timestamp",
            }.difference(item)
            if missing:
                raise ValueError(
                    f"Line {line_number} is missing fields: {sorted(missing)}"
                )
            records.append(
                {
                    "chat_id": int(item["chat_id"]),
                    "root_chat_id": int(item["root_chat_id"]),
                    "timestamp": float(item["timestamp"]),
                    "turn": int(item.get("turn", 0)),
                }
            )
    if not records:
        raise ValueError(f"No records found in {path}")
    records.sort(key=lambda item: (item["timestamp"], item["chat_id"]))
    return records


def calculate_time_scale(
    records: list[dict[str, Any]], requested_scale: float, target_qps: float | None
) -> tuple[float, float, float]:
    duration = records[-1]["timestamp"] - records[0]["timestamp"]
    original_qps = len(records) / duration if duration > 0 else math.inf
    if target_qps is None:
        scale = requested_scale
    else:
        if target_qps <= 0:
            raise ValueError("--target-qps must be positive")
        scale = target_qps / original_qps if duration > 0 else 1.0
    if scale <= 0:
        raise ValueError("--time-scale must be positive")
    return scale, duration, original_qps


def compute_deltas(
    records: list[dict[str, Any]], time_scale: float
) -> tuple[np.ndarray, int, int]:
    sessions: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        sessions[record["root_chat_id"]].append(record)

    deltas: list[float] = []
    multi_request_sessions = 0
    for group in sessions.values():
        if len(group) < 2:
            continue
        multi_request_sessions += 1
        group.sort(
            key=lambda item: (
                item["turn"],
                item["timestamp"],
                item["chat_id"],
            )
        )
        timestamps = np.asarray(
            [item["timestamp"] for item in group], dtype=np.float64
        )
        group_deltas = np.diff(timestamps) / time_scale
        if np.any(group_deltas < 0):
            raise ValueError("Negative inter-turn delta found after sorting")
        deltas.extend(group_deltas.tolist())

    if not deltas:
        raise ValueError("No multi-request sessions were found")
    return np.asarray(deltas, dtype=np.float64), len(sessions), multi_request_sessions


def compute_mwf_parameters(
    deltas: np.ndarray, percentile: float
) -> dict[str, float]:
    if not 0 < percentile < 100:
        raise ValueError("--percentile must be between 0 and 100")
    x = np.sort(deltas)
    value = float(np.percentile(x, percentile))
    cdf = float(np.searchsorted(x, value, side="right") / len(x))
    if not 0 < cdf < 1:
        raise ValueError(
            "The empirical CDF at the selected percentile must be between 0 and 1"
        )
    a = 0.0
    b = -0.975 * value
    alpha = -value / math.log(cdf)
    beta = -value / (40 * math.log(cdf))
    return {
        "percentile": percentile,
        "percentile_seconds": value,
        "empirical_cdf": cdf,
        "MINEXP_A": a,
        "MINEXP_B": b,
        "MINEXP_ALPHA": alpha,
        "MINEXP_BETA": beta,
    }


def write_plot(
    deltas: np.ndarray, parameters: dict[str, float], output_path: Path
) -> None:
    x = np.sort(deltas)
    y = np.arange(1, len(x) + 1) / len(x)
    value = parameters["percentile_seconds"]
    cdf = parameters["empirical_cdf"]

    plt.figure(figsize=(12, 4))
    plt.plot(x, y, linewidth=1.4)
    plt.axvline(value, color="tab:red", linestyle="--", alpha=0.8)
    plt.scatter([value], [cdf], color="tab:red", s=30, zorder=3)
    plt.annotate(
        f"P{parameters['percentile']:g}={value:.3f}s",
        (value, cdf),
        xytext=(8, -18),
        textcoords="offset points",
    )
    plt.title("Qwen Trace A Inter-Turn Time CDF")
    plt.xlabel("Inter-turn time (seconds)")
    plt.ylabel("CDF")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    time_scale, trace_duration, original_qps = calculate_time_scale(
        records, args.time_scale, args.target_qps
    )
    deltas, session_count, multi_request_sessions = compute_deltas(
        records, time_scale
    )
    parameters = compute_mwf_parameters(deltas, args.percentile)

    output_prefix = args.output_prefix or (
        args.input.parent / f"{args.input.stem}_mwf_cdf"
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    result_path = Path(f"{output_prefix}.json")
    plot_path = Path(f"{output_prefix}.png")

    result = {
        "input": str(args.input.resolve()),
        "request_count": len(records),
        "session_count": session_count,
        "multi_request_session_count": multi_request_sessions,
        "inter_turn_delta_count": int(len(deltas)),
        "trace_duration_seconds": trace_duration,
        "original_average_qps": original_qps,
        "target_qps": args.target_qps,
        "time_scale": time_scale,
        "delta_definition": (
            "Adjacent timestamp differences after grouping by root_chat_id and "
            "sorting each session by turn and then timestamp, matching the "
            "grouping/sorting/diff logic of CDF.py; deltas are divided by "
            "replay time_scale."
        ),
        "delta_statistics_seconds": {
            "min": float(np.min(deltas)),
            "mean": float(np.mean(deltas)),
            "median": float(np.median(deltas)),
            "max": float(np.max(deltas)),
        },
        "mwf_parameters": parameters,
        "mwf_config_line": (
            f"{parameters['percentile']:g} "
            f"{parameters['MINEXP_B']:.6e} "
            f"{parameters['MINEXP_ALPHA']:.6e} "
            f"{parameters['MINEXP_BETA']:.6e} "
            f"{parameters['MINEXP_A']}"
        ),
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not args.no_plot:
        write_plot(deltas, parameters, plot_path)

    print(f"requests: {len(records):,}")
    print(f"sessions: {session_count:,}")
    print(f"multi-request sessions: {multi_request_sessions:,}")
    print(f"inter-turn deltas: {len(deltas):,}")
    print(f"time_scale: {time_scale:.9g}")
    print(
        f"P{parameters['percentile']:g}: "
        f"{parameters['percentile_seconds']:.9g}s, "
        f"CDF={parameters['empirical_cdf']:.9g}"
    )
    print(result["mwf_config_line"])
    print(f"result: {result_path}")
    if not args.no_plot:
        print(f"plot: {plot_path}")


if __name__ == "__main__":
    main()
