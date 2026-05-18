import csv
import json
import argparse
import statistics
from pathlib import Path
from datetime import datetime


CSV_FIELDS = [
    "line_no",
    "log_timestamp",
    "payload_timestamp",
    "elapsed_s",
    "prefill_prefills_count",
    "prefill_running_count",
    "prefill_active_count",
    "partial_prefill_prefills_count",
    "partial_prefill_running_count",
    "partial_prefill_total_count",
    "decode_running_count",
    "total_request_batch_size",
    "prefill_chunk_tokens",
    "decode_seqs",
    "total_scheduled_tokens_est",
    "max_prefill_chunk_size",
    "max_prefill_prompt_tokens",
    "swapped_count",
    "partial_prefill_running_limit_hit",
    "partial_prefill_running_limit_exceeded",
    "waiting_queue_len_before",
    "budget_remaining_before_prefill",
    "scheduled_prefill_count",
    "skipped_by_partial_limit",
    "break_by_alloc_later",
    "break_by_budget_or_zero_chunk",
    "break_by_max_batched_tokens",
    "waiting_head_prompt_tokens",
    "waiting_head_num_new_tokens_uncached",
    "budget_remaining_after_prefill",
    "waiting_queue_len_after",
]

PREFILL_DEBUG_FIELDS = [
    "waiting_queue_len_before",
    "budget_remaining_before_prefill",
    "scheduled_prefill_count",
    "skipped_by_partial_limit",
    "break_by_alloc_later",
    "break_by_budget_or_zero_chunk",
    "break_by_max_batched_tokens",
    "waiting_head_prompt_tokens",
    "waiting_head_num_new_tokens_uncached",
    "budget_remaining_after_prefill",
    "waiting_queue_len_after",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Parse vLLM schedule.log DETAILED_INFO lines and summarize batch "
            "size / partial-prefill behavior."
        )
    )
    parser.add_argument("--input", "-i", required=True, help="schedule.log file or directory")
    parser.add_argument("--output_csv", "-o", default="schedule_batch_summary.csv")
    parser.add_argument("--pattern", default="*schedule*.log")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--max_num_partial_prefills",
        type=int,
        default=4,
        help="Expected max-num-partial-prefills value.",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Optional PNG path. Requires matplotlib.",
    )
    return parser.parse_args()


def iter_input_files(input_path: Path, pattern: str, recursive: bool):
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    globber = input_path.rglob if recursive else input_path.glob
    return sorted(p for p in globber(pattern) if p.is_file())


def parse_log_line(line: str):
    marker = " - DETAILED_INFO - "
    log_ts = None
    if marker in line:
        prefix = line.split(marker, 1)[0].strip()
        log_ts = prefix
        payload = line.split(marker, 1)[1].strip()
    else:
        start = line.find("{")
        if start < 0:
            return None, None
        payload = line[start:].strip()
    try:
        return log_ts, json.loads(payload)
    except json.JSONDecodeError:
        return log_ts, None


def parse_ts(value):
    if not value:
        return None
    text = str(value).strip()
    if "," in text:
        text = text.replace(",", ".", 1)
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def seq_prompt_tokens(item):
    total = 0
    for seq in item.get("seqs") or []:
        total += int(seq.get("prompt_tokens") or 0)
    return total


def seq_count(item):
    return int(item.get("num_seqs") or len(item.get("seqs") or []))


def is_partial_prefill(item):
    chunk_idx = item.get("chunk_idx")
    chunk_size = item.get("chunk_size")
    prompt_tokens = seq_prompt_tokens(item)
    if chunk_idx is not None and int(chunk_idx) > 0:
        return True
    if chunk_size is not None and prompt_tokens and int(chunk_size) < prompt_tokens:
        return True
    return False


def summarize_record(source_file, line_no, log_timestamp, record, first_ts, max_num_partial_prefills):
    payload_timestamp = record.get("timestamp")
    ts = parse_ts(log_timestamp) or parse_ts(payload_timestamp)
    elapsed_s = (ts - first_ts).total_seconds() if ts and first_ts else ""

    prefill = record.get("prefill") or {}
    decode = record.get("decode") or {}

    prefill_prefills = prefill.get("Prefills") or []
    prefill_running = prefill.get("Running") or []
    prefill_swapped = prefill.get("Swapped") or []
    decode_running = decode.get("Running") or []
    decode_swapped = decode.get("Swapped") or []

    prefill_active = prefill_prefills + prefill_running
    partial_prefill_prefills_count = sum(1 for item in prefill_prefills if is_partial_prefill(item))
    partial_prefill_running_count = sum(1 for item in prefill_running if is_partial_prefill(item))
    partial_prefill_total_count = partial_prefill_prefills_count + partial_prefill_running_count

    prefill_chunk_tokens = sum(int(item.get("chunk_size") or 0) for item in prefill_active)
    decode_seqs = sum(seq_count(item) for item in decode_running)
    total_scheduled_tokens_est = prefill_chunk_tokens + decode_seqs
    max_prefill_chunk_size = max([int(item.get("chunk_size") or 0) for item in prefill_active] or [0])
    max_prefill_prompt_tokens = max([seq_prompt_tokens(item) for item in prefill_active] or [0])

    total_request_batch_size = len(prefill_active) + len(decode_running)
    swapped_count = len(prefill_swapped) + len(decode_swapped)

    row = {
        "source_file": source_file,
        "line_no": line_no,
        "log_timestamp": log_timestamp,
        "payload_timestamp": payload_timestamp,
        "elapsed_s": elapsed_s,
        "prefill_prefills_count": len(prefill_prefills),
        "prefill_running_count": len(prefill_running),
        "prefill_active_count": len(prefill_active),
        "partial_prefill_prefills_count": partial_prefill_prefills_count,
        "partial_prefill_running_count": partial_prefill_running_count,
        "partial_prefill_total_count": partial_prefill_total_count,
        "decode_running_count": len(decode_running),
        "total_request_batch_size": total_request_batch_size,
        "prefill_chunk_tokens": prefill_chunk_tokens,
        "decode_seqs": decode_seqs,
        "total_scheduled_tokens_est": total_scheduled_tokens_est,
        "max_prefill_chunk_size": max_prefill_chunk_size,
        "max_prefill_prompt_tokens": max_prefill_prompt_tokens,
        "swapped_count": swapped_count,
        "partial_prefill_running_limit_hit": int(partial_prefill_running_count == max_num_partial_prefills),
        "partial_prefill_running_limit_exceeded": int(partial_prefill_running_count > max_num_partial_prefills),
    }
    prefill_debug = record.get("prefill_debug") or {}
    for field in PREFILL_DEBUG_FIELDS:
        row[field] = prefill_debug.get(field, "")
    return row


def parse_file(path: Path, max_num_partial_prefills: int):
    parsed = []
    first_ts = None
    pending = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            log_timestamp, record = parse_log_line(line)
            if not record:
                continue
            ts = parse_ts(log_timestamp) or parse_ts(record.get("timestamp"))
            if ts and first_ts is None:
                first_ts = ts
            pending.append((line_no, log_timestamp, record))

    for line_no, log_timestamp, record in pending:
        parsed.append(
            summarize_record(
                source_file=path.name,
                line_no=line_no,
                log_timestamp=log_timestamp,
                record=record,
                first_ts=first_ts,
                max_num_partial_prefills=max_num_partial_prefills,
            )
        )
    return parsed


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pct(values, q):
    if not values:
        return 0
    values = sorted(values)
    idx = int(round((len(values) - 1) * q))
    return values[idx]


def write_plot(path: Path, rows, max_num_partial_prefills: int):
    import matplotlib.pyplot as plt
    x = [float(r["elapsed_s"]) if r["elapsed_s"] != "" else i for i, r in enumerate(rows)]
    series = [
        ("partial_prefill_running_count", "partial prefill running"),
        ("prefill_active_count", "prefill active"),
        ("decode_running_count", "decode running"),
        ("total_request_batch_size", "request batch size"),
    ]
    plt.figure(figsize=(16, 8))
    for field, label in series:
        y = [float(r[field]) for r in rows]
        plt.plot(x, y, linewidth=1.2, label=label)
    plt.axhline(
        y=max_num_partial_prefills,
        color="black",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label=f"max partial prefills ({max_num_partial_prefills})",
    )
    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("Count")
    plt.title("vLLM Scheduler Batch Size Over Time")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    

def main():
    args = parse_args()
    input_files = iter_input_files(Path(args.input), args.pattern, args.recursive)
    if not input_files:
        raise ValueError(f"No files matched: {args.input} / {args.pattern}")

    rows = []
    for path in input_files:
        file_rows = parse_file(path, args.max_num_partial_prefills)
        rows.extend(file_rows)
        print(f"[OK] parsed {path}: {len(file_rows)} rows")

    write_csv(Path(args.output_csv), rows)
    print(f"[OK] wrote {args.output_csv}")

    if args.plot:
        write_plot(Path(args.plot), rows, args.max_num_partial_prefills)
        print(f"[OK] wrote plot {args.plot}")


if __name__ == "__main__":
    main()
