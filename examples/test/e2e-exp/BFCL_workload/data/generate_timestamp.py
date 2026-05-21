import json
import csv
import random
import argparse
from pathlib import Path
from collections import defaultdict


class GammaRequestIntervalGenerator:
    def __init__(self, qps: float, seed: int = 42, shape: float = 2.0):
        if qps <= 0:
            raise ValueError("qps must be > 0")
        if shape <= 0:
            raise ValueError("shape must be > 0")
        self.qps = float(qps)
        self.shape = float(shape)
        self.scale = 1.0 / (self.qps * self.shape)
        self.std = 1.0 / self.qps
        self.max_interval = self.std * 3.0
        self.rng = random.Random()
        self.rng.seed(seed)

    def get_next_inter_request_time(self) -> float:
        next_interval = self.rng.gammavariate(self.shape, self.scale)
        next_interval = min(next_interval, self.max_interval)
        return float(next_interval)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, default="2x_bfcl_web_search_prompts_qps2.0.csv",)
    parser.add_argument("--qps", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shape", type=float, default=16.0)
    parser.add_argument("--first_timestamp", type=float, default=0.0)
    parser.add_argument("--timestamp_mode", type=str, choices=["session_start_all_rows", "first_row_only"], default="first_row_only")
    return parser.parse_args()


def load_llm_records(input_path: Path):
    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Bad JSON at line {line_no}: {e}")
                continue
            record["_line_no"] = line_no
            records.append(record)
    return records


def get_session_key(record: dict):
    sample_id = record.get("sample_id")
    turn_id = record.get("turn_id", 0)
    if sample_id is None:
        raise ValueError(f"Found llm_call without sample_id at jsonl line {record.get('_line_no')}")
    return f"{sample_id}:{turn_id}"


def group_records_by_session(records):
    sessions = defaultdict(list)
    for record in records:
        key = get_session_key(record)
        sessions[key].append(record)
    for key in sessions:
        def sort_key(r):
            iteration = r.get("iteration")
            try:
                iteration = int(iteration)
            except (TypeError, ValueError):
                iteration = 10**18
            return (iteration, r.get("_line_no", 10**18))
        sessions[key].sort(key=sort_key)
    ordered_keys = sorted(
        sessions.keys(),
        key=lambda k: sessions[k][0].get("_line_no", 10**18),
    )
    return ordered_keys, sessions


def generate_session_start_times(ordered_session_keys, qps, seed, shape, first_timestamp):
    generator = GammaRequestIntervalGenerator(qps=qps, seed=seed, shape=shape)
    session_start = {}
    cur_t = float(first_timestamp)
    for i, key in enumerate(ordered_session_keys):
        if i == 0:
            session_start[key] = cur_t
        else:
            cur_t += generator.get_next_inter_request_time()
            session_start[key] = cur_t
    return session_start


def build_csv_rows(ordered_keys, sessions, session_start, timestamp_mode):
    rows = []
    for _, key in enumerate(ordered_keys):
        records = sessions[key]
        start_ts = session_start[key]
        for local_iteration, record in enumerate(records):
            if timestamp_mode == "session_start_all_rows":
                timestamp = start_ts
            elif timestamp_mode == "first_row_only":
                timestamp = start_ts if local_iteration == 0 else ""
            else:
                raise ValueError(f"Unknown timestamp_mode: {timestamp_mode}")
            row = {
                "Timestamp": timestamp,
                "session_key": key,
                "sample_id": record.get("sample_id"),
                "turn_id": record.get("turn_id", 0),
                "iteration": record.get("iteration"),
                "prompt_text": record.get("prompt_text", ""),
                "output_text": record.get("output_text", ""),
                "local_iteration": local_iteration
            }
            rows.append(row)
    rows.sort(
        key=lambda r: (
            float(r["Timestamp"]) if r["Timestamp"] != "" else session_start[r["session_key"]],
            int(r["local_iteration"]),
        )
    )
    return rows


def write_csv(rows, output_path: Path):
    fieldnames = [
        "Timestamp",
        "sample_id",
        "turn_id",
        "iteration",
        "prompt_text",
        "output_text",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "Timestamp": row["Timestamp"],
                "sample_id": row["sample_id"],
                "turn_id": row["turn_id"],
                "iteration": row["iteration"],
                "prompt_text": row["prompt_text"],
                "output_text": row["output_text"],
            })


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    records = load_llm_records(input_path)
    print(f"[INFO] Loaded llm_call records: {len(records)}")
    ordered_keys, sessions = group_records_by_session(records)
    print(f"[INFO] Num sessions: {len(ordered_keys)}")
    session_start = generate_session_start_times(
        ordered_session_keys=ordered_keys,
        qps=args.qps,
        seed=args.seed,
        shape=args.shape,
        first_timestamp=args.first_timestamp,
    )
    rows = build_csv_rows(
        ordered_keys=ordered_keys,
        sessions=sessions,
        session_start=session_start,
        timestamp_mode=args.timestamp_mode,
    )
    write_csv(rows, output_path)
    print(f"[INFO] Num output rows: {len(rows)}")
    print("\n[INFO] First 5 sessions:")
    for key in ordered_keys[:5]:
        print(
            {
                "session_key": key,
                "session_start": round(session_start[key], 6),
                "num_llm_calls": len(sessions[key]),
            }
        )
    print("\n[INFO] First 5 rows:")
    for row in rows[:5]:
        print(
            {
                "Timestamp": row["Timestamp"],
                "sample_id": row["sample_id"],
                "turn_id": row["turn_id"],
                "iteration": row["iteration"],
            }
        )


if __name__ == "__main__":
    main()