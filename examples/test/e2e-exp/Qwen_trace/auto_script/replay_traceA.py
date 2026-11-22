#!/usr/bin/env python3
"""Replay a prepared Trace A manifest against a vLLM completions endpoint."""

from __future__ import annotations
import os
import time
import json
import asyncio
import argparse
from typing import Any
from pathlib import Path
from dataclasses import dataclass
from map_token import TraceTokenMapper


DEFAULT_API_URL = os.environ.get("TRACEA_API_URL", "http://127.0.0.1:8052/v1/completions")
DEFAULT_MODEL = os.environ.get("TRACEA_MODEL_PATH")
DEFAULT_TARGET_QPS = (
    float(os.environ["TRACEA_TARGET_QPS"])
    if os.environ.get("TRACEA_TARGET_QPS")
    else None
)
RETRYABLE_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


@dataclass(slots=True)
class ReplayRecord:
    raw: dict[str, Any]
    chat_id: int
    parent_chat_id: int
    root_chat_id: int
    timestamp: float
    input_length: int
    output_length: int
    hash_ids: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-i", type=Path, required=True)
    parser.add_argument("--output_file", "-o", type=Path, required=True)
    parser.add_argument("--summary-file", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--baseline")
    parser.add_argument("--model-size")
    parser.add_argument("--api_url", default=DEFAULT_API_URL)
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, required=DEFAULT_MODEL is None
    )
    parser.add_argument("--mapping-config", type=Path)
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Replay speed multiplier; 10 means ten times faster",
    )
    timing.add_argument(
        "--target-qps",
        type=float,
        default=DEFAULT_TARGET_QPS,
        help="Scale the complete selected trace to this average arrival QPS",
    )
    parser.add_argument(
        "--ttl-mode", choices=("trace", "none"), default="trace"
    )
    parser.add_argument("--request-timeout", type=float, default=150 * 60 * 60)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum HTTP attempts per request, including the first attempt",
    )
    parser.add_argument(
        "--retry-fast-failure-seconds",
        type=float,
        default=2.0,
        help="Fast-failure threshold for retryable errors other than server disconnects",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=0.5,
        help="Base delay for retries other than ServerDisconnectedError",
    )
    parser.add_argument("--max-in-flight", type=int, default=0)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument(
        "--root-chat-id",
        type=int,
        help="Replay only one complete session with this root_chat_id",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--append", action="store_true")
    parser.add_argument(
        "--fail-on-request-error",
        action="store_true",
        help="Exit nonzero after replay finishes if any request finally failed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and materialize sample payloads without HTTP requests",
    )
    return parser.parse_args()


def load_manifest(
    path: Path,
    max_requests: int | None,
    root_chat_id: int | None = None,
) -> list[ReplayRecord]:
    records: list[ReplayRecord] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if (
                root_chat_id is not None
                and int(item["root_chat_id"]) != root_chat_id
            ):
                continue
            record = ReplayRecord(
                raw=item,
                chat_id=int(item["chat_id"]),
                parent_chat_id=int(item["parent_chat_id"]),
                root_chat_id=int(item["root_chat_id"]),
                timestamp=float(item["timestamp"]),
                input_length=int(item["input_length"]),
                output_length=int(item["output_length"]),
                hash_ids=[int(value) for value in item["hash_ids"]],
            )
            expected_hashes = (
                record.input_length + 15
            ) // 16
            if len(record.hash_ids) != expected_hashes:
                raise ValueError(
                    f"Manifest line {line_number} has an inconsistent hash count"
                )
            records.append(record)
            if max_requests is not None and len(records) >= max_requests:
                break
    if not records:
        raise ValueError("Replay manifest is empty")
    records.sort(key=lambda record: (record.timestamp, record.chat_id))
    return records


def materialize_output(
    record: ReplayRecord,
    records_by_id: dict[int, ReplayRecord],
    mapper: TraceTokenMapper,
) -> tuple[list[int], str, int]:
    recipe = record.raw.get("output_recipe", {})
    output_ids: list[int] = []
    source_kind = str(recipe.get("kind", "deterministic_leaf"))
    if source_kind == "next_input_slice":
        source = records_by_id.get(int(recipe["source_chat_id"]))
        if source is not None:
            offset = int(recipe.get("offset", record.input_length))
            output_ids = canonical_prompt_slice(
                source,
                records_by_id,
                mapper,
                offset,
                offset + record.output_length,
            )
        else:
            source_kind = "missing_child_fallback"

    recovered = len(output_ids)
    if recovered < record.output_length:
        fallback = mapper.synthetic_output_ids(
            record.chat_id, record.output_length - recovered
        )
        output_ids.extend(fallback)
    return output_ids, source_kind, recovered


def canonical_prompt_slice(
    record: ReplayRecord,
    records_by_id: dict[int, ReplayRecord],
    mapper: TraceTokenMapper,
    start: int,
    end: int,
) -> list[int]:
    """Materialize a prompt slice and repair an incomplete final 16-token block.

    A partial Trace A hash can change when the next turn appends tokens even
    though the already-present token prefix did not change. For non-leaf
    requests, source those partial-tail IDs from the child prompt so that the
    parent prompt plus reconstructed output is continuous with the child.
    """
    start = max(0, min(int(start), record.input_length))
    end = max(start, min(int(end), record.input_length))
    result = mapper.prompt_slice(
        record.hash_ids, record.input_length, start, end
    )
    tail_length = record.input_length % mapper.trace_block_size
    next_chat_id = int(record.raw.get("next_chat_id", -1))
    child = records_by_id.get(next_chat_id)
    if tail_length == 0 or child is None:
        return result

    tail_start = record.input_length - tail_length
    overlap_start = max(start, tail_start)
    overlap_end = min(end, record.input_length, child.input_length)
    if overlap_start >= overlap_end:
        return result
    replacement = mapper.prompt_slice(
        child.hash_ids,
        child.input_length,
        overlap_start,
        overlap_end,
    )
    result_offset = overlap_start - start
    result[result_offset : result_offset + len(replacement)] = replacement
    return result


def materialize_prompt(
    record: ReplayRecord,
    records_by_id: dict[int, ReplayRecord],
    mapper: TraceTokenMapper,
) -> tuple[list[int], int]:
    base_tail = mapper.prompt_slice(
        record.hash_ids,
        record.input_length,
        record.input_length - record.input_length % mapper.trace_block_size,
        record.input_length,
    )
    prompt_ids = canonical_prompt_slice(
        record, records_by_id, mapper, 0, record.input_length
    )
    repaired_tail = prompt_ids[-len(base_tail) :] if base_tail else []
    repaired_count = sum(
        left != right for left, right in zip(base_tail, repaired_tail)
    )
    return prompt_ids, repaired_count


def calculate_time_scale(
    records: list[ReplayRecord], requested_scale: float, target_qps: float | None
) -> float:
    duration = records[-1].timestamp - records[0].timestamp
    if target_qps is None:
        scale = requested_scale
    else:
        if target_qps <= 0:
            raise ValueError("--target-qps must be positive")
        original_qps = len(records) / duration if duration > 0 else target_qps
        scale = target_qps / original_qps
    if scale <= 0:
        raise ValueError("--time-scale must be positive")
    return scale


def validate_dry_run(
    records: list[ReplayRecord], mapper: TraceTokenMapper
) -> None:
    records_by_id = {record.chat_id: record for record in records}
    sample_indices = sorted({0, len(records) // 2, len(records) - 1})
    for index in sample_indices:
        record = records[index]
        prompt_ids, repaired_count = materialize_prompt(
            record, records_by_id, mapper
        )
        output_ids, source_kind, recovered = materialize_output(
            record, records_by_id, mapper
        )
        if len(prompt_ids) != record.input_length:
            raise AssertionError("Prompt length mismatch")
        if len(output_ids) != record.output_length:
            raise AssertionError("Output length mismatch")
        print(
            f"dry-run chat_id={record.chat_id} prompt={len(prompt_ids)} "
            f"output={len(output_ids)} source={source_kind} "
            f"recovered_from_child={recovered} "
            f"tail_tokens_repaired={repaired_count}"
        )

    first_hash_owner: dict[int, list[int]] = {}
    checked = 0
    for record in records:
        for hash_id in record.hash_ids:
            mapped = mapper.tokens_for_hash(hash_id)
            previous = first_hash_owner.setdefault(hash_id, mapped)
            if previous != mapped:
                raise AssertionError("Hash mapping is not deterministic")
            checked += 1
            if checked >= 10000:
                break
        if checked >= 10000:
            break
    print(f"validated {checked:,} hash occurrences for deterministic mapping")

    non_leaf_count = 0
    leaf_count = 0
    repaired_tail_tokens = 0
    for record in records:
        output_ids, source_kind, recovered = materialize_output(
            record, records_by_id, mapper
        )
        if len(output_ids) != record.output_length:
            raise AssertionError(
                f"Output length mismatch for chat_id={record.chat_id}"
            )
        _, repaired = materialize_prompt(record, records_by_id, mapper)
        repaired_tail_tokens += repaired
        if source_kind != "next_input_slice":
            leaf_count += 1
            continue

        non_leaf_count += 1
        child = records_by_id[int(record.raw["next_chat_id"])]
        if recovered != record.output_length:
            raise AssertionError(
                f"Incomplete child output recovery for chat_id={record.chat_id}"
            )
        tail_length = record.input_length % mapper.trace_block_size
        compare_start = record.input_length - tail_length
        parent_boundary = canonical_prompt_slice(
            record,
            records_by_id,
            mapper,
            compare_start,
            record.input_length,
        ) + output_ids
        child_boundary = canonical_prompt_slice(
            child,
            records_by_id,
            mapper,
            compare_start,
            record.input_length + record.output_length,
        )
        if parent_boundary != child_boundary:
            raise AssertionError(
                f"Parent/output continuity mismatch for chat_id={record.chat_id}"
            )
    print(
        f"validated all outputs: non_leaf={non_leaf_count:,} "
        f"leaf={leaf_count:,} repaired_tail_tokens={repaired_tail_tokens:,}"
    )


async def replay(
    args: argparse.Namespace, records: list[ReplayRecord]
) -> dict[str, Any]:
    try:
        import aiohttp
    except ImportError as exc:
        raise RuntimeError("Install aiohttp before live replay") from exc

    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    if args.retry_fast_failure_seconds < 0:
        raise ValueError("--retry-fast-failure-seconds must be non-negative")
    if args.retry_backoff_seconds < 0:
        raise ValueError("--retry-backoff-seconds must be non-negative")

    config_path = args.mapping_config or (
        args.input_file.parent / "mapping_config.json"
    )
    mapper = TraceTokenMapper.from_path(config_path)
    records_by_id = {record.chat_id: record for record in records}
    if len(records_by_id) != len(records):
        raise ValueError("Replay manifest contains duplicate chat_id values")

    time_scale = calculate_time_scale(
        records, args.time_scale, args.target_qps
    )
    first_timestamp = records[0].timestamp
    completion_events = {
        record.chat_id: asyncio.Event() for record in records
    }
    semaphore = (
        asyncio.Semaphore(args.max_in_flight)
        if args.max_in_flight > 0
        else None
    )
    file_lock = asyncio.Lock()
    progress_lock = asyncio.Lock()
    completed = 0
    succeeded = 0
    failed = 0
    total_attempts = 0
    failed_chat_ids: list[int] = []
    started_at = time.monotonic()
    run_id = args.run_id or args.output_file.stem

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)

    async with aiohttp.ClientSession(timeout=timeout) as http_session:
        with args.output_file.open(mode, encoding="utf-8", newline="") as output:

            async def run_one(record: ReplayRecord) -> None:
                nonlocal completed, succeeded, failed, total_attempts
                target = (record.timestamp - first_timestamp) / time_scale
                wait = target - (time.monotonic() - started_at)
                if wait > 0:
                    await asyncio.sleep(wait)

                parent_event = completion_events.get(record.parent_chat_id)
                dependency_wait_started = time.monotonic()
                if parent_event is not None:
                    await parent_event.wait()
                dependency_wait = time.monotonic() - dependency_wait_started

                prompt_ids, repaired_count = materialize_prompt(
                    record, records_by_id, mapper
                )
                output_ids, output_source, recovered = materialize_output(
                    record, records_by_id, mapper
                )
                payload = {
                    "model": args.model,
                    "prompt": prompt_ids,
                    "max_tokens": record.output_length,
                    "add_special_tokens": False,
                    "std_output_tokens": output_ids,
                }

                request_started = time.monotonic()
                attempt_history: list[dict[str, Any]] = []
                response: dict[str, Any] = {}

                async def perform_attempt(
                    attempt: int,
                ) -> tuple[dict[str, Any], bool, bool]:
                    attempt_started = time.monotonic()
                    send_timestamp = time.time()
                    caught_exception: Exception | None = None
                    server_disconnected = False
                    try:
                        attempt_result = await post_json(
                            http_session,
                            args.api_url,
                            payload,
                            record.chat_id,
                        )
                    except aiohttp.ServerDisconnectedError as exc:
                        caught_exception = exc
                        server_disconnected = True
                        attempt_result = {
                            "error": "request_failed",
                            "exception": repr(exc),
                            "exception_type": type(exc).__name__,
                            "http_status": None,
                        }
                    except Exception as exc:
                        caught_exception = exc
                        attempt_result = {
                            "error": "request_failed",
                            "exception": repr(exc),
                            "exception_type": type(exc).__name__,
                            "http_status": None,
                        }

                    attempt_finished = time.monotonic()
                    finish_timestamp = time.time()
                    duration = attempt_finished - attempt_started
                    http_status = attempt_result.get("http_status")
                    retryable_exception = isinstance(
                        caught_exception,
                        (aiohttp.ClientConnectionError, asyncio.TimeoutError),
                    )
                    retryable_status = http_status in RETRYABLE_HTTP_STATUSES
                    failed = caught_exception is not None or (
                        isinstance(http_status, int) and http_status >= 400
                    )
                    retryable = server_disconnected or (
                        failed
                        and duration <= args.retry_fast_failure_seconds
                        and (retryable_exception or retryable_status)
                    )
                    will_retry = retryable and attempt < args.max_attempts

                    if server_disconnected:
                        action = "retrying immediately" if will_retry else "no attempts left"
                        print(
                            "WARNING "
                            f"request_id=tracea-{record.chat_id} "
                            f"attempt={attempt}/{args.max_attempts} "
                            "ServerDisconnectedError "
                            f"after {duration:.6f}s; {action}",
                            flush=True,
                        )

                    attempt_history.append(
                        {
                            "attempt": attempt,
                            "request_id": f"tracea-{record.chat_id}",
                            "send_timestamp": send_timestamp,
                            "finish_timestamp": finish_timestamp,
                            "failure_timestamp": finish_timestamp if failed else None,
                            "send_seconds": attempt_started - started_at,
                            "finish_seconds": attempt_finished - started_at,
                            "duration_seconds": duration,
                            "http_status": http_status,
                            "exception_type": (
                                type(caught_exception).__name__
                                if caught_exception is not None
                                else None
                            ),
                            "exception": attempt_result.get("exception"),
                            "retryable": retryable,
                        }
                    )
                    return attempt_result, retryable, server_disconnected

                for attempt in range(1, args.max_attempts + 1):
                    if semaphore is None:
                        response, should_retry, disconnected = (
                            await perform_attempt(attempt)
                        )
                    else:
                        async with semaphore:
                            response, should_retry, disconnected = (
                                await perform_attempt(attempt)
                            )
                    if not should_retry or attempt == args.max_attempts:
                        break
                    if not disconnected:
                        await asyncio.sleep(args.retry_backoff_seconds * attempt)

                request_finished = time.monotonic()
                result = response
                http_status = result.get("http_status")
                request_succeeded = (
                    isinstance(http_status, int)
                    and 200 <= http_status < 300
                    and "error" not in result
                    and bool(result.get("choices"))
                )
                result.update(
                    {
                        "run_id": run_id,
                        "baseline": args.baseline,
                        "model_size": args.model_size,
                        "request_succeeded": request_succeeded,
                        "chat_id": record.chat_id,
                        "parent_chat_id": record.parent_chat_id,
                        "root_chat_id": record.root_chat_id,
                        "turn": record.raw.get("turn"),
                        "type": record.raw.get("type"),
                        "trace_timestamp": record.timestamp,
                        "scheduled_seconds": target,
                        "actual_start_seconds": request_started - started_at,
                        "request_seconds": request_finished - request_started,
                        "request_attempts": len(attempt_history),
                        "retry_count": max(0, len(attempt_history) - 1),
                        "attempt_history": attempt_history,
                        "dependency_wait_seconds": dependency_wait,
                        "input_length": record.input_length,
                        "output_length": record.output_length,
                        "output_source": output_source,
                        "output_tokens_recovered_from_child": recovered,
                        "prompt_tail_tokens_repaired": repaired_count,
                        "time_scale": time_scale,
                    }
                )
                async with file_lock:
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
                completion_events[record.chat_id].set()

                async with progress_lock:
                    completed += 1
                    total_attempts += len(attempt_history)
                    if request_succeeded:
                        succeeded += 1
                    else:
                        failed += 1
                        failed_chat_ids.append(record.chat_id)
                    if completed % args.progress_every == 0 or completed == len(records):
                        elapsed = time.monotonic() - started_at
                        print(
                            f"completed {completed:,}/{len(records):,} "
                            f"elapsed={elapsed:.1f}s"
                        )

            tasks = [asyncio.create_task(run_one(record)) for record in records]
            await asyncio.gather(*tasks)

    elapsed = time.monotonic() - started_at
    return {
        "run_id": run_id,
        "baseline": args.baseline,
        "model_size": args.model_size,
        "input_file": str(args.input_file),
        "output_file": str(args.output_file),
        "total_requests": len(records),
        "completed_requests": completed,
        "successful_requests": succeeded,
        "failed_requests": failed,
        "failed_chat_ids": sorted(failed_chat_ids),
        "total_http_attempts": total_attempts,
        "total_retries": total_attempts - completed,
        "time_scale": time_scale,
        "elapsed_seconds": elapsed,
        "complete": completed == len(records) and failed == 0,
    }


async def post_json(session: Any, api_url: str, payload: dict[str, Any], chat_id: int) -> dict[str, Any]:
    stable_request_id = f"tracea-{chat_id}"
    async with session.post(
        api_url,
        json=payload,
        headers={"Content-Type": "application/json", "X-Request-Id": stable_request_id},
    ) as response:
        text = await response.text()
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            result = {"response_text": text}
        if not isinstance(result, dict):
            result = {"response_json": result}
        result["http_status"] = response.status
        if response.status >= 400:
            result.setdefault("error", f"http_{response.status}")
        return result


def main() -> None:
    args = parse_args()
    records = load_manifest(
        args.input_file, args.max_requests, args.root_chat_id
    )
    config_path = args.mapping_config or (
        args.input_file.parent / "mapping_config.json"
    )
    mapper = TraceTokenMapper.from_path(config_path)
    scale = calculate_time_scale(records, args.time_scale, args.target_qps)
    duration = (records[-1].timestamp - records[0].timestamp) / scale
    print(
        f"loaded {len(records):,} requests; time_scale={scale:.6g}; "
        f"scheduled_duration={duration:.1f}s"
    )
    if args.dry_run:
        validate_dry_run(records, mapper)
        return
    summary = asyncio.run(replay(args, records))
    summary_path = args.summary_file
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as output:
            json.dump(summary, output, ensure_ascii=False, indent=2)
            output.write("\n")
        print(f"wrote replay summary: {summary_path}")
    print(
        "replay summary: "
        f"success={summary['successful_requests']:,} "
        f"failed={summary['failed_requests']:,} "
        f"retries={summary['total_retries']:,}"
    )
    if args.fail_on_request_error and summary["failed_requests"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
