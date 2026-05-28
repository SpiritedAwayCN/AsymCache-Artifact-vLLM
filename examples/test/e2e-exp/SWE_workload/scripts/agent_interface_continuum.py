import os
import tqdm
import json
import time
import random
import aiohttp
import asyncio
import argparse
import pandas as pd
from transformers import AutoTokenizer


model = "/jizhicfs/marvinhjia/sca/models/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b"
vllm_host_ip = os.popen("hostname -I | awk '{print $1}'").read().strip()
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=150 * 60 * 60)
API_URL = "http://127.0.0.1:8166/v1/completions"
tokenizer = AutoTokenizer.from_pretrained(model)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_file", "-i", type=str, default="../data/swe_sampled_100_prompts_qps2.0.csv")
    parser.add_argument("--output_file","-o", type=str, default="../results/100_CONTINUUM_qps2.0/CONTINUUM_replay_output_qps2.0.jsonl")
    # 输入文件字段名
    parser.add_argument("--prompt_col", type=str, default="prompt_text")
    parser.add_argument("--output_col", type=str, default="output_text")
    parser.add_argument("--timestamp_col", type=str, default="Timestamp")
    parser.add_argument("--iteration_col", type=str, default="iteration")
    parser.add_argument("--instance_col", type=str, default="instance_id")
    parser.add_argument("--tool_sleep_col", type=str, default="tool_sleep_seconds")
    return parser.parse_args()


args = parse_args()
random.seed(args.seed)


def load_requests(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    required_cols = [
        args.timestamp_col,
        args.instance_col,
        args.iteration_col,
        args.prompt_col,
        args.output_col,
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    if args.tool_sleep_col not in df.columns:
        df[args.tool_sleep_col] = 0.0
    return df


def encode_text(text: str):
    return tokenizer.encode(str(text), add_special_tokens=False)


def clamp_tool_sleep(value) -> float:
    try:
        sleep_seconds = float(value)
    except (TypeError, ValueError):
        sleep_seconds = 0.0
    sleep_seconds = max(0.0, sleep_seconds)
    return sleep_seconds


def get_session_key(row):
    return str(row[args.instance_col])


def build_sessions(df: pd.DataFrame):
    sessions = []
    df = df.copy()
    df["_session_key"] = df.apply(get_session_key, axis=1)
    df["_iteration_int"] = df[args.iteration_col].astype(int)
    for session_key, group in df.groupby("_session_key", sort=False):
        group = group.sort_values("_iteration_int").reset_index(drop=True)
        first_ts = group.iloc[0][args.timestamp_col]
        if pd.isna(first_ts) or first_ts == "":
            raise ValueError(f"Session {session_key} has invalid first Timestamp")
        prev_prompt_text = None
        prev_prompt_token_ids = None
        rows = group.to_dict("records")
        
        for row in rows:
            prompt_text = str(row[args.prompt_col])
            if prev_prompt_text is not None and prompt_text.startswith(prev_prompt_text):
                suffix_text = prompt_text[len(prev_prompt_text):]
                prompt_token_ids = prev_prompt_token_ids + encode_text(suffix_text)
            else:
                prompt_token_ids = encode_text(prompt_text)
            if prev_prompt_token_ids is not None and prompt_text.startswith(prev_prompt_text):
                if prompt_token_ids[: len(prev_prompt_token_ids)] != prev_prompt_token_ids:
                    raise ValueError(
                        f"Incremental token prefix failed for session={session_key}, "
                        f"iteration={row[args.iteration_col]}"
                    )
            row["_prompt_token_ids"] = prompt_token_ids
            row["_tool_sleep_seconds"] = clamp_tool_sleep(row.get(args.tool_sleep_col, 0.0))
            prev_prompt_text = prompt_text
            prev_prompt_token_ids = prompt_token_ids
        
        sessions.append(
            {
                "session_key": session_key,
                "session_start": float(first_ts),
                "rows": rows,
            }
        )
    sessions.sort(key=lambda x: x["session_start"])
    return sessions


async def fetch_one_llm(http_session, row, local_iteration, output_file, file_lock, session_key, ttl):
    prompt_token_ids = row["_prompt_token_ids"]
    answer_token_ids = encode_text(row.get(args.output_col, ""))
    payload = {
        "model": model,
        "prompt": prompt_token_ids,
        "max_tokens": 16000,
        "add_special_tokens": False,
        "std_output_tokens": answer_token_ids,
        "estimated_ttl": ttl,
    }
    if session_key is not None:
        payload["program_id"] = session_key
    headers = {
        "Content-Type": "application/json"
    }
    try:
        async with http_session.post(API_URL, json=payload, headers=headers) as response:
            response_json = await response.json()
    except Exception as exc:
        response_json = {
            "error": "request_failed",
            "exception": repr(exc),
        }
    response_json.update(
        {
            "instance_id": row.get(args.instance_col),
            "iteration": row.get(args.iteration_col),
            "local_iteration": local_iteration,
        }
    )
    async with file_lock:
        output_file.write(json.dumps(response_json, ensure_ascii=False) + "\n")
        output_file.flush()
    return response_json


async def replay_one_session(session_info, output_file, file_lock, global_t0_mono):
    now = time.monotonic() - global_t0_mono
    wait = session_info["session_start"] - now
    if wait > 0:
        await asyncio.sleep(wait)
    rows = session_info["rows"]
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as http_session:
        for local_iteration, row in enumerate(rows):
            tool_sleep = row.get("_tool_sleep_seconds", 0.0)
            ttl = tool_sleep
            await fetch_one_llm(
                http_session=http_session,
                row=row,
                local_iteration=local_iteration,
                output_file=output_file,
                file_lock=file_lock,
                session_key=session_info["session_key"],
                ttl=ttl,
            )
            if local_iteration == len(rows) - 1:
                break
            if tool_sleep > 0:
                await asyncio.sleep(tool_sleep)


async def schedule_sessions(output_file):
    file_lock = asyncio.Lock()
    df = load_requests(args.input_file)
    sessions = build_sessions(df)
    global_t0_mono = time.monotonic()
    tasks = [
        asyncio.create_task(
            replay_one_session(
                session_info=session_info,
                output_file=output_file,
                file_lock=file_lock,
                global_t0_mono=global_t0_mono,
            )
        )
        for session_info in sessions
    ]
    for task in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        await task


if __name__ == "__main__":
    with open(args.output_file, "a", encoding="utf-8") as output_file:
        asyncio.run(schedule_sessions(output_file))