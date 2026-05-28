import os
import re
import json
import tqdm
import time
import random
import asyncio
import aiohttp
import argparse
import pandas as pd
from transformers import AutoTokenizer


model = "/jizhicfs/marvinhjia/sca/models/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b"
vllm_host_ip = os.popen("hostname -I | awk '{print $1}'").read().strip()
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=150 * 60 * 60)
API_URL = "http://127.0.0.1:8156/v1/completions"
tokenizer = AutoTokenizer.from_pretrained(model)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_file", "-i", type=str, default="./data/2x_bfcl_web_search_prompts_qps3.0.csv")
    parser.add_argument("--output_file","-o", type=str, default="./70B_results/2x_CONTINUUM_qps3.0/2x_CONTINUUM_replay_output_qps3.0.jsonl")
    # 输入文件字段名
    parser.add_argument("--timestamp_col", type=str, default="Timestamp")
    parser.add_argument("--sample_col", type=str, default="sample_id")
    parser.add_argument("--turn_col", type=str, default="turn_id")
    parser.add_argument("--iteration_col", type=str, default="iteration")
    parser.add_argument("--prompt_col", type=str, default="prompt_text")
    parser.add_argument("--output_col", type=str, default="output_text")
    # 两种工具的统计参数
    parser.add_argument("--fetch_std", type=float, default=0.47622)
    parser.add_argument("--search_std", type=float, default=1.68118)
    parser.add_argument("--fetch_mean", type=float, default=1.80425)
    parser.add_argument("--search_mean", type=float, default=4.22863)
    parser.add_argument("--min_tool_sleep", type=float, default=0.0)
    parser.add_argument("--max_tool_sleep", type=float, default=None)
    return parser.parse_args()


args = parse_args()
random.seed(args.seed)


def load_requests(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    required_cols = [
        args.timestamp_col,
        args.sample_col,
        args.iteration_col,
        args.prompt_col,
        args.output_col,
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    if args.turn_col not in df.columns:
        df[args.turn_col] = 0
    return df


def encode_text(text: str):
    return tokenizer.encode(str(text), add_special_tokens=False)


def generate_prompt_and_answer(prompt_text: str, output_text: str, prompt_token_ids=None):
    prompt = prompt_token_ids if prompt_token_ids is not None else encode_text(prompt_text)
    answer_token_ids = encode_text(output_text)
    return prompt, answer_token_ids


def normalize_tool_name(tool_name: str):
    if tool_name is None:
        return None
    tool_name = str(tool_name).strip()
    if not tool_name:
        return None
    if "search_engine_query" in tool_name:
        return "WebSearchAPI_search_engine_query"
    if "fetch_url_content" in tool_name:
        return "WebSearchAPI_fetch_url_content"
    return tool_name


# 从当前 LLM 的 output_text 解析工具调用列表
def parse_tool_names_from_output(output_text: str):
    if output_text is None:
        return []
    output_text = str(output_text)
    tool_names = []
    for m in re.finditer(r"Action:\s*([A-Za-z0-9_]+)", output_text):
        tool_name = normalize_tool_name(m.group(1))
        if tool_name:
            tool_names.append(tool_name)
    if not tool_names:
        if "fetch_url_content" in output_text:
            tool_names.append("WebSearchAPI_fetch_url_content")        
        if "search_engine_query" in output_text:
            tool_names.append("WebSearchAPI_search_engine_query")
    return tool_names


# 根据工具类型, 从对应正态分布采样执行时间
def sample_one_tool_sleep(tool_name: str, no_random: bool = False) -> float:
    tool_name = normalize_tool_name(tool_name)
    if tool_name == "WebSearchAPI_search_engine_query":
        val = args.search_mean if no_random else random.gauss(args.search_mean, args.search_std)
    elif tool_name == "WebSearchAPI_fetch_url_content":
        val = args.fetch_mean if no_random else random.gauss(args.fetch_mean, args.fetch_std)
    val = max(args.min_tool_sleep, val)
    if args.max_tool_sleep is not None:
        val = min(args.max_tool_sleep, val)
    return val


def sample_tools_sleep(tool_names, no_random):
    details = []
    for tool_name in tool_names:
        sleep_seconds = sample_one_tool_sleep(tool_name, no_random=no_random)
        details.append(
            {
                "tool_name": tool_name,
                "sleep_seconds": sleep_seconds,
            }
        )
    total_sleep = sum(x["sleep_seconds"] for x in details)
    return total_sleep


def get_session_key(row):
    return f"{row[args.sample_col]}:{row[args.turn_col]}"


def build_sessions(df: pd.DataFrame):
    sessions = []
    df = df.copy()
    df["_session_key"] = df.apply(get_session_key, axis=1)
    df["_iteration_int"] = df[args.iteration_col].astype(int)
    for session_key, group in df.groupby("_session_key", sort=False):
        group = group.sort_values("_iteration_int").reset_index(drop=True)
        first_ts = group.iloc[0][args.timestamp_col]
        if pd.isna(first_ts):
            raise ValueError(f"Session {session_key} has invalid first Timestamp")
        session_start = float(first_ts)
        rows = group.to_dict("records")
        prev_prompt_text = None
        prev_prompt_token_ids = None
        for row in rows:
            prompt_text = str(row[args.prompt_col])
            if prev_prompt_text is not None and prompt_text.startswith(prev_prompt_text):
                suffix_text = prompt_text[len(prev_prompt_text):]
                prompt_token_ids = prev_prompt_token_ids + encode_text(suffix_text)
            else:
                prompt_token_ids = encode_text(prompt_text)
            if prev_prompt_token_ids is not None and prompt_text.startswith(prev_prompt_text):
                if prompt_token_ids[:len(prev_prompt_token_ids)] != prev_prompt_token_ids:
                    raise ValueError(
                        f"Incremental token prefix failed for session={session_key}, "
                        f"iteration={row[args.iteration_col]}"
                    )
            row["_prompt_token_ids"] = prompt_token_ids
            prev_prompt_text = prompt_text
            prev_prompt_token_ids = prompt_token_ids
        sessions.append(
            {
                "session_key": session_key,
                "session_start": session_start,
                "rows": rows
            }
        )
    sessions.sort(key=lambda x: x["session_start"])
    return sessions


async def fetch_one_llm(http_session, row, local_iteration, output_file, file_lock, program_id=None, ttl=0):
    prompt_text = row[args.prompt_col]
    output_text = row[args.output_col]
    prompt, answer_token_ids = generate_prompt_and_answer(
        prompt_text,
        output_text,
        prompt_token_ids=row.get("_prompt_token_ids"),
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 16000,
        "add_special_tokens": False,
        "std_output_tokens": answer_token_ids,
        "estimated_ttl": ttl,
    }
    if program_id is not None:
        payload["program_id"] = program_id
    
    headers = {
        "Content-Type": "application/json",
    }
    try:
        async with http_session.post(API_URL, json=payload, headers=headers) as response:
            response_json = await response.json()
    except Exception as e:
        response_json = {
            "error": "request_failed",
            "exception": repr(e),
        }
    response_json.update({
        "sample_id": row.get(args.sample_col),
        "turn_id": row.get(args.turn_col),
        "iteration": row.get(args.iteration_col),
        "local_iteration": local_iteration,
    })
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
    # n_rows = len(rows)
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as http_session:
        for local_iteration, row in enumerate(rows):
            output_text = row.get(args.output_col)
            tool_names = parse_tool_names_from_output(output_text)
            ttl = sample_tools_sleep(tool_names, no_random=True)
            tool_sleep = sample_tools_sleep(tool_names, no_random=False)
            
            await fetch_one_llm(
                http_session=http_session,
                row=row,
                local_iteration=local_iteration,
                output_file=output_file,
                file_lock=file_lock,
                program_id=session_info["session_key"],
                ttl=ttl
            )
            
            
            if local_iteration == len(rows) - 1:
                break
            
            if not tool_names:
                print(
                    f"[WARN] No tool parsed from output_text: "
                    f"session={session_info['session_key']}, "
                    f"iteration={row.get(args.iteration_col)}"
                )
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
