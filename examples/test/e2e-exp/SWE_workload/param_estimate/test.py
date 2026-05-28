import json
import time
import uuid
import argparse
import requests
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
DEFAULT_API_URL = "http://127.0.0.1:8056/v1/completions"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", "-i", type=str, required=True)
    parser.add_argument("--model", default="/jizhicfs/marvinhjia/sca/models/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b")
    parser.add_argument("--api_url", type=str, default=DEFAULT_API_URL)
    parser.add_argument("--instance_col", type=str, default="instance_id")
    parser.add_argument("--iteration_col", type=str, default="iteration")
    parser.add_argument("--prompt_col", type=str, default="prompt_text")
    parser.add_argument("--output_col", type=str, default="output_text")
    parser.add_argument("--prompt_percentile", type=float, default=90)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=16000)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument(
        "--use_nonce",
        action="store_true",
        help="Prepend a unique nonce to avoid prefix-cache hits across benchmark runs.",
    )
    parser.add_argument(
        "--nonce_prefix",
        type=str,
        default="NO_CACHE_NONCE:",
        help="Text used to build the unique nonce prefix.",
    )
    return parser.parse_args()


def encode(tokenizer, text):
    if text is None or pd.isna(text):
        return []
    return tokenizer.encode(str(text), add_special_tokens=False)


def count_tokens(tokenizer, text):
    return len(encode(tokenizer, text))


def make_fixed_output_tokens(tokenizer, output_texts, target_len):
    token_ids = []
    for text in output_texts:
        token_ids.extend(encode(tokenizer, text))
        if len(token_ids) >= target_len:
            break
    if not token_ids:
        token_ids = encode(tokenizer, "Final Answer: done.\n")
    base = token_ids[:]
    while len(token_ids) < target_len:
        token_ids.extend(base)
    return token_ids[:target_len]


def make_prompt_token_ids(tokenizer, base_prompt_token_ids, args):
    if not args.use_nonce:
        return base_prompt_token_ids
    nonce_text = f"{args.nonce_prefix} {uuid.uuid4().hex}\n\n"
    nonce_token_ids = encode(tokenizer, nonce_text)
    # Preserve the target input length while changing the first tokens.
    return (nonce_token_ids + base_prompt_token_ids)[:len(base_prompt_token_ids)]


def post_once(api_url, model, prompt_token_ids, answer_token_ids, max_tokens, timeout):
    payload = {
        "model": model,
        "prompt": prompt_token_ids,
        "std_output_tokens": answer_token_ids,
        "max_tokens": max_tokens,
        "add_special_tokens": False,
    }
    t0 = time.monotonic()
    session = requests.Session()
    session.trust_env = False
    resp = session.post(api_url, json=payload, timeout=timeout)
    t1 = time.monotonic()
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}
    return t1 - t0, resp.status_code, body


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, use_fast=True,
    )
    df = pd.read_csv(args.input_csv, keep_default_na=False)
    for col in [args.instance_col, args.iteration_col, args.prompt_col, args.output_col]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    prompt_lens = df[args.prompt_col].apply(lambda x: count_tokens(tokenizer, x)).to_numpy()
    output_lens = df[args.output_col].apply(lambda x: count_tokens(tokenizer, x)).to_numpy()
    prompt_p = float(np.percentile(prompt_lens, args.prompt_percentile))
    output_mean = int(round(np.mean(output_lens)))
    idx = int(np.argmin(np.abs(prompt_lens - prompt_p)))
    base_prompt = str(df.iloc[idx][args.prompt_col])
    base_prompt_token_ids = encode(tokenizer, base_prompt)
    answer_token_ids = make_fixed_output_tokens(
        tokenizer=tokenizer,
        output_texts=df[args.output_col].tolist(),
        target_len=max(1, output_mean),
    )
    print(f"target_prompt_tokens: {prompt_p:.2f}")
    print(f"api_url: {args.api_url}")
    print(f"selected_row_index: {idx}")
    print(f"selected_instance_id: {df.iloc[idx][args.instance_col]}")
    print(f"selected_iteration: {df.iloc[idx][args.iteration_col]}")
    print(f"selected_prompt_tokens: {len(base_prompt_token_ids)}")
    print(f"target_output_tokens_mean: {output_mean}")
    print(f"std_output_tokens_len: {len(answer_token_ids)}")
    latencies = []
    total_runs = args.warmup + args.num_runs
    for run_id in range(total_runs):
        prompt_token_ids = make_prompt_token_ids(tokenizer, base_prompt_token_ids, args)
        latency, status, body = post_once(
            api_url=args.api_url,
            model=args.model,
            prompt_token_ids=prompt_token_ids,
            answer_token_ids=answer_token_ids,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        is_warmup = run_id < args.warmup
        print(
            f"{'[WARMUP]' if is_warmup else '[MEASURE]'} "
            f"run={run_id}, status={status}, "
            f"prompt_tokens={len(prompt_token_ids)}, latency={latency:.6f}s"
        )
        if status != 200:
            print("Response error preview:")
            print(json.dumps(body, ensure_ascii=False)[:1000])
        if status == 200 and not is_warmup:
            latencies.append(latency)
    if not latencies:
        raise RuntimeError("No successful measurement runs; check api_url/server/proxy settings.")
    print("=== No-cache latency result ===")
    print(f"num_measure_runs: {len(latencies)}")
    print(f"mean: {np.mean(latencies):.6f}s")
    print(f"median: {np.percentile(latencies, 50):.6f}s")
    print(f"p90: {np.percentile(latencies, 90):.6f}s")
    print(f"p95: {np.percentile(latencies, 95):.6f}s")
    print(f"p99: {np.percentile(latencies, 99):.6f}s")
    print(f"min: {np.min(latencies):.6f}s")
    print(f"max: {np.max(latencies):.6f}s")


if __name__ == "__main__":
    main()