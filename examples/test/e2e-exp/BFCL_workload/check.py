import argparse
import pandas as pd
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", "-i", default="bfcl_web_search_prompts_qps0.1.csv")
    parser.add_argument("--model", default="/jizhicfs/marvinhjia/sca/models/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b")
    parser.add_argument("--sample_col", default="sample_id")
    parser.add_argument("--turn_col", default="turn_id")
    parser.add_argument("--iteration_col", default="iteration")
    parser.add_argument("--prompt_col", default="prompt_text")
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--max_print", type=int, default=20)
    return parser.parse_args()


def common_prefix_len(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def first_char_mismatch(a: str, b: str):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else None


def show_text_context(a: str, b: str, pos: int, width: int = 120):
    left = max(0, pos - width)
    right = pos + width
    return {
        "pos": pos,
        "cur_context": repr(a[left:right]),
        "next_context": repr(b[left:right]),
    }


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    df = pd.read_csv(args.csv, keep_default_na=False)
    required = [
        args.sample_col,
        args.turn_col,
        args.iteration_col,
        args.prompt_col,
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    df[args.iteration_col] = df[args.iteration_col].astype(int)
    total_pairs = 0
    bad_text_prefix = []
    bad_incremental_token_prefix = []
    weak_incremental_block_reuse = []
    stats = []

    def encode(text):
        return tokenizer.encode(str(text), add_special_tokens=False)

    grouped = df.groupby([args.sample_col, args.turn_col], sort=False)
    for (sample_id, turn_id), group in grouped:
        group = group.sort_values(args.iteration_col).reset_index(drop=True)
        incremental_tokens = []
        prev_prompt = None
        prev_tokens = None
        for row_idx in range(len(group)):
            prompt = str(group.loc[row_idx, args.prompt_col])
            if prev_prompt is not None and prompt.startswith(prev_prompt):
                tokens = prev_tokens + encode(prompt[len(prev_prompt):])
            else:
                tokens = encode(prompt)
            incremental_tokens.append(tokens)
            prev_prompt = prompt
            prev_tokens = tokens
        for i in range(len(group) - 1):
            total_pairs += 1
            cur_iter = int(group.loc[i, args.iteration_col])
            next_iter = int(group.loc[i + 1, args.iteration_col])
            cur_prompt = str(group.loc[i, args.prompt_col])
            next_prompt = str(group.loc[i + 1, args.prompt_col])
            text_prefix_ok = next_prompt.startswith(cur_prompt)
            if not text_prefix_ok:
                pos = first_char_mismatch(cur_prompt, next_prompt)
                bad_text_prefix.append({
                    "sample_id": sample_id,
                    "turn_id": turn_id,
                    "iteration": cur_iter,
                    "next_iteration": next_iter,
                    "cur_chars": len(cur_prompt),
                    "next_chars": len(next_prompt),
                    "mismatch": show_text_context(cur_prompt, next_prompt, pos),
                })

            cur_tokens = incremental_tokens[i]
            next_tokens = incremental_tokens[i + 1]
            common_tokens = common_prefix_len(cur_tokens, next_tokens)
            token_prefix_ok = next_tokens[:len(cur_tokens)] == cur_tokens
            cur_full_blocks = len(cur_tokens) // args.block_size
            reusable_full_blocks = common_tokens // args.block_size
            block_reuse_ratio = (
                reusable_full_blocks / cur_full_blocks
                if cur_full_blocks > 0 else 1.0
            )
            item = {
                "sample_id": sample_id,
                "turn_id": turn_id,
                "iteration": cur_iter,
                "next_iteration": next_iter,
                "text_prefix_ok": text_prefix_ok,
                "incremental_token_prefix_ok": token_prefix_ok,
                "incremental_cur_tokens": len(cur_tokens),
                "incremental_next_tokens": len(next_tokens),
                "incremental_common_prefix_tokens": common_tokens,
                "incremental_lost_prefix_tokens": len(cur_tokens) - common_tokens,
                "incremental_cur_full_blocks": cur_full_blocks,
                "incremental_reusable_full_blocks": reusable_full_blocks,
                "incremental_block_reuse_ratio": block_reuse_ratio,
            }
            stats.append(item)
            if not token_prefix_ok:
                bad_incremental_token_prefix.append(item)
            if block_reuse_ratio < 0.99:
                weak_incremental_block_reuse.append(item)

    print("=== Overall ===")
    print(f"num_sessions: {grouped.ngroups}")
    print(f"total_adjacent_pairs: {total_pairs}")
    print(f"bad_text_prefix: {len(bad_text_prefix)}")
    print(f"bad_incremental_token_prefix: {len(bad_incremental_token_prefix)}")
    print(f"weak_incremental_block_reuse_ratio_lt_0.99: {len(weak_incremental_block_reuse)}")
    if stats:
        avg_lost = sum(x["incremental_lost_prefix_tokens"] for x in stats) / len(stats)
        avg_block_ratio = sum(x["incremental_block_reuse_ratio"] for x in stats) / len(stats)
        print(f"avg_incremental_lost_prefix_tokens: {avg_lost:.4f}")
        print(f"avg_incremental_block_reuse_ratio: {avg_block_ratio:.6f}")
    print()
    print("=== First bad text-prefix cases ===")
    for x in bad_text_prefix[:args.max_print]:
        print(x)
    print()
    print("=== First bad incremental token-prefix cases ===")
    for x in bad_incremental_token_prefix[:args.max_print]:
        print(x)
    print()
    print("=== First weak incremental block-reuse cases ===")
    for x in weak_incremental_block_reuse[:args.max_print]:
        print(x)


if __name__ == "__main__":
    main()
